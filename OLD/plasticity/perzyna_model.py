from plasticity_simu import *

import os
from abc import ABC, abstractmethod

import meshio
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver



class NortonVPstate():
    """Internal variables for isotropic Norton viscoplastic flow."""

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # cumulative viscoplastic strain p
        self.eps_p_old = fem.Function(WT)  # viscoplastic strain tensor εᵛᵖ
        self._W        = W
        self._WT       = WT

class NortonViscoplasticModel(PlasticityModel):
    """
        Viscoplasticité isotrope de Norton (loi puissance, von Mises).

    σ_eq(σ) = sqrt(3/2 s:s),   s = dev(σ)          (contrainte équivalente de von Mises)
    Fonction de surtension :  f(σ, p) = σ_eq(σ) – σ_Y – Q·(1 – exp(–k·p))
    Loi d'écoulement (loi de Norton, puissance) :
        ṗ = (⟨f⟩ / K)^N ,      K = consistance visqueuse, N = sensibilité à la vitesse
        ε̇ᵖ = ṗ · n,            n = ∂σ_eq/∂σ = (3/2)·s/σ_eq   (écoulement associé)
    Intégration temporelle :  schéma implicite (Euler retour). La loi étant non
    linéaire en Δp (puissance N), on procède comme pour Hill48/Maxwell à un seul
    pas de Newton local linéarisé autour de Δp = 0, ce qui donne une valeur
    approchée de Δp cohérente avec le couplage élastique et l'écrouissage.
    (Pour une convergence stricte, une boucle de Newton locale complète serait
    nécessaire lorsque N s'écarte fortement de 1.)

    Parameters
    ----------
    elastic     : ElasticModel
    sigma_Y     : float – seuil de viscoplasticité initial
    K           : float – consistance visqueuse de Norton (MPa·s^(1/N))
    N           : float – exposant de Norton (sensibilité à la vitesse, N ≥ 1)
    Q_var       : float – contrainte de saturation (écrouissage isotrope)
    k_hardening : float – vitesse d'écrouissage
    dt          : float – pas de temps (indispensable pour une loi visqueuse)
    """

    def __init__(self, elastic: ElasticModel, sigma_Y: float, K: float, N: float,
                 Q_var: float, k_hardening: float, dt: float):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.K = K
        self.N = N
        self.Q_var = Q_var
        self.k = k_hardening
        self.dt = dt

    # -- internal helpers (UFL) --------------------------------------------
    def _sigma_eq(self, sigma):
        """Contrainte équivalente de von Mises (isotrope)."""
        Id = ufl.Identity(self.elastic.tdim)
        s = sigma - (1.0 / 3.0) * ufl.tr(sigma) * Id
        sigma_eq = ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-12)  # évite sqrt(0)
        return sigma_eq

    def _overstress_func(self, sigma, p):
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self._sigma_eq(sigma) - self.sigma_Y - R

    def _flow_normal_ufl(self, sigma):
        """Calcul de la normale avec déclaration explicite de la variable UFL"""
        s_smeared = sigma + 1e-10 * ufl.Identity(self.elastic.tdim)

        # On enregistre explicitement le tenseur comme variable UFL
        s_var = ufl.variable(s_smeared)

        # On exprime la contrainte équivalente à partir de cette variable
        sigma_eq = self._sigma_eq(s_var)

        # On dérive par rapport à cette variable spécifique
        return ufl.diff(sigma_eq, s_var)

    def _flow_normal(self, sigma):
        """Normale analytique (associée) : n = (3/2) s / σ_eq."""
        Id = ufl.Identity(self.elastic.tdim)
        s = sigma - (1.0 / 3.0) * ufl.tr(sigma) * Id
        sigma_eq = self._sigma_eq(sigma)
        return 1.5 * s / sigma_eq

    # -- PlasticityModel interface -----------------------------------------
    def create_state(self, domain, W, WT) -> NortonVPstate:
        return NortonVPstate(W, WT)

    def update(self, state: NortonVPstate, eps) -> tuple:
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._overstress_func(sigma_tr, state.p_old)

        # Surtension positive uniquement (crochets de Macaulay) : <f> = max(f, 0)
        f_pos = ufl.conditional(ufl.ge(f_val, 0.0), f_val, 0.0)

        # Normale associée à la surface de von Mises
        n = self._flow_normal(sigma_tr)

        # Résidu local (implicite) : g(Δp) = Δp - Δt·((f_trial - (R'+E_n)Δp)/K)^N
        # Un pas de Newton local autour de Δp = 0 donne :
        #   g(0)  = -Δt·(f_pos/K)^N
        #   g'(0) =  1 + Δt·N·(R'+E_n)/K · (f_pos/K)^(N-1)
        #   Δp    = -g(0)/g'(0)
        R_prime = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)
        E_n     = ufl.inner(n, self.elastic.sigma(n))

        base    = f_pos / self.K
        # base**(N-1) sécurisé au voisinage de f_pos = 0
        base_pow_Nm1 = base**(self.N - 1.0)
        base_pow_N   = base * base_pow_Nm1

        g0       = self.dt * base_pow_N
        g_prime  = 1.0 + self.dt * self.N * (R_prime + E_n) / self.K * base_pow_Nm1

        delta_p0 = g0 / g_prime

        # Activation du critère (écoulement seulement si surtension positive)
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        delta_p     = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0, 0.0)

        return delta_p, delta_eps_p

    def commit(self, state: NortonVPstate, uh) -> None:
        eps                  = self.elastic.epsilon(uh)
        delta_p, delta_eps_p = self.update(state, eps)

        state.p_old.interpolate(
            fem.Expression(state.p_old + delta_p, state._W.element.interpolation_points)
        )
        delta_eps_p_proj = fem.Function(state._WT)
        delta_eps_p_proj.interpolate(
            fem.Expression(delta_eps_p, state._WT.element.interpolation_points)
        )
        state.eps_p_old.x.array[:] += delta_eps_p_proj.x.array[:]


norton_params = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name   = "donnes_ref_norton",
    # Elastic constants (used when no model is supplied)
    E           = 200_000.0,
    nu          = 0.3,
    # Viscoplastic / hardening parameters
    sigma_Y     = 100.0,
    K           = 200.0,      # consistance visqueuse de Norton (MPa·s^(1/N))
    N           = 5.0,        # exposant de Norton (sensibilité à la vitesse)
    Q_var       = 50.0,
    k_hardening = 1_000.0,
)
# pas de temps déduit de T / num_steps, requis par la loi de Norton
norton_params["dt"] = norton_params["T"] / norton_params["num_steps"]

modèle_norton = NortonViscoplasticModel(
    elastic=ElasticModel(norton_params["E"], norton_params["nu"], tdim=3),
    sigma_Y=norton_params["sigma_Y"],
    K=norton_params["K"],
    N=norton_params["N"],
    Q_var=norton_params["Q_var"],
    k_hardening=norton_params["k_hardening"],
    dt=norton_params["dt"],
)


def plot_norton_flow_law(params, p_dot_max=None, n_points=200, N_values=None):
    """
    Trace la loi puissance de Norton : σ_over = K · ṗ^(1/N)
    C'est l'équivalent, pour ce modèle visqueux non linéaire, de la surface de
    charge tracée par plot_hill_surface (Hill48) et de la loi linéaire tracée
    par plot_maxwell_overstress (Maxwell) : ici on visualise l'effet de
    l'exposant N sur la relation contrainte de surtension <-> vitesse
    de déformation viscoplastique.
    """
    K = params["K"]
    N_ref = params["N"]
    if N_values is None:
        N_values = sorted(set([1.0, N_ref, 3.0 * N_ref]))
    if p_dot_max is None:
        p_dot_max = 5.0 * (params["sigma_Y"] / K) ** N_ref

    p_dot = np.linspace(1e-8, p_dot_max, n_points)

    plt.figure(figsize=(7, 6))
    for N in N_values:
        sigma_over = K * p_dot ** (1.0 / N)
        plt.plot(p_dot, sigma_over, lw=2, label=fr'$N={N:.1f}$')

    plt.xlabel(r'$\dot p$ (s$^{-1}$)')
    plt.ylabel(r'$\sigma_{over}$ (MPa)')
    plt.title(f"Loi de Norton : σ_over = K·ṗ^(1/N)  (K={K})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# Utilisation :

if __name__ == "__main__":
    config = dict(
        t_start     = 0.0,
        T           = 3.0,
        num_steps   = 50,
        load_amp    = 0.01,       # amplitude of the applied displacement
        length      = 10.0,       # half-length of the specimen
        mesh_file   = "carre_trou.msh",
        output_dir  = "results_plasticity",
        file_name   = "carre_trou_norton_isotrope",
        # Elastic constants (used when no model is supplied)
        E           = 200_000.0,
        nu          = 0.3,
        # Viscoplastic / hardening parameters
        sigma_Y     = 100.0,
        K           = 200.0,
        N           = 5.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,
    )
    config["dt"] = config["T"] / config["num_steps"]

    modèle_norton = NortonViscoplasticModel(
        elastic=ElasticModel(config["E"], config["nu"], tdim=3),
        sigma_Y=config["sigma_Y"],
        K=config["K"],
        N=config["N"],
        Q_var=config["Q_var"],
        k_hardening=config["k_hardening"],
        dt=config["dt"],
    )
    domain = load_and_write_mesh(config["mesh_file"])

    V, W, WT = build_function_spaces(domain)
    from time import time
    start_time = time()
    forces, _ = run_simulation_V3(domain, V, W, WT, config=config, model=modèle_norton)
    end_time = time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print("pas de soucis la team")
    plt.figure()
    plt.plot(forces)
    plt.xlabel("Time step")
    plt.ylabel("Reaction force")
    plt.title("Reaction force vs. time step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()