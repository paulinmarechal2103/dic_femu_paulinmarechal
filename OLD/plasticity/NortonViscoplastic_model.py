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
        # 1. Contrainte de l'état d'essai (Trial state)
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        sigma_eq = self._sigma_eq(sigma_tr)
        
        # 2. Écrouissage au début du pas
        R = self.Q_var * (1.0 - ufl.exp(-self.k * state.p_old))
        
        # 3. Calcul de la surtension (Macaulay brackets)
        f_val = sigma_eq - self.sigma_Y - R
        f_pos = ufl.conditional(ufl.ge(f_val, 0.0), f_val, 0.0)
        
        # 4. Normale d'écoulement (sécurisée contre la division par zéro)
        Id = ufl.Identity(self.elastic.tdim)
        s = sigma_tr - (1.0 / 3.0) * ufl.tr(sigma_tr) * Id
        n = 1.5 * s / (sigma_eq + 1e-12)
        
        # 5. Loi de Norton directe (Schéma semi-implicite)
        # Delta_p = dt * (<f> / K)^N
        delta_p = self.dt * (f_pos / self.K)**self.N
        delta_eps_p = delta_p * n
        
        return delta_p, delta_eps_p

    def commit(self, state: NortonVPstate, uh) -> None:
        eps = self.elastic.epsilon(uh)
        delta_p, delta_eps_p = self.update(state, eps)
        
        # On prépare les expressions des NOUVELLES valeurs globales à affecter
        # Cela évite les incrémentations floues en mémoire.
        new_p_expr = fem.Expression(state.p_old + delta_p, state._W.element.interpolation_points)
        new_eps_p_expr = fem.Expression(state.eps_p_old + delta_eps_p, state._WT.element.interpolation_points)
        
        # On met à jour proprement les fonctions FEniCSx
        state.p_old.interpolate(new_p_expr)
        state.eps_p_old.interpolate(new_eps_p_expr)


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

# Utilisation : comparaison J2 classique (statique) vs Norton (visqueux)

if __name__ == "__main__":
    # 1. Configuration commune des simulations
    config = dict(
        t_start     = 0.0,
        T           = 3.0,
        num_steps   = 50,
        load_amp    = 0.01,       # Amplitude augmentée pour bien entrer en plasticité
        length      = 10.0,
        mesh_file   = "Flat_specimen_refined.msh",
        output_dir  = "results_plasticity",
        file_name   = "comparaison_j2_norton",
        E           = 200_000.0,
        nu          = 0.3,
        sigma_Y     = 100.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,        # Paramètre d'écrouissage de Voce
        K           = 100.0,        # Paramètres spécifiques à la loi de Norton
        N           = 4.0,
    )
    # Calcul du pas de temps imposé pour le modèle visqueux
    dt_pas = (config["T"] - config["t_start"]) / config["num_steps"]

    # Chargement unique du maillage et construction des espaces fonctionnels
    domain = load_and_write_mesh(config["mesh_file"])
    V, W, WT = build_function_spaces(domain)
    elastic_shared = ElasticModel(config["E"], config["nu"], tdim=domain.topology.dim)

    # -----------------------------------------------------------------------
    # Simulation 1 : Modèle J2 Plastique Isotrope Classique (indépendant du temps)
    # -----------------------------------------------------------------------
    print("\n--- Lancement de la simulation avec le modèle J2 classique ---")
    modèle_j2 = J2IsotropicHardening(
        elastic=elastic_shared,
        sigma_Y=config["sigma_Y"],
        Q_var=config["Q_var"],
        k=config["k_hardening"]
    )
    from time import time
    t0 = time()
    forces_j2, _ = run_simulation_relax(domain, V, W, WT, config=config, model=modèle_j2, num_supp_steps=200)
    print(f"Simulation J2 terminée en {time() - t0:.2f} secondes.")

    # -----------------------------------------------------------------------
    # Simulation 2 : Modèle Viscoplastique Isotrope de Norton (loi puissance)
    # -----------------------------------------------------------------------
    print("\n--- Lancement de la simulation avec le modèle Viscoplastique de Norton ---")
    modèle_norton = NortonViscoplasticModel(
        elastic=elastic_shared,
        sigma_Y=config["sigma_Y"],
        Q_var=config["Q_var"],
        k_hardening=config["k_hardening"],
        K=config["K"],
        N=config["N"],
        dt=dt_pas * 0.0001
    )
    t0 = time()
    forces_norton, _ = run_simulation_relax(domain, V, W, WT, config=config, model=modèle_norton, num_supp_steps=200)
    print(f"Simulation Viscoplastique de Norton terminée en {time() - t0:.2f} secondes.")

    # -----------------------------------------------------------------------
    # Graphique de comparaison des réponses macroscopiques
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    # Courbe J2
    plt.plot(forces_j2, color='blue', linestyle='--', marker='o', markersize=4,
              label="Plastique J2 classique (Statique)")
    # Courbe Viscoplastique de Norton
    plt.plot(forces_norton, color='darkorange', linestyle='-', marker='s', markersize=4,
              label=f"Viscoplastique Norton (K={config['K']}, N={config['N']}, dt = {dt_pas:.3f}s)")

    plt.xlabel("Pas de temps", fontsize=11)
    plt.ylabel("Force de réaction macroscopique (N)", fontsize=11)
    plt.title("Comparaison de la réponse mécanique : J2 vs Viscoplastique de Norton", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Sauvegarde et affichage du graphique obtenu
    plt.savefig(os.path.join(config["output_dir"], "comparaison_modeles.png"), dpi=150)
    print(f"\nGraphique sauvegardé dans {config['output_dir']}/comparaison_modeles.png")
    plt.show()