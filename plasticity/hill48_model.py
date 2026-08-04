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



class Hill48state():
    """Internal variables for hill anisotropic hardening."""

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # cumulative plastic strain p
        self.eps_p_old = fem.Function(WT)  # plastic strain tensor εᵖ
        self._W        = W
        self._WT       = WT

class Hill48Model(PlasticityModel):
    """
        Hill48 anisotropic plasticity with Voce isotropic hardening.

    σ_hill48(σ) = sqrt(F(σ_22 - σ_33)² + G(σ_33 - σ_11)² + H(σ_11 - σ_22)² + 2Lσ_23² + 2Mσ_13² + 2Nσ_12²))
    Yield function:   f(σ, p) = σ_hill49(σ) – σ_Y – Q·(1 – exp(–k·p))
    Flow rule:        Δεᵖ = Δp · n,   n = ∂σ_hill48/∂σ
    Return mapping:   one linearised Newton step

    Parameters
    ----------
    elastic  : ElasticModel
    sigma_Y  : float – initial yield stress
    Q_var    : float – saturation stress (isotropic hardening)
    k        : float – hardening rate
    """

    def __init__(self, elastic: ElasticModel, sigma_Y: float, H: float,F: float,G: float,L: float,M: float,N: float,Q_var: float, k_hardening: float):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.H = H
        self.F = F
        self.G = G
        self.L = L
        self.M = M
        self.N = N
        self.Q_var = Q_var
        self.k = k_hardening

    # -- internal helpers (UFL) --------------------------------------------
    def _sigma_hill48(self, sigma):
        sigma_hill48 = ufl.sqrt(
            self.F * (sigma[1,1] - sigma[2,2])**2 +
            self.G * (sigma[2,2] - sigma[0,0])**2 +
            self.H * (sigma[0,0] - sigma[1,1])**2 + 
            self.L * (sigma[1,2]**2 + sigma[2,1]**2) + 
            self.M * (sigma[0,2]**2 + sigma[2,0]**2) + 
            self.N * (sigma[0,1]**2 + sigma[1,0]**2) + 
            1e-12 # évite sqrt(0)
        ) 
        return sigma_hill48
    
    def _yield_func(self, sigma, p):
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self._sigma_hill48(sigma) - self.sigma_Y - R

    def _flow_normal_ufl(self, sigma):
        """Calcul de la normale avec déclaration explicite de la variable UFL"""
        s_smeared = sigma + 1e-10 * ufl.Identity(self.elastic.tdim)
        
        # On enregistre explicitement le tenseur comme variable UFL
        s_var = ufl.variable(s_smeared)
        
        # On exprime la fonction de Hill à partir de cette variable
        hill = self._sigma_hill48(s_var)
        
        # On dérive par rapport à cette variable spécifique
        return ufl.diff(hill, s_var)

    def _flow_normal(self, sigma):
        sig_hill = self._sigma_hill48(sigma) + 1e-10
        
        dPhi_00 = 2.0 * self.G * (sigma[0,0] - sigma[2,2]) + 2.0 * self.H * (sigma[0,0] - sigma[1,1])
        dPhi_11 = 2.0 * self.F * (sigma[1,1] - sigma[2,2]) + 2.0 * self.H * (sigma[1,1] - sigma[0,0])
        dPhi_22 = 2.0 * self.F * (sigma[2,2] - sigma[1,1]) + 2.0 * self.G * (sigma[2,2] - sigma[0,0])
        
        dPhi_01 = 2.0 * self.N * sigma[0,1]
        dPhi_10 = 2.0 * self.N * sigma[1,0]
        dPhi_02 = 2.0 * self.M * sigma[0,2]
        dPhi_20 = 2.0 * self.M * sigma[2,0]
        dPhi_12 = 2.0 * self.L * sigma[1,2]
        dPhi_21 = 2.0 * self.L * sigma[2,1]
        
        dPhi = ufl.as_tensor([
            [dPhi_00, dPhi_01, dPhi_02],
            [dPhi_10, dPhi_11, dPhi_12],
            [dPhi_20, dPhi_21, dPhi_22]
        ])
        
        return (1.0 / (2.0 * sig_hill)) * dPhi

    # -- PlasticityModel interface -----------------------------------------
    def create_state(self, domain, W, WT) -> Hill48state:
        return Hill48state(W, WT)

    def update(self, state: Hill48state, eps) -> tuple:
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._yield_func(sigma_tr, state.p_old)
        
        # Normale cohérente avec Hill48
        n = self._flow_normal(sigma_tr)

        # Calcul correct de f_prime pour Hill48 via les modules élastiques généralisés
        # Pour une convergence stricte, une écriture de Newton-Raphson locale complète est requise.
        # Approximation locale de la raideur plastique :
        R_prime  = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)
        
        E_n = ufl.inner(n, self.elastic.sigma(n))
        f_prime = -R_prime - E_n
        
        delta_p0 = -(1.0 / f_prime) * f_val

        # Activation du critère
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        delta_p     = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0, 0.0)
        
        return delta_p, delta_eps_p

    def commit(self, state: Hill48state, uh) -> None:
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


hill_params = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name    = "donnes_ref",
    # Elastic constants (used when no model is supplied)
    E           = 200_000.0,
    nu          = 0.3,
    # J2 isotropic hardening parameters (used when no model is supplied)
    sigma_Y     = 100.0,
    Q_var       = 50.0,
    k_hardening = 1_000.0,
    F = 0.900,  # Anisotropie dans le plan transverse
    G = 0.600,  # Anisotropie dans le plan longitudinal
    H = 0.400,  # Terme d'interaction (souvent proche de 0.5)
    L = 1.7,  # Cisaillement hors-plan (souvent supposé isotrope = 1.5)
    M = 1.3,  # Cisaillement hors-plan (souvent supposé isotrope = 1.5)
    N = 1.350 
    
)

modèle_hill48 = Hill48Model(
    elastic=ElasticModel(hill_params["E"], hill_params["nu"], tdim=3),
    sigma_Y=hill_params["sigma_Y"],
    H=hill_params["H"],
    F=hill_params["F"],
    G=hill_params["G"],
    L=hill_params["L"],
    M=hill_params["M"],
    N=hill_params["N"],
    Q_var=hill_params["Q_var"],
    k_hardening=hill_params["k_hardening"]
)


def plot_hill_surface(params, sigma_y_val=None):
    """
    Trace la surface de charge de Hill48 vs Von Mises dans le plan (sigma_11, sigma_22)
    en supposant les contraintes de cisaillement nulles.
    """
    if sigma_y_val is None:
        sigma_y_val = params["sigma_Y"]

    # Création d'une grille de contraintes
    s_max = sigma_y_val * 1.5
    s11, s22 = np.meshgrid(np.linspace(-s_max, s_max, 200), 
                           np.linspace(-s_max, s_max, 200))
    
    # Calcul de la contrainte équivalente de Hill48
    # En 2D (plan), sigma_33 = 0, donc :
    # hill = sqrt(F*s22**2 + G*s11**2 + H*(s11-s22)**2)
    F, G, H = params["F"], params["G"], params["H"]
    hill_sq = F * (s22)**2 + G * (-s11)**2 + H * (s11 - s22)**2
    hill_val = np.sqrt(hill_sq)
    
    # Calcul de Von Mises pour comparaison (F=G=H=0.5)
    vm_val = np.sqrt(0.5 * (s22**2 + s11**2 + (s11 - s22)**2))

    # Plot
    plt.figure(figsize=(8, 7))
    plt.contour(s11, s22, hill_val, levels=[sigma_y_val], colors='red', linewidths=2)
    plt.contour(s11, s22, vm_val, levels=[sigma_y_val], colors='blue', linestyles='dashed')
    
    # Esthétique
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel(r'$\sigma_{11}$ (MPa)')
    plt.ylabel(r'$\sigma_{22}$ (MPa)')
    plt.title(f"Surface de charge : Hill48 (Rouge) vs Von Mises (Bleu)" +f"\n(F={F}, G={G}, H={H})" )
    plt.legend(['Hill48', 'Von Mises'], loc='upper right')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Utilisation :

if __name__ == "__main__":
    config = dict(
        t_start     = 0.0,
        T           = 3.0,
        num_steps   = 50,
        load_amp    = 0.025,       # amplitude of the applied displacement
        length      = 10.0,       # half-length of the specimen
        mesh_file   = "carre_trou.msh",
        output_dir  = "results_plasticity",
        file_name    = "carre_trou_ortho",
        # Elastic constants (used when no model is supplied)
        E           = 200_000.0,
        nu          = 0.3,
        # J2 isotropic hardening parameters (used when no model is supplied)
        sigma_Y     = 100.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,
        F = 0.250,  # AnisotrF,G,H,N,σ0,Q,k]opie dans le plan transverse
        G = 0.3500,  # Anisotropie dans le plan longitudinal
        H = 0.6500,  # Terme d'interaction (souvent proche
        L = 1.5,  # Cisaillement hors-plan
        M = 1.5,  # Cisaillement hors-plan
        N = 1.450
    )
    modèle_hill48 = Hill48Model(
        elastic=ElasticModel(config["E"], config["nu"], tdim=3),
        sigma_Y=config["sigma_Y"],
        H=config["H"],
        F=config["F"],
        G=config["G"],
        L=config["L"],
        M=config["M"],
        N=config["N"],
        Q_var=config["Q_var"],
        k_hardening=config["k_hardening"]
    )
    domain = load_and_write_mesh(config["mesh_file"])

    V, W, WT = build_function_spaces(domain)
    from time import time
    start_time = time()
    forces, _ = run_simulation_write(domain,V,W,WT,config=config, model=modèle_hill48)
    np.save("forces_anisotrope_carre_trou.npy", forces)
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