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


class IsotropicViscoState():
    """Variables internes pour l'écrouissage isotrope viscoplastique."""

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # Déformation viscoplastique cumulée p
        self.eps_p_old = fem.Function(WT)  # Tenseur des déformations viscoplastiques εᵛᵖ
        self._W        = W
        self._WT       = WT


class IsotropicViscoPlasticModel(PlasticityModel):
    """
    Viscoplasticité isotrope (Type Perzyna) avec écrouissage non linéaire de Voce.

    Fonction de charge : f(σ, p) = σ_vm(σ) – σ_Y – Q·(1 – exp(–k·p))
    Loi d'écoulement :   Δεᵛᵖ = Δp · n,   n = (3/2) * s / σ_vm
    Loi visqueuse :      Δp = dt * < f(σ, p) / K_visco >^n_visco

    Parameters
    ----------
    elastic      : ElasticModel
    sigma_Y      : float – Limite d'élasticité initiale
    Q_var        : float – Contrainte de saturation (Voce)
    k_hardening  : float – Vitesse de saturation (Voce)
    K_visco      : float – Paramètre de résistance visqueuse
    n_visco      : float – Exposant de viscosité
    dt           : float – Pas de temps courant
    """

    def __init__(self, elastic: ElasticModel, sigma_Y: float, Q_var: float, k_hardening: float, 
                 K_visco: float, n_visco: float, dt: float):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.Q_var   = Q_var
        self.k       = k_hardening
        self.K_visco = K_visco
        self.n_visco = n_visco
        self.dt      = dt

    # -- internal helpers (UFL) --------------------------------------------
    def _von_mises(self, sigma):
        """Contrainte équivalente de Von Mises (Isotrope)."""
        s = sigma - (1.0 / 3.0) * ufl.tr(sigma) * ufl.Identity(self.elastic.tdim)
        return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-12)
    
    def _yield_func(self, sigma, p):
        # Écrouissage non linéaire de Voce identique à ton modèle Hill48
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self._von_mises(sigma) - self.sigma_Y - R

    def _flow_normal(self, sigma):
        """Normale à la surface de charge."""
        sigma_vm = self._von_mises(sigma)
        s = sigma - (1.0 / 3.0) * ufl.tr(sigma) * ufl.Identity(self.elastic.tdim)
        return (3.0 / (2.0 * sigma_vm)) * s

    # -- PlasticityModel interface -----------------------------------------
    def create_state(self, domain, W, WT) -> IsotropicViscoState:
        return IsotropicViscoState(W, WT)

    def update(self, state: IsotropicViscoState, eps) -> tuple:
        # Élastique prédit (Trial state)
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._yield_func(sigma_tr, state.p_old)
        n        = self._flow_normal(sigma_tr)

        # Dérivée de l'écrouissage non linéaire de Voce par rapport à p
        R_prime  = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)
        
        # PROTECTION UFL CONTRE LES NaN : 
        # Si f_val < 0, on applique une valeur fictive de 1.0 purement pour que
        # les puissances et leurs dérivées restent finies et calculables.
        f_safe = ufl.conditional(ufl.ge(f_val, 0.0), f_val / self.K_visco, 1.0)
        
        # Linéarisation de Newton locale (utilisant la valeur sécurisée f_safe)
        denom = 1.0 + (self.dt * self.n_visco / self.K_visco) * \
                (f_safe ** (self.n_visco - 1)) * (3.0 * self.elastic.mu + R_prime)
        
        # Résolution du Δp local théorique
        delta_p0 = (self.dt * (f_safe ** self.n_visco)) / denom

        # Application conditionnelle stricte : on n'active le flux que si f_val >= 0
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        delta_p     = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0, 0.0)
        
        return delta_p, delta_eps_p

    def commit(self, state: IsotropicViscoState, uh) -> None:
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


# ---------------------------------------------------------------------------
# Validation / Plot Utilities & Execution
# ---------------------------------------------------------------------------


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
        file_name   = "comparaison_j2_visco",
        E           = 200_000.0,
        nu          = 0.3,
        sigma_Y     = 100.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,        # Paramètre d'écrouissage de Voce
        K_visco     = 100.0,        # Paramètres spécifiques à la viscosité
        n_visco     = 4.0,
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
    # Simulation 2 : Modèle Viscoplastique Isotrope (Perzyna)
    # -----------------------------------------------------------------------
    print("\n--- Lancement de la simulation avec le modèle Viscoplastique ---")
    modèle_visco = IsotropicViscoPlasticModel(
        elastic=elastic_shared,
        sigma_Y=config["sigma_Y"],
        Q_var=config["Q_var"],
        k_hardening=config["k_hardening"],
        K_visco=config["K_visco"],
        n_visco=config["n_visco"],
        dt=dt_pas*0.0001
    )
    
    t0 = time()
    forces_visco, _ = run_simulation_relax(domain, V, W, WT, config=config, model=modèle_visco, num_supp_steps=200)
    print(f"Simulation Viscoplastique terminée en {time() - t0:.2f} secondes.")

    # -----------------------------------------------------------------------
    # Graphique de comparaison des réponses macroscopiques
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    
    # Courbe J2
    plt.plot(forces_j2, color='blue', linestyle='--', marker='o', markersize=4, 
             label="Plastique J2 classique (Statique)")
    
    # Courbe Viscoplastique
    plt.plot(forces_visco, color='purple', linestyle='-', marker='s', markersize=4, 
             label=f"Viscoplastique Perzyna (dt = {dt_pas:.3f}s)")
    
    plt.xlabel("Pas de temps", fontsize=11)
    plt.ylabel("Force de réaction macroscopique (N)", fontsize=11)
    plt.title("Comparaison de la réponse mécanique : J2 vs Viscoplastique", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Sauvegarde et affichage du graphique obtenu
    plt.savefig(os.path.join(config["output_dir"], "comparaison_modeles.png"), dpi=150)
    print(f"\nGraphique sauvegardé dans {config['output_dir']}/comparaison_modeles.png")
    plt.show()