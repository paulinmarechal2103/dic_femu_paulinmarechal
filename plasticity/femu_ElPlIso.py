"""
femu.py
=======
Finite Element Model Updating (FEMU) for J2 plasticity parameter identification.

Modèle cible (Voce isotrope) :
    f(σ, p) = J(σ) – σ_Y – Q·(1 – exp(–k·p)) = 0
    Paramètres : θ = [E, ν, σ_Y, Q, k]

Méthode :
    • FEMU-Epsilon : résidu sur le tenseur des déformations ||Eps_sim(θ) – Eps_ref||

Le fichier XDMF de référence est supposé généré par run_simulation_V2()
(qui stocke Epsilon dans le HDF5).

Utilisation rapide :
    python femu.py
"""
import os

import sys


# Redirection absolue de tous les caches vers le SSD local

os.environ["TMPDIR"] = "/tmp/pmarechal_tmp"

os.environ["FFCX_CACHE_DIR"] = "/tmp/pmarechal_fenics_cache_v3"

os.environ["DIALECT_DILL_CACHE_DIR"] = "/tmp/pmarechal_fenics_cache_v3"

os.environ["DOLFINX_JIT_TIMEOUT"] = "300" # 10 minutes d'attente max !



# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, OptimizeResult

import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, log

# Import depuis le module de simulation
from plasticity_simu import (
    ElasticModel,
    J2IsotropicHardening,
    PlasticityModel,
    build_function_spaces,
    load_and_write_mesh,
    build_right_facet_tag,
    run_simulation_V2,
    DEFAULT_CONFIG,
)

# Silence les logs PETSc/DOLFIN pendant l'optimisation
log.set_log_level(log.LogLevel.ERROR)


# ===========================================================================
# Noms canoniques des paramètres
# ===========================================================================
PARAM_NAMES = ["E", "nu", "sigma_Y", "Q", "k"]


# ===========================================================================
# Chargement des données de référence (Epsilon)
# ===========================================================================
@dataclass
class ReferenceData:
    """
    Données expérimentales / synthétiques de référence chargées depuis le fichier H5.

    Attributes
    ----------
    epsilons : np.ndarray
        Tenseurs Epsilon cibles à chaque pas de temps (macroscopiques ou champs aplatis).
    h5_path : str
        Chemin du fichier HDF5 source.
    """
    epsilons: np.ndarray
    h5_path: str

    @classmethod
    def load(cls, xdmf_path: str, eps_base: str = "Function/Epsilon") -> "ReferenceData":
        """
        Charge Epsilon depuis le fichier H5 associé à l'XDMF.
        """
        h5_path = str(Path(xdmf_path).with_suffix(".h5"))
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"Fichier HDF5 introuvable : {h5_path}\n"
                "Lancez d'abord run_simulation_V2(write_output=True) "
                "pour générer les données de référence."
            )

        with h5py.File(h5_path, "r") as f:
            # --- Tenseurs Epsilon ---
            # On tente d'abord de lire une valeur globale si elle existe
            if "Epsilon/value" in f:
                epsilons = f["Epsilon/value"][:]
            # Sinon, on lit pas à pas comme un champ (Function/epsilon)
            elif eps_base in f:
                step = 0
                eps_list = []
                while str(step) in f[eps_base]:
                    eps_list.append(f[f"{eps_base}/{step}"][:].ravel())
                    step += 1
                epsilons = np.array(eps_list)
            else:
                raise KeyError(
                    f"Impossible de trouver Epsilon dans le H5 (ni 'Epsilon/value', ni '{eps_base}').\n"
                    "Assurez-vous d'utiliser run_simulation_V2() qui écrit Epsilon dans le H5."
                )

        print(
            f"[ReferenceData] Chargé depuis {h5_path}\n"
            f"  • Epsilon      : {len(epsilons)} pas de temps"
        )
        return cls(epsilons=epsilons, h5_path=h5_path)


# ===========================================================================
# Evaluateur FEM (FEMU-Epsilon)
# ===========================================================================
class FEMUEvaluator:
    """
    Encapsule la simulation FEM et calcule le résidu FEMU pour un jeu de paramètres.
    """

    def __init__(self, ref: ReferenceData, config: dict, method: str = "FEMU-Epsilon"):
        self.ref    = ref
        self.config = {**DEFAULT_CONFIG, **config}
        self.method = method

        # Pré-chargement du maillage (coûteux, fait une seule fois)
        print("[FEMUEvaluator] Chargement du maillage…")
        self._domain = load_and_write_mesh(self.config["mesh_file"])
        self._V, self._W, self._WT = build_function_spaces(self._domain)
        self._ds = build_right_facet_tag(self._domain, self.config["length"])

        print(f"[FEMUEvaluator] Prêt (méthode={method})")

    # ------------------------------------------------------------------
    # Constructeur de modèle à partir du vecteur θ
    # ------------------------------------------------------------------
    def _make_model(self, theta: np.ndarray) -> J2IsotropicHardening:
        E, nu, sigma_Y, Q_var, k = theta
        elastic = ElasticModel(E=E, nu=nu, tdim=self._domain.topology.dim)
        return J2IsotropicHardening(elastic, sigma_Y=sigma_Y, Q_var=Q_var, k=k)

    # ------------------------------------------------------------------
    # Vérification des bornes physiques
    # ------------------------------------------------------------------
    @staticmethod
    def _is_physical(theta: np.ndarray) -> bool:
        E, nu, sigma_Y, Q_var, k = theta
        return (E > 0) and (0 < nu < 0.5) and (sigma_Y > 0) and (Q_var >= 0) and (k > 0)

    # ------------------------------------------------------------------
    # FEMU-Epsilon : résidu sur les déformations
    # ------------------------------------------------------------------
    def _cost_femu_epsilon(self, theta: np.ndarray) -> float:
        """
        J_E(θ) = ||Eps_sim(θ) – Eps_ref||² / ||Eps_ref||²

        Lance une simulation complète et récupère le 3e argument (Epsilon).
        """
        model = self._make_model(theta)
        try:
            # Récupération de Epsilon en 3e position
            _, _, eps_sim_raw = run_simulation_V2(
                config=self.config,
                model=model,
                write_output=False,
            )
        except Exception as exc:
            warnings.warn(f"Simulation échouée : {exc}")
            return 1e12

        # Aplatissement des tenseurs pour le calcul de la norme
        eps_sim = np.asarray(eps_sim_raw).reshape(len(eps_sim_raw), -1)
        eps_ref = np.asarray(self.ref.epsilons).reshape(len(self.ref.epsilons), -1)

        # Alignement en longueur (au cas où les pas diffèrent)
        n = min(len(eps_sim), len(eps_ref))
        diff     = eps_sim[:n] - eps_ref[:n]
        norm_ref = np.linalg.norm(eps_ref[:n])
        
        return float(np.sum(diff**2) / (norm_ref**2 + 1e-30))

    # ------------------------------------------------------------------
    # Point d'entrée unique
    # ------------------------------------------------------------------
    def __call__(self, theta: np.ndarray) -> float:
        """Calcule et retourne le coût J(θ)."""
        if not self._is_physical(theta):
            return 1e12
        if self.method == "FEMU-Epsilon":
            return self._cost_femu_epsilon(theta)
        else:
            raise ValueError(f"Méthode inconnue : {self.method!r}")


# ===========================================================================
# Problème FEMU (gestion de l'optimisation)
# ===========================================================================
@dataclass
class FEMUProblem:
    evaluator : FEMUEvaluator
    theta0    : np.ndarray
    bounds    : list
    scale     : np.ndarray = field(default_factory=lambda: np.ones(5))
    history   : list       = field(default_factory=list)

    def _scale(self, theta: np.ndarray) -> np.ndarray:
        return theta * self.scale

    def _unscale(self, theta_s: np.ndarray) -> np.ndarray:
        return theta_s / self.scale

    def _scaled_bounds(self):
        return [(lo * s, hi * s) for (lo, hi), s in zip(self.bounds, self.scale)]

    def objective(self, theta_s: np.ndarray) -> float:
        theta = self._unscale(theta_s)
        t0    = time.perf_counter()
        cost  = self.evaluator(theta)
        dt    = time.perf_counter() - t0

        record = dict(theta=theta.copy(), cost=cost, time=dt,
                      iter=len(self.history) + 1)
        self.history.append(record)

        params_str = "  ".join(
            f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, theta)
        )
        print(
            f"  [{record['iter']:3d}]  cost={cost:.5e}  ({dt:.1f}s)  "
            f"{params_str}"
        )
        return cost


# ===========================================================================
# Point d'entrée principal
# ===========================================================================
def run_femu_identification(
    xdmf_path     : str,
    theta0        : list,
    bounds        : list,
    config        : Optional[dict]  = None,
    method        : str             = "FEMU-Epsilon",
    optimizer     : str             = "Nelder-Mead",
    tol           : float           = 1e-6,
    maxiter       : int             = 300,
    true_params   : Optional[list]  = None,
) -> tuple[OptimizeResult, FEMUProblem]:
    
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    theta0_arr = np.asarray(theta0, dtype=float)
    scale      = 1.0 / (np.abs(theta0_arr) + 1e-30)

    print("=" * 65)
    print("  FEMU — Identification de paramètres (Basé sur Epsilon)")
    print("=" * 65)
    print(f"  Méthode    : {method}")
    print(f"  Optimiseur : {optimizer}")
    print(f"  Référence  : {xdmf_path}")
    print(f"  θ₀         : E={theta0[0]}, ν={theta0[1]}, "
          f"σ_Y={theta0[2]}, Q={theta0[3]}, k={theta0[4]}")
    if true_params:
        print(f"  θ_vrai     : E={true_params[0]}, ν={true_params[1]}, "
              f"σ_Y={true_params[2]}, Q={true_params[3]}, k={true_params[4]}")
    print("=" * 65)

    ref = ReferenceData.load(xdmf_path)
    evaluator = FEMUEvaluator(ref=ref, config=cfg, method=method)

    problem = FEMUProblem(
        evaluator=evaluator,
        theta0=theta0_arr,
        bounds=bounds,
        scale=scale,
    )

    print(f"\n[Optimisation] Démarrage ({optimizer})…\n")
    t_start = time.perf_counter()

    theta0_s        = problem._scale(theta0_arr)
    scaled_bounds   = problem._scaled_bounds()

    if optimizer == "differential_evolution":
        result = differential_evolution(
            problem.objective,
            bounds=scaled_bounds,
            seed=42,
            tol=tol,
            maxiter=maxiter,
            popsize=10,
            disp=False,
            workers=1,
            init="latinhypercube",
        )
    elif optimizer == "L-BFGS-B":
        result = minimize(
            problem.objective,
            x0=theta0_s,
            method="L-BFGS-B",
            bounds=scaled_bounds,
            options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-7, "disp": False},
        )
    else:  # Nelder-Mead (défaut)
        result = minimize(
            problem.objective,
            x0=theta0_s,
            method="Nelder-Mead",
            options={
                "maxiter"  : maxiter,
                "xatol"    : tol,
                "fatol"    : tol,
                "adaptive" : True,
                "disp"     : False,
            },
        )

    elapsed = time.perf_counter() - t_start
    theta_id = problem._unscale(result.x)
    result.x = theta_id   # on écrase avec les vraies unités

    _print_results(theta0_arr, theta_id, result, true_params, elapsed)

    return result, problem


# ===========================================================================
# Utilitaires d'affichage
# ===========================================================================
def _print_results(
    theta0    : np.ndarray,
    theta_id  : np.ndarray,
    result    : OptimizeResult,
    true_params: Optional[list],
    elapsed   : float,
):
    header = f"{'Paramètre':>10}  {'θ₀':>12}  {'θ_id':>12}"
    if true_params:
        header += f"  {'θ_vrai':>12}  {'Erreur (%)':>10}"
    print("\n" + "=" * 65)
    print("  RÉSULTATS FEMU")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for i, name in enumerate(PARAM_NAMES):
        row = f"  {name:>8}  {theta0[i]:>12.4g}  {theta_id[i]:>12.4g}"
        if true_params:
            err = 100.0 * abs(theta_id[i] - true_params[i]) / (abs(true_params[i]) + 1e-30)
            row += f"  {true_params[i]:>12.4g}  {err:>9.3f}%"
        print(row)
    print("-" * 65)
    print(f"  Coût final : {result.fun:.6e}")
    print(f"  Convergé   : {result.success}")
    print(f"  Message    : {result.message}")
    print(f"  Nb éval.   : {result.nfev}")
    print(f"  Temps total: {elapsed:.1f} s")
    print("=" * 65)


def plot_femu_results(
    problem      : FEMUProblem,
    result       : OptimizeResult,
    ref          : ReferenceData,
    config       : dict,
    true_params  : Optional[list] = None,
    save_path    : Optional[str]  = None,
):
    cfg     = {**DEFAULT_CONFIG, **config}
    history = problem.history

    iters = [h["iter"] for h in history]
    costs = [h["cost"] for h in history]

    theta_id = result.x
    elastic_id = ElasticModel(E=theta_id[0], nu=theta_id[1], tdim=3)
    model_id   = J2IsotropicHardening(
        elastic_id, sigma_Y=theta_id[2], Q_var=theta_id[3], k=theta_id[4]
    )
    print("[plot] Simulation avec θ_id pour validation…")
    _, _, eps_id_raw = run_simulation_V2(config=cfg, model=model_id, write_output=False)

    theta0 = problem.theta0
    elastic0 = ElasticModel(E=theta0[0], nu=theta0[1], tdim=3)
    model0   = J2IsotropicHardening(
        elastic0, sigma_Y=theta0[2], Q_var=theta0[3], k=theta0[4]
    )
    print("[plot] Simulation avec θ₀ pour comparaison…")
    _, _, eps0_raw = run_simulation_V2(config=cfg, model=model0, write_output=False)

    # Calcul de la norme d'Epsilon pour l'affichage (L2 de chaque pas de temps)
    eps_ref_norm = [np.linalg.norm(e) for e in ref.epsilons]
    eps_id_norm  = [np.linalg.norm(e) for e in eps_id_raw]
    eps0_norm    = [np.linalg.norm(e) for e in eps0_raw]

    steps_ref = np.arange(len(eps_ref_norm))
    steps_id  = np.arange(len(eps_id_norm))
    steps0    = np.arange(len(eps0_norm))

    ncols = 3 if true_params else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    # ---- 1. Convergence ----
    ax = axes[0]
    ax.semilogy(iters, costs, "b.-", lw=1.2, markersize=4)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Coût J(θ)")
    ax.set_title("Convergence FEMU")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # ---- 2. Norme d'Epsilon ----
    ax = axes[1]
    ax.plot(steps_ref, eps_ref_norm, "k-",  lw=2,   label="Référence")
    ax.plot(steps0,    eps0_norm,    "r--", lw=1.5, label="θ₀ (initial)")
    ax.plot(steps_id,  eps_id_norm,  "g-",  lw=1.5, label="θ_id (identifié)")
    ax.set_xlabel("Pas de temps")
    ax.set_ylabel("||Epsilon||")
    ax.set_title("Epsilon : référence vs identifié")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # ---- 3. Erreur relative par paramètre ----
    if true_params and ncols == 3:
        ax = axes[2]
        true_arr = np.asarray(true_params)
        errs_0   = 100.0 * np.abs(theta0   - true_arr) / (np.abs(true_arr) + 1e-30)
        errs_id  = 100.0 * np.abs(theta_id - true_arr) / (np.abs(true_arr) + 1e-30)
        x = np.arange(len(PARAM_NAMES))
        w = 0.35
        ax.bar(x - w/2, errs_0,  w, label="θ₀",     color="tomato",   alpha=0.8)
        ax.bar(x + w/2, errs_id, w, label="θ_id",   color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_NAMES)
        ax.set_ylabel("Erreur relative (%)")
        ax.set_title("Erreur d'identification par paramètre")
        ax.legend()
        ax.grid(True, axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Figure sauvegardée → {save_path}")
    plt.show()


# ===========================================================================
# Script de démonstration (cas synthétique)
# ===========================================================================
if __name__ == "__main__":

    TRUE_PARAMS = dict(
        E        = 200_000.0,
        nu       = 0.3,
        sigma_Y  = 100.0,
        Q_var    = 50.0,
        k        = 10.0,
    )

    REF_CONFIG = dict(
        mesh_file   = "Flat_specimen_refined.msh",
        output_dir  = "femu_files",
        file_name   = "res",
        num_steps   = 30,
        T           = 3.0,
        load_amp    = 0.01,
        length      = 10.0,
    )

    xdmf_ref = f"{REF_CONFIG['output_dir']}/{REF_CONFIG['file_name']}.xdmf"

    if not os.path.exists(xdmf_ref):
        print("="*60)
        print("  Génération des données de référence (simulation vraie)…")
        print("="*60)
        from plasticity_simu import ElasticModel, J2IsotropicHardening
        elastic_true = ElasticModel(
            E=TRUE_PARAMS["E"], nu=TRUE_PARAMS["nu"], tdim=3
        )
        model_true = J2IsotropicHardening(
            elastic_true,
            sigma_Y = TRUE_PARAMS["sigma_Y"],
            Q_var   = TRUE_PARAMS["Q_var"],
            k       = TRUE_PARAMS["k"],
        )
        run_simulation_V2(
            config=REF_CONFIG,
            model=model_true,
            write_output=True,
        )
        print("  Référence générée :", xdmf_ref)
    else:
        print(f"  Référence déjà existante : {xdmf_ref}")

    THETA0 = [
        180_000.0,   # E
        0.25,        # nu
        85.0,        # σ_Y
        40.0,        # Q
        8.0,         # k
    ]

    BOUNDS = [
        (50_000.0, 400_000.0),   # E
        (0.05,     0.45),        # nu
        (10.0,     500.0),       # σ_Y
        (0.0,      300.0),       # Q
        (0.1,      200.0),       # k
    ]

    result, problem = run_femu_identification(
        xdmf_path       = xdmf_ref,
        theta0          = THETA0,
        bounds          = BOUNDS,
        config          = REF_CONFIG,
        method          = "FEMU-Epsilon",
        optimizer       = "Nelder-Mead",
        tol             = 1e-5,
        maxiter         = 300,
        true_params     = list(TRUE_PARAMS.values()),
    )

    ref_data = ReferenceData.load(xdmf_ref)
    plot_femu_results(
        problem     = problem,
        result      = result,
        ref         = ref_data,
        config      = REF_CONFIG,
        true_params = list(TRUE_PARAMS.values()),
        save_path   = "femu_results.png",
    )