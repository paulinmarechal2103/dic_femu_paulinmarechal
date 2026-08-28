
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from dic_projection.dic_importation import process_csv_series_to_cad_mesh
from femu_toolbox import femu_res_toolbox
from plasticity_solver_import_bc.plasticity_simu import run_simulation_bc_vtu_fast
from simu_tools import (
    ElasticModel,
    J2IsotropicHardening,
    build_function_spaces,
    get_vtu_files_from_pvd,
    load_domain_from_vtu,
)

# ---------------------------------------------------------------------------
# Pipeline Execution Flags
# ---------------------------------------------------------------------------
project_csv = 0   # Set to 1 to run DIC CSV projection onto CAD, 0 to skip
femu = 1  # Set to 1 to run FEMU parameter identification, 0 to skip


# ---------------------------------------------------------------------------
# 3. Stage 1 – Project DIC CSV Series onto CAD Mesh
# ---------------------------------------------------------------------------
if project_csv == 1:
    print("[Stage 1] Launching DIC CSV displacement projection onto CAD mesh...")
    process_csv_series_to_cad_mesh(
        folder_path="MAINTEST/CSV/carre_trou_ortho_y0_10_csv/",
        file_prefix="carre_trou_ortho_y0_10_",
        mesh_cad_path="msh/carre_trou.msh",
        tform_img_to_cad_4D=np.identity(4),  # Inverse calibration matrix (image -> CAD)
        output_pvd_path="MAINTEST/pyvista_exports/carre_trou_ortho_y0_10_/dic_series_projected.pvd",
        alpha=20.0,
        ech=1,
        start_idx=0,
        end_idx=51,
    )


# ---------------------------------------------------------------------------
# 4. Stage 2 – FEMU Parameter Identification (J2 Isotropic Material)
# ---------------------------------------------------------------------------

if femu == 1:
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection_carre_ortho/dic_series_projected.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection_carre_ortho/forces_anisotrope_carre_trou.npy"

    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)
    from plasticity_solver_import_bc.hill48_model import Hill48Model
    def my_solver(run_cfg: Dict[str, Any]) -> Tuple[np.ndarray, pv.MultiBlock]:
        """
        FEM solver example wrapper for J2 isotropic hardening parameter identification.

        Parameters
        ----------
        run_cfg : Dict[str, Any]
            Configuration dictionary containing current material parameter values.

        Returns
        -------
        f_sim : np.ndarray
            Scaled force vector output from the simulation (N).
        sim_multiblock : pv.MultiBlock
            PyVista MultiBlock containing simulated displacement fields across timesteps.
        """
        model = Hill48Model(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            Q_var=run_cfg["Q_var"],
            k_hardening=run_cfg["k_hardening"],
            F=run_cfg["F"],
            G=run_cfg["G"],
            H=1-run_cfg["G"],
            L=1.5,
            M=1.5,
            N=run_cfg["N"],
        )
        f_sim, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)

        return np.array(f_sim), sim_multiblock

    # Initial guesses and fixed values for model parameters
    Optim_params = {
        "E": 200_000.0,          # Young's modulus (MPa)
        "nu": 0.30,               # Poisson's ratio (fixed)
        "sigma_Y": 80.0,         # Initial yield stress (MPa)
        "Q_var": 80.0,            # Voce isotropic hardening modulus (MPa)
        "k_hardening": 800.0,    # Voce hardening exponent
        "t_start": 0.0,  
        "F":0.5,
        "G":0.5,
        "N":1.5,
    }

    # Optimization bounds (parameters not listed here remain fixed)
    Bounds = {
        "Q_var": (20.0, 200.0),
        "sigma_Y": (50.0, 400),
        "k_hardening": (10.0, 1_500.0),
        "F":(0,1),
        "G":(0,1),
        "N":(1,2),

    }

    config = {
        "pvd_file_path": PVD_FILE,
        "num_steps": 50,
        "T": 3.0,
        "weight_u": 0.0,
        "weight_f": 5.0,
    }

    # Run FEMU identification optimization loop
    result = femu_res_toolbox(
        PVD_file=PVD_FILE,
        FORCE_file=FORCE_FILE,
        Optim_params=Optim_params,
        Bounds=Bounds,
        solver=my_solver,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Identification complete.")
    for name, val in zip(result.param_names, result.x):
        print(f"  {name:>15s} = {val:.4f}")
    if result.fixed_params:
        print("Fixed parameters used:")
        for name, val in result.fixed_params.items():
            print(f"  {name:>15s} = {val}")
    print(f"  Final cost (sum r^2) = {float(np.sum(result.fun ** 2)):.6e}")
    print("=" * 60)
