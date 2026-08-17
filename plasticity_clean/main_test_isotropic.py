"""
main_test_isotropic.py
----------------------
Entry point script for FEMU parameter identification of J2 isotropic materials.

Pipeline Overview
~~~~~~~~~~~~~~~~~
1. Stage 1 (project_csv):
   Projects raw DIC displacement CSV files onto a 3D/2D CAD specimen mesh, generating
   a PVD XML collection manifest and associated VTU timestep files.
   
2. Stage 2 (femu):
   Runs the Finite Element Model Updating (FEMU) optimization algorithm to identify 
   the J2 isotropic hardening parameters (Yield stress sigma_Y, Voce saturation Q_var,
   Voce exponent k_hardening, boundary displacement velocities) against experimental 
   DIC kinematic fields and force measurements.
"""

import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh

from dic_importation import process_csv_series_to_cad_mesh
from femu_generic import femu_res_generic


# ---------------------------------------------------------------------------
# Pipeline Execution Flags
# ---------------------------------------------------------------------------
project_csv = 0   # Set to 1 to run DIC CSV projection onto CAD, 0 to skip
femu        = 1   # Set to 1 to run FEMU parameter identification, 0 to skip


# ---------------------------------------------------------------------------
# Stage 1 – Project DIC CSV series onto CAD mesh
# ---------------------------------------------------------------------------
if project_csv == 1:
    print("[Stage 1] Lancement de la projection des fichiers CSV DIC sur la CAO...")
    process_csv_series_to_cad_mesh(
        folder_path="/home/pmarechal/Documents/synthetic_csv/carre_trou_ortho_y0_10_csv",
        file_prefix="carre_trou_ortho_y0_10_",
        mesh_cad_path="carre_trou.msh",
        tform_img_to_cad_4D=np.identity(4),
        output_pvd_path="MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        alpha=20.0,
        ech=1,
        start_idx=0,
        end_idx=52,
    )


# ---------------------------------------------------------------------------
# Stage 2 – FEMU Optimisation for J2 Isotropic Material Model
# ---------------------------------------------------------------------------
if femu == 1:
    PVD_FILE   = "MAINTEST/pyvista_exports/csv_projection_tensile_iso/dic_series_projected.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection_tensile_iso/forces_sample.npy"

    print("Lancement de l'optimisation FEMU mixte (Champs u + Forces F) avec J2IsotropicHardening...")

    # Execute FEMU optimization algorithm for J2 isotropic hardening
    optimizer_result = femu_res_generic(
        PVD_FILE,
        FORCE_FILE,
        model_name="J2IsotropicHardening",
        params0_overrides={
            # Initial guess overrides can be specified here
        },
        free_param_names=[
            "t_start",      # Initial time offset
            "sigma_Y",      # Initial yield stress [MPa]
            "Q_var",        # Voce hardening saturation stress [MPa]
            "k_hardening",  # Voce hardening rate exponent
            "uy_up",        # Prescribed displacement rate (top boundary, Y)
            "uy_down",      # Prescribed displacement rate (bottom boundary, Y)
            "ux_up",        # Prescribed displacement rate (top boundary, X)
            "ux_down",      # Prescribed displacement rate (bottom boundary, X)
        ],
        fixed_param_overrides={
            "E":      200_000.0, # Fixed Young's modulus [MPa]
            "nu":     0.3,       # Fixed Poisson ratio
            "uz_up":  0.0,       # Fixed top boundary Z displacement rate
            "uz_down":0.0,       # Fixed bottom boundary Z displacement rate
        },
        config={
            "weight_u": 1.0,     # Kinematic field error weighting
            "weight_f": 4.0,     # Force discrepancy weighting
        },
    )

    print("\n================ OPTIMISATION TERMINÉE ================")
    print(
        "Paramètres identifiés :",
        dict(zip(optimizer_result.param_names, optimizer_result.x)),
    )
