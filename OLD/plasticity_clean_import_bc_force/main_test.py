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
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pipeline Execution Flags
# ---------------------------------------------------------------------------


project_csv = 0   # Set to 1 to run DIC CSV projection onto CAD, 0 to skip
femu        = 1 # Set to 1 to run FEMU parameter identification, 0 to skip


# ---------------------------------------------------------------------------
# Import 4x4 calibration matrix
# ---------------------------------------------------------------------------

T = np.load('calibration_matrix_A305.npy')  # Load the calibration matrix from a .npy file generated with test_calibration.py

# ---------------------------------------------------------------------------
# Stage 1 – Project DIC CSV series onto CAD mesh
# ---------------------------------------------------------------------------
if project_csv == 1:
    print("[Stage 1] Lancement de la projection des fichiers CSV DIC sur la CAO...")
    process_csv_series_to_cad_mesh(
        folder_path="/home/pmarechal/Documents/A305/A305_full",
        file_prefix="test00",
        mesh_cad_path="/home/pmarechal/Documents/projet_dic/plasticity/A305_COARSE.msh",
        tform_img_to_cad_4D=np.linalg.inv(T),  # Use the inverse of the calibration matrix because T maps from CAD to image coordinates, but we need the opposite for projection.
        output_pvd_path="MAINTEST/pyvista_exports/csv_projection_full/dic_series_projected_A305.pvd",
        alpha=20.0,
        ech=108,
        start_idx=9,
        end_idx=5791,
        fill_outside_with_nearest=True
    )


# ---------------------------------------------------------------------------
# Stage 2 – FEMU Optimisation for J2 Isotropic Material Model
# ---------------------------------------------------------------------------

print("salut")

if femu == 1:
    PVD_FILE   = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected_A305.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection/forces_sample_A305.npy"

    print("Lancement de l'optimisation FEMU mixte (Champs u + Forces F) avec J2IsotropicHardening...")

    # Execute FEMU optimization algorithm for J2 isotropic hardening
    optimizer_result = femu_res_generic(
        PVD_FILE,
        FORCE_FILE,
        model_name="J2IsotropicHardening",
        params0_overrides={
            "Q_var":200.0,
            "k_hardening":100.0,
            "sigma_Y": 300.0,
        },
        free_param_names=[
            "Q_var",
            "k_hardening",
            "sigma_Y",
        ],
        fixed_param_overrides={
            "nu": 0.3,

            "E":210_000.0,
        },
        bounds_overrides={
            "k_hardening":(10.0,15000.0),
            'sigma_Y':(180.0,10000.0),
            'Q_var':(10.0, 10000.0),
            "E":(50_000.0, 250_0000.0),
        },
        config={
            "weight_u": 5.0,
            "weight_f": 1.0,
        }
    )
    print("FEMU optimization completed. Optimized parameters:")
    print(
            "Paramètres identifiés :",
            dict(zip(optimizer_result.param_names, optimizer_result.x)),
        )
