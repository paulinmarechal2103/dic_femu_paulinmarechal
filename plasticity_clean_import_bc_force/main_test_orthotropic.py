"""
main_test_orthotropic.py
------------------------
Entry point script for FEMU parameter identification of Hill48 orthotropic materials.

Pipeline Overview
~~~~~~~~~~~~~~~~~
Runs the Finite Element Model Updating (FEMU) optimization algorithm to identify 
the Hill48 anisotropic parameters (F, G, N), isotropic hardening parameters 
(sigma_Y, Q_var, k_hardening), and boundary velocities against experimental 
DIC displacement fields and force measurements.
"""

import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh

from dic_importation import process_csv_series_to_cad_mesh
from femu_generic import femu_res_generic


# ---------------------------------------------------------------------------
# FEMU Optimisation for Hill48_2 Orthotropic Model
# ---------------------------------------------------------------------------
PVD_FILE   = "MAINTEST/pyvista_exports/csv_projection_carre_trou/dic_series_projected.pvd"
FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection_carre_trou/forces_anisotrope_carre_trou.npy"

print("Lancement de l'optimisation FEMU mixte (Champs u + Forces F) avec Hill48_2...")

# Execute FEMU optimization algorithm for Hill48 orthotropic plasticity
optimizer_result = femu_res_generic(
    PVD_FILE,
    FORCE_FILE,
    model_name="Hill48_2",
    params0_overrides={
        "F": 0.5,
        "G": 0.5,
        "N": 1.5,
    },
    free_param_names=[
        "t_start",      # Initial time offset
        "F",            # Hill48 transverse anisotropy parameter
        "G",            # Hill48 longitudinal anisotropy parameter
        "N",            # Hill48 in-plane shear anisotropy parameter
        "sigma_Y",      # Initial yield stress [MPa]
        "Q_var",        # Voce hardening saturation stress [MPa]
        "k_hardening",  # Voce hardening exponent
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
        "L":      1.5,       # Fixed out-of-plane shear parameter
        "M":      1.5,       # Fixed out-of-plane shear parameter
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
