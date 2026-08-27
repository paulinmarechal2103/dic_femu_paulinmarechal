from dic_importation import *
from image_calibration import *
from femu_generic import *

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh

import os


project_csv = 1 # 1 pour projeter, 0 pour ne pas le faire
femu = 0  # 1 pour lancer l'optimisation, 0 pour ne pas le faire



print("hey")
if project_csv == 1:
    process_csv_series_to_cad_mesh(
        folder_path="/home/pmarechal/Documents/synthetic_csv/carre_trou_ortho_y0_10_csv",
        file_prefix="carre_trou_ortho_y0_10_", 
        mesh_cad_path="A305.msh", 
        tform_h5_to_cad_4D = np.identity(4), 
        output_pvd_path = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        alpha=20.0,
        ech=1,
        start_idx = 0,
        end_idx = 52,
    )
    
if femu == 1:
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection/forces_anisotrope_carre_trou.npy"

    print("Lancement de l'optimisation FEMU mixte (Champs u + Forces F) avec Hill48...")

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
            "t_start",
            "F",
            "G",
            "N",
            "sigma_Y",
            "Q_var",
            "k_hardening",
            "uy_up",
            "uy_down",
            "ux_up",
            "ux_down",
        ],
        fixed_param_overrides={
            "E": 200_000.0,
            "nu": 0.3,
            "uz_up": 0.0,
            "uz_down": 0.0,
            "L": 1.5,
            "M": 1.5,
        },
        config={
            "weight_u": 1.0,
            "weight_f": 4.0,
        }
    )

    print("\n================ OPTIMISATION TERMINÉE ================")
    print("Paramètres identifiés :", dict(zip(optimizer_result.param_names, optimizer_result.x)))