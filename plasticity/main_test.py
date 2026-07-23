from dic_importation import *
from image_calibration import *
from femu_DIC import *

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh

import os

dossier_csv = "/home/pmarechal/Documents/synthetic_csv/fenicsx_surface_z0_csv"
file_prefix = "FE_z0_step_"


H5_FILE = "MAINTEST/dic_series.h5"
GMSH_FILE = "x65.msh"
OUTPUT_XDMF = "MAINTEST/projection_cad_temporelle_mask.xdmf"

project_csv = 1  # 1 pour projeter, 0 pour ne pas le faire
femu = 0  # 1 pour lancer l'optimisation, 0 pour ne pas le faire




import os


if project_csv == 1:
    process_csv_series_to_cad_mesh(
        folder_path="/home/pmarechal/Documents/synthetic_csv/fenicsx_surface_z0_csv",
        file_prefix="FE_z0_step_", 
        mesh_cad_path="Flat_specimen_refined.msh", 
        tform_h5_to_cad_4D = np.identity(4), 
        output_pvd_path = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        alpha=20.0,
        ech=1,
        start_idx = 0,
        end_idx = 52,
    )
    
if femu == 1:
    from random import uniform, seed
    import numpy as np

    # 1. Définition des bornes (inchangées)
    bounds_ref_J2_centr= [
            (200000, 200000+1e-6),   # E [MPa]
            (0.3, 0.3+1e-10),         # nu 
            (10.0, 500.0),        # sigma_Y [MPa]
            (5.0, 400.0),         # Q_var [MPa]
            (10.0, 1500.0),          # k_hardening
        ]

    # 2. Passage au fichier PVD (au lieu du XDMF)
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    # PVD_FILE = "results/projection_cad_temporelle_mask.pvd"
    
    real_params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]
    params_names = ["E", "nu", "sigma_Y", "Q_var", "k_hardening"]
    
    # 3. Génération de la perturbation (inchangée)
    seed(43)  # Pour la reproductibilité
    perturbation_percentage = 0.15  # 15% de perturbation aléatoire
    
    normalized_result = normalize_params(real_params, bounds_ref_J2_centr)
    normalized_disturbed = [i + uniform(-perturbation_percentage, perturbation_percentage) for i in normalized_result]
    normalized_disturbed = [min(max(i, 0.0), 1.0) for i in normalized_disturbed]  # Clamp entre 0 et 1
    parameters_disturbed = denormalize_params(normalized_disturbed, bounds_ref_J2_centr)
    
    print("Paramètres initiaux perturbés (physiques) :", parameters_disturbed)
    print("Lancement de l'optimisation FEMU via le pipeline PyVista...")

    # 4. Lancement de l'optimisation avec le fichier PVD
    optimizer_result = femu_res_J2_DIC_BC(
        PVD_FILE, 
        bounds=bounds_ref_J2_centr, 
        params0=real_params,#parameters_disturbed,
        params_names=params_names
    )
    
    # 5. Affichage des résultats et calcul de l'erreur
    print("\n================ OPTIMISATION TERMINÉE ================")
    print("Optimized parameters (phys):", optimizer_result.x)
    print("Normalized error:")
    for i in range(len(real_params)):
        err_percent = abs(optimizer_result.x[i] - real_params[i]) / abs(real_params[i]) * 100
        print(f"  - {params_names[i]} : {err_percent:.5f}%")