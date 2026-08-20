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
force_export = 0

project_csv = 0   # Set to 1 to run DIC CSV projection onto CAD, 0 to skip
femu        = 1 # Set to 1 to run FEMU parameter identification, 0 to skip



if force_export == 1:
    import numpy as np

    def exporter_force_npy(fichier_txt, img_debut, img_fin, ech, fichier_sortie="matrice_force.npy"):
        """
        Lit un fichier d'acquisition texte, extrait la PREMIÈRE force axiale en Newtons
        pour chaque image cible (de img_debut à img_fin avec un pas 'ech'),
        puis sauvegarde le résultat sous forme de matrice NumPy (.npy).
        """
        forces = []
        images_traitees = set()
        
        # Définition des numéros d'images cibles
        images_cibles = set(range(img_debut, img_fin + 1, ech))
        
        with open(fichier_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Ignorer les lignes vides et le bloc d'en-tête/métadonnées
                if not line or any(line.startswith(prefix) for prefix in ["Chemin", "Essai", "Exécutions", "Date", "Axial", "kN"]):
                    continue
                
                # Remplacement de la virgule décimale par un point
                elements = line.replace(',', '.').split()
                
                if len(elements) >= 5:
                    try:
                        force_kN = float(elements[0])
                        nb_image = int(float(elements[4]))
                        
                        # On garde uniquement la première occurrence de chaque image cible
                        if nb_image in images_cibles and nb_image not in images_traitees:
                            force_N = force_kN * 1000.0  # Conversion kN -> N
                            forces.append(force_N)
                            images_traitees.add(nb_image)  # Marque l'image comme traitée
                            
                    except ValueError:
                        continue

        # Transformation en matrice NumPy (vecteur colonne 2D : N lignes, 1 colonne)
        matrice_force = np.array(forces).reshape(-1, 1)
        
        # Export dans un fichier .npy
        np.save(fichier_sortie, matrice_force)
        
        print(f"Export terminé : {len(forces)} valeurs enregistrées dans '{fichier_sortie}'.")
        return matrice_force



    matrice = exporter_force_npy(
        fichier_txt=r"/home/pmarechal/Documents/ESCAL/transfer_12868454_files_9cfdfa8b/machine/Acqui_image - (Niveau delta)1.txt",
        img_debut=9,
        img_fin=5971,
        ech=108,
        fichier_sortie="forces_echantillonnees.npy"
    )
    
    plt.plot(matrice)
    plt.show()


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
        folder_path="/home/pmarechal/Documents/A305/A305_rectangle",
        file_prefix="test00",
        mesh_cad_path="/home/pmarechal/Documents/projet_dic/plasticity/A305_COARSE.msh",
        tform_img_to_cad_4D=np.linalg.inv(T),  # Use the inverse of the calibration matrix because T maps from CAD to image coordinates, but we need the opposite for projection.
        output_pvd_path="MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        alpha=20.0,
        ech=108,
        start_idx=9,
        end_idx=5791,
    )


# ---------------------------------------------------------------------------
# Stage 2 – FEMU Optimisation for J2 Isotropic Material Model
# ---------------------------------------------------------------------------
if femu == 1:
    PVD_FILE   = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection_A305_FULL/forces_sample_A305.npy"

    print("Lancement de l'optimisation FEMU mixte (Champs u + Forces F) avec J2IsotropicHardening...")

    # Execute FEMU optimization algorithm for J2 isotropic hardening
    optimizer_result = femu_res_generic(
        PVD_FILE,
        FORCE_FILE,
        model_name="J2IsotropicHardening",
        params0_overrides={
            "k_hardening":300.0,
            "sigma_Y": 700.0,
            'Q_var': 200.0,
            "uy_down": -1.0,
            "uy_up": 1.0,
            # Initial guess overrides can be specified here
        },
        free_param_names=[      # Initial time offset
            "sigma_Y",      # Initial yield stress [MPa]
            "Q_var",        # Voce hardening saturation stress [MPa]
            "k_hardening",  # Voce hardening rate exponent
            "uy_up",        # Prescribed displacement rate (top boundary, Y)
            "uy_down",      # Prescribed displacement rate (bottom boundary, Y)
            "ux_up",        # Prescribed displacement rate (top boundary, X)
            "ux_down",  # Prescribed displacement rate (bottom boundary, X)
        ],
        fixed_param_overrides={

            "t_start":0.0,
            "E":      200_000.0, # Fixed Young's modulus [MPa]
            "nu":     0.3,       # Fixed Poisson ratio
            "uz_up":  0.0,       # Fixed top boundary Z displacement rate
            "uz_down":0.0,       # Fixed bottom boundary Z displacement rate
        },
        config={
            "T" : 3.0,
            "weight_u": 5.0,     # Kinematic field error weighting
            "weight_f": 1.0,     # Force discrepancy weighting
        },
        bounds_overrides={
            "k_hardening":(30.0,1500.0),
            'sigma_Y':(180,1000),
            'Q_var':(100.0, 1000.0),
            "ux_up":       (-0.01, 0.01),
            "uy_up":       (0.1, 5),
            "ux_down":     (-0.01, 0.01),
            "uy_down":     (-5, -0.1),
        },
    )

    print("\n================ OPTIMISATION TERMINÉE ================")
    print(
        "Paramètres identifiés :",
        dict(zip(optimizer_result.param_names, optimizer_result.x)),
    )
