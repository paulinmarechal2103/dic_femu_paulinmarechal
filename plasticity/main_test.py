from dic_importation import *
from image_calibration import *
from femu_DIC import *

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh

import os

dossier_csv = "/home/pmarechal/Documents/X65_L/VCXU-51C_700005517948_260616-123640"
file_prefix = "X65_L_000"

H5_FILE = "MAINTEST/dic_series.h5"
GMSH_FILE = "x65.msh"
OUTPUT_XDMF = "MAINTEST/projection_cad_temporelle_mask.xdmf"

import_csv = 0 # 1 pour importer les CSV, 0 pour ne pas le faire
project_csv = 0  # 1 pour projeter, 0 pour ne pas le faire
resample = 0  # 1 pour rééchantillonner, 0 pour ne pas le faire
femu = 1  # 1 pour lancer l'optimisation, 0 pour ne pas le faire






if import_csv == 1:
    process_csv_series_fenicsx(
        folder_path=dossier_csv, 
        output_xdmf="MAINTEST/dic_series.xdmf", 
        file_prefix=file_prefix, 
        alpha=20.0,
        ech=7,
        start_idx = 20,
        end_idx = 373,
    )



if project_csv == 1:
    domain = load_and_write_mesh(GMSH_FILE)


    ref_image = skimage.io.imread("/home/pmarechal/Documents/X65_L/VCXU-51C_700005517948_260616-123640/X65_L_0000000.tif", as_gray=True)


    tform_cad_to_img_4d = calibrate_2d_manual(domain,ref_image)
    #tform_cad_to_img_4d = np.identity(4)
    tform_h5_to_cad = np.linalg.inv(tform_cad_to_img_4d)



    print("=" * 60)
    print("  VÉRIFICATION ET LANCEMENT DE LA PROJECTION TEMPORELLE")
    print("=" * 60)

    if not os.path.exists(H5_FILE):
        print(f"[Erreur] Le fichier source H5 est introuvable : {H5_FILE}")
        print("-> Veuillez corriger la variable 'H5_FILE'.")
        
    elif not os.path.exists(GMSH_FILE):
        print(f"[Erreur] Le fichier cible Gmsh (.msh) est introuvable : {GMSH_FILE}")
        print("-> Veuillez corriger la variable 'GMSH_FILE'.")
        
    else:
        print(f"[OK] Fichier H5 trouvé : {H5_FILE}")
        print(f"[OK] Fichier Gmsh trouvé : {GMSH_FILE}")
        print(f"[Info] Fichier de sortie prévu : {OUTPUT_XDMF}\n")
        
        try:
            project_h5_series_to_cad_mesh_mask_3(
                h5_path=H5_FILE,
                mesh_cad=domain,
                tform_h5_to_cad_4D=tform_h5_to_cad,
                output_xdmf_path=OUTPUT_XDMF
            )
            print("\n" + "=" * 60)
            print("[Succès] Traitement terminé sans accroc.")
            print(f"[Aide] Vous pouvez maintenant ouvrir '{OUTPUT_XDMF}' dans ParaView")
            print("       pour visualiser le déplacement projeté sur la CAO au cours du temps.")
            print("=" * 60)
            
        except Exception as e:
            print("\n" + "!" * 60)
            print("[Échec] Une erreur est survenue pendant l'interpolation :")
            print("!" * 60)
            import traceback
            traceback.print_exc()


if resample == 1:
    resample_h5_time_series(
        input_h5_path= OUTPUT_XDMF,
        output_xdmf_path = "MAINTEST/projection_cad_temporelle_mask_resampled.xdmf",
        target_num = 50,
        kind = "linear"
    )


if femu == 1:
    bounds_ref_J2_centr= [
            (200000, 200000+1e-6),   # E [MPa]
            (0.3, 0.3+1e-10),         # nu 
            (10.0, 500.0),        # sigma_Y [MPa]
            (5.0, 400.0),         # Q_var [MPa]
            (10.0, 1500.0),          # k_hardening
        ]

    XDMF_FILE = "validation_import_dic/projection_cad_temporelle_mask.xdmf"
    # XDMF_FILE = "results/projection_cad_temporelle_mask.xdmf"

    real_params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]
    params_names = ["E", "nu", "sigma_Y", "Q_var", "k_hardening"]
    from random import uniform,seed
    seed(43)  # Pour la reproductibilité
    perturbation_percentage = 0.15  # 15% de perturbation aléatoire
    normalized_result = normalize_params(real_params, bounds_ref_J2_centr)
    normalized_disturbed = [i + uniform(-perturbation_percentage, perturbation_percentage) for i in normalized_result]
    normalized_disturbed = [min(max(i, 0.0), 1.0) for i in normalized_disturbed]  # Clamp entre 0 et 1
    parameters_disturbed = denormalize_params(normalized_disturbed, bounds_ref_J2_centr)
    optimizer_result = femu_res_J2_DIC_BC(XDMF_FILE, bounds=bounds_ref_J2_centr, params0=parameters_disturbed,params_names = params_names)

    print("Optimized parameters (phys):", optimizer_result.x)
    print("normalized error:", [f"{params_names[i]} : {round(abs(optimizer_result.x[i] - real_params[i])/abs(real_params[i])*100,5)}%" for i in range(len(real_params))])
