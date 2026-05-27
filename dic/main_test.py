

import numpy as np
import pyvista as pv
import pandas as pd
import skimage.io
from dolfinx.io import XDMFFile
from mpi4py import MPI

# Import de vos fonctions du fichier dic_calibration.py
from projet_dic.dic.dic_calibration_V1 import (
    calibrate_2d, # On utilise la version non-cached pour garder le contrôle
    check_calibration_2d,
    dolfinx_mesh_to_pv_mesh
)

def main():
    # --- 1. Configuration des chemins ---
    path_cad_mesh = "astar_2D_uniform.xdmf"          # Votre maillage CAD de référence
    path_dic_csv = "VK03-1-16-0201_0.csv"           # Export CSV de VIC-2D
    path_ref_image = "VK03-1-16-0001_0.tif"      # Image de l'état 0 (pixels)
    output = "dic_calibrated_on_cad.vtk"

    print("--- Chargement du maillage CAD ---")
    # Chargement manuel pour s'assurer que XDMFFile est bien utilisé
    try:
        with XDMFFile(MPI.COMM_WORLD, path_cad_mesh, "r") as xdmf:
            mesh_cad = xdmf.read_mesh(name="Grid") # "Grid" est le nom par défaut souvent utilisé
    except Exception as e:
        print(f"Erreur lors du chargement du maillage : {e}")
        return

    print("--- Démarrage de la calibration ---")
    try:
        # On charge l'image de référence
        ref_img = skimage.io.imread(path_ref_image, as_gray=True)
        
        # On appelle la calibration 2D directe
        tform_cad_to_img_4d = calibrate_2d(
            mesh=mesh_cad,
            ref_img=ref_img,
            min_scale=0.7,
            max_scale=1.3
        )
        print("Matrice de calibration calculée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la calibration : {e}")
        return

    # --- Traitement des données DIC (CSV) ---
    print(f"Lecture du CSV : {path_dic_csv}")
    df = pd.read_csv(path_dic_csv)
    df.columns = [s.replace('"', "").replace(" ", "") for s in df.columns]

    # On ne garde que les points corrélés
    retained_data = df[df["eyy"] != 0].copy()
    
    # Construction des points en pixels (homogènes)
    pts_pixel = np.zeros((len(retained_data), 4))
    pts_pixel[:, 0] = retained_data["x"]
    pts_pixel[:, 1] = retained_data["y"]
    pts_pixel[:, 3] = 1.0 # Coordonnée homogène

    # --- Projection inverse : Pixels -> Coordonnées CAD (mm) ---
    tform_inv = np.linalg.inv(tform_cad_to_img_4d)
    pts_cad_real = (tform_inv @ pts_pixel.T).T

    # --- Export PyVista ---
    cloud = pv.PolyData(pts_cad_real[:, :3])
    for col in retained_data.columns:
        if col not in ["x", "y"]:
            cloud[col] = retained_data[col].values

    # Reconstruction de la surface et sauvegarde
    surf = cloud.delaunay_2d(alpha=1.0)
    surf.save(output)
    print(f"Fichier calibré enregistré sous : {output}")

if __name__ == "__main__":
    main()