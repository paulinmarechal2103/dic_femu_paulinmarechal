# --- Bibliothèque standard ---
import glob
import os
import re
from typing import Optional

# --- Calcul et manipulation de données ---
import h5py
import numpy as np
import pandas as pd
import skimage
from numpy.typing import NDArray
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d,LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import KDTree

# --- Maillage, Visualisation et Éléments Finis ---
import basix
import dolfinx.fem as fem
import dolfinx.io
import meshio
import pyvista as pv
import ufl
from mpi4py import MPI

# --- Modules locaux ---
from image_calibration import calibrate_2d_manual, check_calibration_2d
from plasticity_simu import load_and_write_mesh


def create_reference_mesh_from_csv(csv_path: str, alpha: float) -> pv.PolyData:
    """Creates a 2D triangulated mesh from CSV coordinates using PyVista."""
    df = pd.read_csv(csv_path)
    
    # Flexible matching for common coordinate column names
    x_col = next((c for c in df.columns if c.lower() in ['x', 'pos_x', 'coord_x']), None)
    y_col = next((c for c in df.columns if c.lower() in ['y', 'pos_y', 'coord_y']), None)
    z_col = next((c for c in df.columns if c.lower() in ['z', 'pos_z', 'coord_z']), None)
    
    if not x_col or not y_col:
        raise ValueError(f"Could not find coordinate columns (x, y) in {csv_path}. Columns: {df.columns.tolist()}")
    
    # PyVista points must be 3D vectors (x, y, z)
    points = np.zeros((len(df), 3))
    points[:, 0] = df[x_col].values
    points[:, 1] = df[y_col].values
    if z_col:
        points[:, 2] = df[z_col].values
        
    # Build PolyData mesh and triangulate using PyVista's built-in 2D Delaunay with alpha shapes
    poly = pv.PolyData(points)
    mesh = poly.delaunay_2d(alpha=alpha)
    return mesh

def read_msh_safely(msh_path: str) -> pv.UnstructuredGrid:
    mesh = meshio.read(msh_path)
    mesh.cell_sets = {}
    pv_mesh = pv.from_meshio(mesh)
    
    # --- AJOUT : Ne conserver que les tétraèdres (Type 10) ---
    # VTK_TETRA est le type 10
    pv_mesh = pv_mesh.extract_cells(pv_mesh.celltypes == 10)
    
    return pv_mesh
import os
import re
import glob
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import KDTree

# NOTE : create_reference_mesh_from_csv et read_msh_safely sont supposées déjà
# définies ailleurs dans le projet (elles n'étaient pas incluses dans le code fourni).
# from mon_module_existant import create_reference_mesh_from_csv, read_msh_safely


def update_displacement_field_pyvista(csv_path: str, mesh: pv.DataSet) -> None:
    """Lit les déplacements depuis un CSV et les assigne comme Point Data sur le maillage.

    Corrections apportées par rapport à la version d'origine :
      - Vérification de l'existence du fichier CSV.
      - Vérification que le nombre de lignes du CSV correspond bien au nombre de points
        du maillage de référence (sinon le déplacement était silencieusement mal assigné,
        ou l'assignation levait une erreur numpy peu explicite plus loin dans le pipeline).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    ux_col = next((c for c in df.columns if c.lower() in ("u", "u_x", "disp_x")), None)
    uy_col = next((c for c in df.columns if c.lower() in ("v", "u_y", "disp_y")), None)
    uz_col = next((c for c in df.columns if c.lower() in ("w", "u_z", "disp_z")), None)

    if ux_col is None or uy_col is None:
        raise ValueError(
            f"Colonnes de déplacement (ux, uy) introuvables dans {csv_path}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    if len(df) != mesh.n_points:
        raise ValueError(
            f"Incohérence de taille entre {csv_path} ({len(df)} lignes) et le maillage de "
            f"référence ({mesh.n_points} points). Le CSV doit correspondre point à point au "
            f"maillage créé depuis le premier pas de temps retenu."
        )

    ux = df[ux_col].to_numpy()
    uy = df[uy_col].to_numpy()
    uz = df[uz_col].to_numpy() if uz_col is not None else np.zeros_like(ux)

    mesh.point_data["displacement"] = np.column_stack((ux, uy, uz))


def interpolate_displacement_obs_mesh_to_cad_mesh_2D(
    mesh_obs: pv.DataSet,
    mesh_cad_target: pv.DataSet,
    tform_cad_to_img_4D: np.ndarray,
    distance_threshold: Optional[float] = None,
) -> None:
    """Interpole le champ 'displacement' de mesh_obs vers mesh_cad_target par plus proche
    voisin en 2D, puis transforme les vecteurs dans l'espace CAD.

    Corrections :
      - Le paramètre était nommé `mesh_cad`, ce qui masquait la variable `mesh_cad`
        (maillage CAD complet) utilisée dans la fonction appelante d'origine. Renommé
        `mesh_cad_target` pour lever toute ambiguïté.
      - Ajout d'un `distance_threshold` optionnel : au-delà de ce seuil, le déplacement
        est mis à NaN plutôt que d'être extrapolé silencieusement depuis un plus proche
        voisin potentiellement très éloigné. Laissé à None par défaut car, dans le
        pipeline combiné ci-dessous, le masquage est fait une fois pour toutes via
        'is_imported' (moins coûteux qu'une requête KDTree à seuil à chaque pas de temps).
    """
    if "displacement" not in mesh_obs.point_data:
        raise ValueError("Le maillage observé ne contient pas de champ 'displacement'.")

    points_cad = mesh_cad_target.points
    points_cad_hom = np.hstack([points_cad, np.ones((len(points_cad), 1))])
    points_img = (tform_cad_to_img_4D @ points_cad_hom.T).T[:, :3]

    points_obs_2d = mesh_obs.points[:, :2]
    points_img_2d = points_img[:, :2]

    tree_obs = KDTree(points_obs_2d)

    if distance_threshold is not None:
        distances, indices = tree_obs.query(
            points_img_2d, k=1, distance_upper_bound=distance_threshold
        )
        valid = np.isfinite(distances)
        indices_clamped = np.where(valid, indices, 0)
        disp_obs = mesh_obs.point_data["displacement"][indices_clamped].astype(float)
        disp_obs[~valid] = np.nan
    else:
        _, indices = tree_obs.query(points_img_2d, k=1)
        disp_obs = mesh_obs.point_data["displacement"][indices]

    tform_jac_cad_to_img_3D = tform_cad_to_img_4D[:3, :3]
    inv_jacobian = np.linalg.inv(tform_jac_cad_to_img_3D)
    disp_cad_transformed = (inv_jacobian @ disp_obs.T).T

    mesh_cad_target.point_data["displacement_projected"] = disp_cad_transformed

def process_csv_series_to_cad_mesh(
    folder_path: str,
    file_prefix: str,
    mesh_cad_path: str,
    tform_h5_to_cad_4D: np.ndarray,
    output_pvd_path: str,
    alpha: float = 0.2,
    ech: int = 20,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    distance_threshold: float = 1.0,
) -> None:
    """Projection d'une série temporelle de CSV sur un maillage CAD global.

    Lit la série de CSV de déplacements, construit le maillage de référence,
    charge le maillage CAD complet, calcule le masque de présence des données
    expérimentales, puis interpole et sauvegarde chaque pas de temps sur le CAD global.

    Les nœuds du CAD situés en dehors de la zone observée (au-delà de `distance_threshold`)
    voient leur déplacement 'displacement_projected' imposé à 0.0.
    """
    # ---- 1. Détection dynamique des fichiers CSV disponibles ----
    search_pattern = os.path.join(folder_path, f"{file_prefix}[0-9][0-9][0-9][0-9].csv")
    all_files = glob.glob(search_pattern)
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier {file_prefix}XXXX.csv trouvé dans {folder_path}.")

    steps: List[int] = sorted(
        int(m.group(1)) for f in all_files if (m := re.search(r"(\d{4})\.csv$", f))
    )

    total_steps = len(steps)
    steps = steps[start_idx:end_idx]
    if not steps:
        raise ValueError(
            f"L'intervalle [{start_idx}:{end_idx}] ne contient aucune donnée (total : {total_steps})."
        )

    steps_to_process = steps[::ech]
    min_step, max_step = steps_to_process[0], steps_to_process[-1]

    first_csv = os.path.join(folder_path, f"{file_prefix}{min_step:04d}.csv")
    if not os.path.exists(first_csv):
        raise FileNotFoundError(f"Fichier de référence introuvable : {first_csv}")

    print(
        f"[Info] {total_steps} pas de temps trouvés, {len(steps_to_process)} retenus "
        f"({min_step:04d} -> {max_step:04d}, ech={ech})."
    )

    # ---- 2. Maillage de référence (géométrie fixe) depuis le premier CSV retenu ----
    print(f"[1/3] Création du maillage de référence depuis {first_csv}...")
    mesh_obs = create_reference_mesh_from_csv(first_csv, alpha=alpha)
    points_obs = mesh_obs.points

    # ---- 3. Chargement du maillage CAD global + calcul du masque ----
    print(f"[2/3] Chargement du maillage CAD : {mesh_cad_path}")
    if mesh_cad_path.lower().endswith(".msh"):
        mesh_cad = read_msh_safely(mesh_cad_path)
    else:
        mesh_cad = pv.read(mesh_cad_path)

    # Passage des points observés dans le repère CAD
    points_hom = np.hstack([points_obs, np.ones((points_obs.shape[0], 1))])
    points_obs_in_cad = (tform_h5_to_cad_4D @ points_hom.T).T[:, :3]
    points_obs_2d = points_obs_in_cad[:, :2]
    tree_obs_2d = KDTree(points_obs_2d)

    # Calcul de proximité sur TOUS les points du maillage CAD complet
    cad_distances, _ = tree_obs_2d.query(
        mesh_cad.points[:, :2], distance_upper_bound=distance_threshold
    )
    is_imported = cad_distances <= distance_threshold

    if not np.any(is_imported):
        raise ValueError(
            "Aucun point du maillage CAD ne correspond aux données observées "
            f"(seuil de distance = {distance_threshold})."
        )

    mesh_cad.point_data["is_imported"] = np.where(is_imported, 0.1, 0.0)
    outside_mask = ~is_imported  # Prépare le masque binaire d'exclusion

    # ---- 4. Transformation inverse ----
    tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

    # ---- 5. Boucle temporelle : déplacement -> interpolation -> masquage -> sauvegarde ----
    output_dir = os.path.dirname(output_pvd_path) or "."
    pvd_name_no_ext = os.path.splitext(os.path.basename(output_pvd_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[3/3] Projection des {len(steps_to_process)} pas de temps sur le CAD global "
        f"-> {output_pvd_path}"
    )
    processed_steps: List[Tuple[float, str]] = []

    for step in steps_to_process:
        csv_path = os.path.join(folder_path, f"{file_prefix}{step:04d}.csv")
        if not os.path.exists(csv_path):
            print(f"[Attention] Pas de temps {step} manquant ({csv_path}). Ignoré.")
            continue

        update_displacement_field_pyvista(csv_path, mesh_obs)

        interpolate_displacement_obs_mesh_to_cad_mesh_2D(
            mesh_obs=mesh_obs,
            mesh_cad_target=mesh_cad,
            tform_cad_to_img_4D=tform_cad_to_img_4D,
        )

        # Application du masque : les points du CAD hors zone couverte reçoivent 0
        mesh_cad.point_data["displacement_projected"][outside_mask] = 0.0

        vtu_filename = f"{pvd_name_no_ext}_{step:04d}.vtu"
        mesh_cad.save(os.path.join(output_dir, vtu_filename))
        processed_steps.append((float(step), vtu_filename))
        print(f" -> t={step:04d} projeté sur CAD complet ({csv_path}).")

    if not processed_steps:
        raise RuntimeError("Aucun pas de temps n'a pu être traité (tous les CSV étaient manquants).")

    # ---- 6. Écriture du fichier manifeste PVD final ----
    with open(output_pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for t_val, vtu_file in processed_steps:
            f.write(f'    <DataSet timestep="{t_val:.6g}" group="" part="0" file="{vtu_file}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

    print(f"[Succès] Série temporelle globale projetée générée : {output_pvd_path}")

if __name__ == "__main__":

    import os


    # =========================================================================
    # 3. SÉCURITÉ ET LANCEMENT DU TRAITEMENT
    # =========================================================================
    print("=" * 60)
    print("  VÉRIFICATION ET LANCEMENT DE LA PROJECTION TEMPORELLE")
    print("=" * 60)

    try:
        # Exécution de la fonction globale de traitement
        process_csv_series_to_cad_mesh(
            folder_path="/home/pmarechal/Documents/synthetic_csv/fenicsx_surface_z0_y7_csv",
            file_prefix="FE_z0_y7_step_", 
            mesh_cad_path="Flat_specimen_refined.msh", 
            tform_h5_to_cad_4D = np.identity(4), 
            output_pvd_path = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
            alpha=20.0,
            ech=1,
            start_idx = 0,
            end_idx = 52,
        )
        print("\n" + "=" * 60)
        print("[Succès] Traitement terminé sans accroc.")
        print(f"[Aide] Vous pouvez maintenant ouvrir dans ParaView")
        print("       pour visualiser le déplacement projeté sur la CAO au cours du temps.")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "!" * 60)
        print("[Échec] Une erreur est survenue pendant l'interpolation :")
        print("!" * 60)
        import traceback
        traceback.print_exc()




