"""
dic_importation.py
===================

2-D Digital Image Correlation (DIC) data importation and projection utilities.

Purpose
-------
This module imports 2-D DIC displacement fields from CSV time-series files and
projects them onto 3-D volumetric CAD meshes for finite element (FEM) analysis.
It maps data between image coordinate space and CAD space using 4x4 homogeneous
transformation matrices and local Jacobian vector transformations:

    x_img = T_cad_to_img @ x_cad      (using homogeneous column vectors)
    u_cad = J_inv @ u_img             (where J_inv = inv(T_cad_to_img[:3, :3]))

Pipeline overview
------------------
Data processing and projection are performed in five structured stages:

1. Observation mesh construction
   (`create_reference_mesh_from_csv`)
   Reads initial DIC point coordinates (x, y, optional z) and performs 2-D Delaunay
   triangulation with alpha-shape radius filtering to generate a 2-D surface mesh
   matching non-convex specimen boundaries.

2. Safe CAD mesh loading
   (`read_msh_safely`)
   Loads Gmsh `.msh` files via `meshio` and isolates 3-D volumetric elements
   (`VTK_TETRA` type 10 and `VTK_HEXAHEDRON` type 12), filtering out line and surface
   elements.

3. Displacement field assignment
   (`update_displacement_field_pyvista`)
   Parses displacement vector components (u, v, w) from a timestep CSV file and
   assigns them in-place to the observation PyVista mesh point data.

4. 2-D spatial interpolation & vector projection
   (`interpolate_displacement_obs_mesh_to_cad_mesh_2D_linear`)
   Projects target CAD nodes into 2-D image coordinate space using `T_cad_to_img`.
   Performs 2-D linear interpolation (`SciPy`) within the observation convex hull
   and nearest-neighbor extrapolation outside it. Converts 2-D vectors back into
   CAD space using the inverse 3-D rotation/scale matrix `J_inv`.

5. Time-series batch projection
   (`process_csv_series_to_cad_mesh`)
   Executes the full pipeline across a sequence of timesteps. Uses a 2-D `KDTree`
   to flag valid node coverage (`is_imported` and `distance_threshold`), interpolates
   displacements, and exports individual VTU files alongside a master XML PVD
   manifest.

Public API
----------
`create_reference_mesh_from_csv`                         -- build a 2-D observation surface mesh from CSV coordinates.
`read_msh_safely`                                        -- read a Gmsh file and extract 3-D volumetric elements (tet/hex).
`update_displacement_field_pyvista`                      -- attach displacement vectors from a CSV file to a PyVista mesh.
`interpolate_displacement_obs_mesh_to_cad_mesh_2D_linear` -- interpolate 2-D observation vectors onto 3-D CAD nodes.
`process_csv_series_to_cad_mesh`                         -- batch process a CSV time-series to VTU files and XML PVD manifest.

Notes on coordinate transformations
-----------------------------------
The pipeline requires a 4x4 homogeneous matrix `tform_img_to_cad_4D` mapping DIC
image coordinates to CAD space. Its inverse (`tform_cad_to_img_4D`) is computed
internally to project 3-D CAD node positions back into 2-D image space for spatial
interpolation.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import glob
import os
import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Scientific / data stack
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import meshio
import pyvista as pv
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# 1. Reference mesh from CSV
# ---------------------------------------------------------------------------
def create_reference_mesh_from_csv(csv_path: str, alpha: float) -> pv.PolyData:
    """
    Create a 2D triangulated PyVista surface mesh from CSV point coordinates.

    Uses PyVista's 2D Delaunay triangulation algorithm with alpha-shape filtering
    to remove long boundary edges in non-convex specimen geometries.

    Parameters
    ----------
    csv_path : str
        Path to the reference CSV file containing coordinate columns (x, y, z).
    alpha : float
        Alpha-shape radius threshold for Delaunay boundary filtering.

    Returns
    -------
    mesh : pv.PolyData
        Triangulated observation surface mesh.
    """
    df = pd.read_csv(csv_path)

    # Detect coordinate column names flexibly
    x_col = next((c for c in df.columns if c.lower() in ("x", "pos_x", "coord_x","  \"x\"")), None)
    y_col = next((c for c in df.columns if c.lower() in ("y", "pos_y", "coord_y","  \"y\"")), None)
    z_col = next((c for c in df.columns if c.lower() in ("z", "pos_z", "coord_z")), None)

    if not x_col or not y_col:
        raise ValueError(
            f"Could not find coordinate columns (x, y) in {csv_path}. "
            f"Columns: {df.columns.tolist()}"
        )

    # Convert coordinates to 3D point array (N, 3) for PyVista PolyData
    points = np.zeros((len(df), 3))
    points[:, 0] = df[x_col].values
    points[:, 1] = df[y_col].values
    if z_col:
        points[:, 2] = df[z_col].values

    poly = pv.PolyData(points)
    return poly.delaunay_2d(alpha=alpha)


# ---------------------------------------------------------------------------
# 2. Safe mesh reader (keeps only tetrahedra from Gmsh .msh files)
# ---------------------------------------------------------------------------
def read_msh_safely(msh_path: str) -> pv.UnstructuredGrid:
    """
    Read a Gmsh .msh file using meshio and extract volumetric cells (tetrahedral & hexahedral).

    Filters out surface/line elements, preserving VTK_TETRA (type 10) 
    and VTK_HEXAHEDRON (type 12) cells.

    Parameters
    ----------
    msh_path : str
        Path to the Gmsh .msh mesh file.

    Returns
    -------
    pv_mesh : pv.UnstructuredGrid
        PyVista mesh containing tetrahedral and hexahedral cells.
    """
    mesh = meshio.read(msh_path)
    mesh.cell_sets = {}
    pv_mesh = pv.from_meshio(mesh)
    
    # VTK_TETRA = 10, VTK_HEXAHEDRON = 12
    is_volumetric = np.isin(pv_mesh.celltypes, [10, 12])
    pv_mesh = pv_mesh.extract_cells(is_volumetric)
    
    return pv_mesh


# ---------------------------------------------------------------------------
# 3. Assign displacement data from CSV to a PyVista mesh (in-place)
# ---------------------------------------------------------------------------
def update_displacement_field_pyvista(csv_path: str, mesh: pv.DataSet) -> None:
    """
    Read displacement vectors from a CSV file and store them as mesh point data.

    Parameters
    ----------
    csv_path : str
        Path to the displacement CSV file.
    mesh : pv.DataSet
        Target PyVista mesh updated in-place with `point_data["displacement"]`.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    ux_col = next((c for c in df.columns if c.lower() in ("u", "u_x", "disp_x",'  "u"')), None)
    uy_col = next((c for c in df.columns if c.lower() in ("v", "u_y", "disp_y", '  "v"')), None)
    uz_col = next((c for c in df.columns if c.lower() in ("w", "u_z", "disp_z")), None)

    if ux_col is None or uy_col is None:
        raise ValueError(
            f"Colonnes de déplacement (ux, uy) introuvables dans {csv_path}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    if len(df) != mesh.n_points:
        raise ValueError(
            f"Incohérence de taille entre {csv_path} ({len(df)} lignes) et le maillage "
            f"({mesh.n_points} points). Le CSV doit correspondre point à point au maillage."
        )

    ux = df[ux_col].to_numpy()
    uy = df[uy_col].to_numpy()
    uz = df[uz_col].to_numpy() if uz_col is not None else np.zeros_like(ux)

    mesh.point_data["displacement"] = np.column_stack((ux, uy, uz))


# ---------------------------------------------------------------------------
# 4. Interpolate DIC displacement from observation mesh onto CAD mesh (2-D)
# ---------------------------------------------------------------------------

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def interpolate_displacement_obs_mesh_to_cad_mesh_2D_linear(
    mesh_obs: pv.DataSet,
    mesh_cad_target: pv.DataSet,
    tform_cad_to_img_4D: np.ndarray,
) -> None:
    """
    Interpolate 2D displacement from observation mesh onto target CAD mesh.

    Transforms CAD nodes into image space, applies SciPy 2D linear interpolation
    (with nearest-neighbor extrapolation outside the convex hull), converts 2D vectors
    to 3D, and transforms them back to the CAD coordinate system using the inverse transformation matrix.

    Parameters
    ----------
    mesh_obs : pv.DataSet
        Source observation mesh containing 'displacement' point data.
    mesh_cad_target : pv.DataSet
        Target CAD mesh updated in-place with 'displacement_projected' point data.
    tform_cad_to_img_4D : np.ndarray
        4x4 homogeneous transformation matrix from CAD coordinates to image coordinates.
    """
    if "displacement" not in mesh_obs.point_data:
        raise ValueError(
            "Le maillage observé ne contient pas de champ 'displacement'."
        )

    # 1. Passage des points CAD vers l'espace image (projection 2D)
    points_cad = mesh_cad_target.points
    points_cad_hom = np.hstack([points_cad, np.ones((len(points_cad), 1))])
    points_img = (tform_cad_to_img_4D @ points_cad_hom.T).T[:, :3]

    points_obs_2d = mesh_obs.points[:, :2]
    points_img_2d = points_img[:, :2]

    disp_obs_data = np.asarray(mesh_obs.point_data["displacement"], dtype=float)
    if disp_obs_data.ndim == 1:
        disp_obs_data = disp_obs_data[:, np.newaxis]

    # 2. Interpolation linéaire dans l'enveloppe convexe (triangulation 2D)
    lin_interp = LinearNDInterpolator(points_obs_2d, disp_obs_data)
    disp_interp = lin_interp(points_img_2d)

    # 3. Extrapolation par plus proche voisin pour tout le reste du maillage CAD
    nan_mask = np.isnan(disp_interp)
    if np.any(nan_mask):
        nn_interp = NearestNDInterpolator(points_obs_2d, disp_obs_data)
        disp_extrap = nn_interp(points_img_2d)
        disp_interp[nan_mask] = disp_extrap[nan_mask]

    # 4. Ajustement 3D pour la transformation vectorielle
    if disp_interp.shape[1] == 2:
        disp_interp = np.hstack([disp_interp, np.zeros((len(disp_interp), 1))])

    # 5. Transformation inverse des vecteurs dans le repère CAD
    tform_jac_cad_to_img_3D = tform_cad_to_img_4D[:3, :3]
    inv_jacobian = np.linalg.inv(tform_jac_cad_to_img_3D)
    disp_cad_transformed = (inv_jacobian @ disp_interp.T).T

    mesh_cad_target.point_data["displacement_projected"] = disp_cad_transformed


# ---------------------------------------------------------------------------
# 5. Full pipeline: CSV series → projected PVD on CAD mesh
# ---------------------------------------------------------------------------

def process_csv_series_to_cad_mesh(
    folder_path: str,
    file_prefix: str,
    mesh_cad_path: str,
    tform_img_to_cad_4D: np.ndarray,
    output_pvd_path: str,
    alpha: float = 0.2,
    ech: int = 20,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    distance_threshold: float = 1.0,
) -> None:
    """
    Batch project a time-series of DIC CSV displacement files onto a global CAD mesh.

    Loads time-series DIC data, constructs a 2D reference observation mesh, calculates
    a proximity mask (`is_imported`) on the target CAD nodes using KDTree, and projects
    displacements for each timestep using spatial interpolation. Results are written as a series 
    of VTU files linked by a PVD manifest.

    Parameters
    ----------
    folder_path : str
        Directory containing the CSV files.
    file_prefix : str
        Prefix pattern for input CSV files (e.g., 'disp_').
    mesh_cad_path : str
        Path to target CAD mesh file (.msh, .vtu, etc.).
    tform_img_to_cad_4D : np.ndarray
        4x4 transformation matrix mapping image space coordinates to CAD space.
    output_pvd_path : str
        Destination path for the output XML PVD manifest.
    alpha : float, optional
        Alpha parameter for Delaunay boundary filtering (default is 0.2).
    ech : int, optional
        Subsampling factor for timesteps (default is 20).
    start_idx : int, optional
        Index of the first timestep to process (default is 0).
    end_idx : int, optional
        Index of the last timestep to process (default is None).
    distance_threshold : float, optional
        Maximum distance in 2D to consider a CAD node validly covered by observation data (default is 1.0).
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
    points_obs_in_cad = (tform_img_to_cad_4D @ points_hom.T).T[:, :3]
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
    tform_cad_to_img_4D = np.linalg.inv(tform_img_to_cad_4D)

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

        # Interpolation linéaire à l'intérieur + extrapolation plus proche voisin à l'extérieur
        interpolate_displacement_obs_mesh_to_cad_mesh_2D_linear(
            mesh_obs=mesh_obs,
            mesh_cad_target=mesh_cad,
            tform_cad_to_img_4D=tform_cad_to_img_4D,
        )

        vtu_filename = f"{pvd_name_no_ext}_{step:04d}.vtu"
        mesh_cad.save(os.path.join(output_dir, vtu_filename))
        processed_steps.append((float(step), vtu_filename))
        print(f" -> t={step:04d} projeté et extrapolé sur CAD complet ({csv_path}).")

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