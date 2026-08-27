"""
dic_importation.py
------------------
Functions for importing DIC (Digital Image Correlation) displacement data
from CSV files and projecting them onto a CAD mesh for use in FEM simulations.

Processing Pipeline
~~~~~~~~~~~~~~~~~~~
1. `create_reference_mesh_from_csv`: Reads raw point coordinates (x, y) and performs 2D Delaunay
   triangulation with alpha-shape boundary filtering to build the reference observation mesh.
2. `read_msh_safely`: Reads Gmsh mesh file and isolates 3D tetrahedral elements (VTK type 10).
3. `update_displacement_field_pyvista`: Assigns point displacement vectors from step CSV to PyVista mesh.
4. `interpolate_displacement_obs_mesh_to_cad_mesh_2D`: Maps CAD nodes into image coordinate space
   via homogeneous 4D transform, performs KDTree nearest-neighbor spatial interpolation, applies
   Jacobian inverse matrix transformation to displacement vectors, and flags valid imported nodes.
5. `process_csv_series_to_cad_mesh`: Batches the entire time-series projection across all CSV files
   and exports VTU files with a PVD XML manifest.
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
def interpolate_displacement_obs_mesh_to_cad_mesh_2D(
    mesh_obs: pv.DataSet,
    mesh_cad_target: pv.DataSet,
    tform_cad_to_img_4D: np.ndarray,
    distance_threshold: Optional[float] = None,
) -> None:
    """
    Project experimental DIC displacement vectors onto CAD mesh nodes.

    Transformation Procedure:
      1. Map CAD 3D nodes into experimental image space via 4D homogeneous matrix:
         x_img = T_cad_to_img @ x_cad
      2. Construct 2D KDTree on observation mesh point coordinates.
      3. Query nearest-neighbor observation index for each CAD node.
      4. Transform interpolated image displacement vectors back to CAD coordinate space
         using inverse Jacobian matrix: u_cad = J^(-1) @ u_img
      5. Save result to CAD mesh point data as `displacement_projected`.

    Parameters
    ----------
    mesh_obs : pv.DataSet
        Observation mesh holding `displacement` point data.
    mesh_cad_target : pv.DataSet
        Target CAD mesh updated in-place.
    tform_cad_to_img_4D : np.ndarray (4, 4)
        Homogeneous spatial transformation matrix mapping CAD space to Image space.
    distance_threshold : float, optional
        Maximum distance threshold beyond which CAD nodes are set to NaN.
    """
    if "displacement" not in mesh_obs.point_data:
        raise ValueError("Le maillage observé ne contient pas de champ 'displacement'.")

    # 1. Transform CAD node coordinates into homogeneous 4D space
    points_cad     = mesh_cad_target.points
    points_cad_hom = np.hstack([points_cad, np.ones((len(points_cad), 1))])
    points_img     = (tform_cad_to_img_4D @ points_cad_hom.T).T[:, :3]

    points_obs_2d  = mesh_obs.points[:, :2]
    points_img_2d  = points_img[:, :2]

    # 2. KDTree spatial search for nearest neighbor observation point
    tree_obs = KDTree(points_obs_2d)

    if distance_threshold is not None:
        distances, indices = tree_obs.query(
            points_img_2d, k=1, distance_upper_bound=distance_threshold
        )
        valid   = np.isfinite(distances)
        indices = np.where(valid, indices, 0)
        disp_obs = mesh_obs.point_data["displacement"][indices].astype(float)
        disp_obs[~valid] = np.nan
    else:
        _, indices = tree_obs.query(points_img_2d, k=1)
        disp_obs   = mesh_obs.point_data["displacement"][indices]

    # 3. Transform vector displacements back into CAD coordinate frame
    tform_jac_cad_to_img_3D = tform_cad_to_img_4D[:3, :3]
    inv_jac                 = np.linalg.inv(tform_jac_cad_to_img_3D)
    disp_cad                = (inv_jac @ disp_obs.T).T

    mesh_cad_target.point_data["displacement_projected"] = disp_cad


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
    Execute batch time-series projection of DIC CSV displacement files onto global CAD mesh.

    Parameters
    ----------
    folder_path : str
        Directory containing DIC timestep CSV files.
    file_prefix : str
        CSV filename prefix (e.g. "FE_step_").
    mesh_cad_path : str
        Path to CAD mesh (.msh or VTK file).
    tform_img_to_cad_4D : np.ndarray (4, 4)
        Homogeneous transformation matrix mapping DIC space to CAD space.
    output_pvd_path : str
        Output path for the generated PVD collection manifest.
    alpha : float
        Delaunay alpha-shape threshold parameter.
    ech : int
        Timestep subsampling stride factor (subsample 1 file every `ech` steps).
    start_idx : int
        First timestep index to process.
    end_idx : int, optional
        Last timestep index to process (None = process all available steps).
    distance_threshold : float
        Proximity distance threshold marking CAD nodes as valid (`is_imported = 0.1`).
    """
    # 1. Discover matching CSV files in directory
    search_pattern = os.path.join(folder_path, f"{file_prefix}[0-9][0-9][0-9][0-9].csv")
    all_files      = glob.glob(search_pattern)
    if not all_files:
        raise FileNotFoundError(
            f"Aucun fichier {file_prefix}XXXX.csv trouvé dans {folder_path}."
        )

    steps: List[int] = sorted(
        int(m.group(1)) for f in all_files if (m := re.search(r"(\d{4})\.csv$", f))
    )
    total_steps = len(steps)
    steps       = steps[start_idx:end_idx]
    if not steps:
        raise ValueError(
            f"L'intervalle [{start_idx}:{end_idx}] ne contient aucune donnée "
            f"(total : {total_steps})."
        )

    steps_to_process   = steps[::ech]
    min_step, max_step = steps_to_process[0], steps_to_process[-1]

    first_csv = os.path.join(folder_path, f"{file_prefix}{min_step:04d}.csv")
    if not os.path.exists(first_csv):
        raise FileNotFoundError(f"Fichier de référence introuvable : {first_csv}")

    print(
        f"[Info] {total_steps} pas de temps trouvés, {len(steps_to_process)} retenus "
        f"({min_step:04d} -> {max_step:04d}, ech={ech})."
    )

    # 2. Build reference observation mesh from initial CSV
    print(f"[1/3] Création du maillage de référence depuis {first_csv}...")
    mesh_obs   = create_reference_mesh_from_csv(first_csv, alpha=alpha)
    points_obs = mesh_obs.points

    # 3. Load global CAD mesh & compute proximity mask `is_imported`
    print(f"[2/3] Chargement du maillage CAD : {mesh_cad_path}")
    if mesh_cad_path.lower().endswith(".msh"):
        mesh_cad = read_msh_safely(mesh_cad_path)
    else:
        mesh_cad = pv.read(mesh_cad_path)

    points_hom         = np.hstack([points_obs, np.ones((points_obs.shape[0], 1))])
    points_obs_in_cad  = (tform_h5_to_cad_4D @ points_hom.T).T[:, :3]
    tree_obs_2d        = KDTree(points_obs_in_cad[:, :2])

    cad_distances, _   = tree_obs_2d.query(
        mesh_cad.points[:, :2], distance_upper_bound=distance_threshold
    )
    is_imported        = cad_distances <= distance_threshold

    if not np.any(is_imported):
        raise ValueError(
            "Aucun point du maillage CAD ne correspond aux données observées "
            f"(seuil = {distance_threshold})."
        )

    mesh_cad.point_data["is_imported"] = np.where(is_imported, 0.1, 0.0)
    outside_mask = ~is_imported

    tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

    # 4. Iterate over timesteps: interpolate, mask unobserved nodes, save VTU files
    output_dir = os.path.dirname(output_pvd_path) or "."
    pvd_stem   = os.path.splitext(os.path.basename(output_pvd_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[3/3] Projection de {len(steps_to_process)} pas de temps sur le CAD "
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
        # Force unobserved nodes outside distance threshold to 0.0
        mesh_cad.point_data["displacement_projected"][outside_mask] = 0.0

        vtu_filename = f"{pvd_stem}_{step:04d}.vtu"
        mesh_cad.save(os.path.join(output_dir, vtu_filename))
        processed_steps.append((float(step), vtu_filename))
        print(f" -> t={step:04d} projeté sur CAD ({csv_path}).")

    if not processed_steps:
        raise RuntimeError(
            "Aucun pas de temps n'a pu être traité (tous les CSV étaient manquants)."
        )

    # 5. Write PVD XML collection manifest file
    with open(output_pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for t_val, vtu_file in processed_steps:
            f.write(
                f'    <DataSet timestep="{t_val:.6g}" group="" part="0" file="{vtu_file}"/>\n'
            )
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

    print(f"[Succès] Série temporelle globale projetée : {output_pvd_path}")
