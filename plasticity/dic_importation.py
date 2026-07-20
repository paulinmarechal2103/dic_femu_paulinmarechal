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


def update_displacement_field_pyvista(csv_path: str, mesh: pv.PolyData):
    """Reads displacements from CSV and assigns them directly as Point Data on the mesh."""
    df = pd.read_csv(csv_path)
    
    # Flexible matching for displacement column names
    ux_col = next((c for c in df.columns if c.lower() in ['u', 'u_x', 'disp_x']), None)
    uy_col = next((c for c in df.columns if c.lower() in ['v', 'u_y', 'disp_y']), None)
    uz_col = next((c for c in df.columns if c.lower() in ['w', 'u_z', 'disp_z']), None)
    
    if not ux_col or not uy_col:
        raise ValueError(f"Could not find displacement columns (ux, uy) in {csv_path}. Columns: {df.columns.tolist()}")
        
    ux = df[ux_col].values
    uy = df[uy_col].values
    uz = df[uz_col].values if uz_col else np.zeros_like(ux)
    
    # Build 3D displacement vector field for VTK
    disp_vectors = np.column_stack((ux, uy, uz))
    
    # Store the field on the mesh. Point indexing is preserved by PyVista's delaunay_2d.
    mesh.point_data["displacement"] = disp_vectors


def process_csv_series_pyvista(
    folder_path: str, 
    output_pvd: str, 
    file_prefix: str, 
    alpha: float = 0.2, 
    ech: int = 20,
    start_idx: int = 0,
    end_idx: Optional[int] = None
):
    """Boucle sur les fichiers CSV en utilisant PyVista pour l'écriture de la série temporelle (.pvd).

    Ne calcule et n'écrit que le champ de déplacement (displacement).
    S'adapte dynamiquement au nombre de fichiers présents dans le dossier,
    et permet de limiter la plage d'images lues via start_idx et end_idx.
    """
    # 1. Détection dynamique des étapes disponibles dans le dossier
    search_pattern = os.path.join(folder_path, f"{file_prefix}[0-9][0-9][0-9][0-9].csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier correspondant au motif {file_prefix}XXXX.csv n'a été trouvé dans {folder_path}.")
    
    # Extraction des indices numériques
    steps = []
    for f in all_files:
        match = re.search(r"(\d{4})\.csv$", f)
        if match:
            steps.append(int(match.group(1)))
            
    # Tri obligatoire pour garantir l'ordre chronologique avant le découpage
    steps.sort()
    total_steps = len(steps)
    
    # --- Application des limites de sélection ---
    steps = steps[start_idx:end_idx]
    
    if not steps:
        raise ValueError(f"L'intervalle [{start_idx}:{end_idx}] ne contient aucune donnée (Total disponible : {total_steps}).")
        
    # Application du pas d'échantillonnage (ech) sur la sélection
    steps_to_process = steps[::ech]
    
    min_step = steps_to_process[0]
    max_step = steps_to_process[-1]
    
    # Utilisation du premier pas trouvé (après filtrage) comme référence géométrique
    first_csv = os.path.join(folder_path, f"{file_prefix}{min_step:04d}.csv")
    if not os.path.exists(first_csv):
        raise FileNotFoundError(f"Le fichier de référence initial {first_csv} est introuvable.")
        
    print(f"[Info] Fichiers globaux trouvés : {total_steps}")
    print(f"[Info] Fichiers conservés pour traitement (après intervalle et ech={ech}) : {len(steps_to_process)} (Etapes de {min_step:04d} à {max_step:04d})")
    print(f"[Info] Création du maillage de référence depuis {first_csv}...")
    
    # Création du maillage initial avec PyVista
    mesh = create_reference_mesh_from_csv(first_csv, alpha=alpha)

    # Préparation du répertoire de sortie
    output_dir = os.path.dirname(output_pvd) or "."
    pvd_filename = os.path.basename(output_pvd)
    pvd_name_no_ext, _ = os.path.splitext(pvd_filename)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[Info] Début du traitement de la série dans {output_pvd}...")
    
    processed_steps = []  # Utilisé pour construire le fichier de métadonnées .pvd à la fin

    # Traitement chronologique de la série filtrée
    for step in steps_to_process:
        csv_name = f"{file_prefix}{step:04d}.csv"
        csv_path = os.path.join(folder_path, csv_name)

        if not os.path.exists(csv_path):
            print(f"[Attention] Pas de temps {step} manquant ({csv_name}). Saut.")
            continue

        t = float(step)

        # 1. Mise à jour du déplacement depuis le CSV directement sur le maillage PyVista
        update_displacement_field_pyvista(csv_path, mesh)
        
        # 2. Sauvegarde du pas de temps sous forme de fichier VTU (Unstructured/PolyData) individuel
        vtu_filename = f"{pvd_name_no_ext}_{step:04d}.vtu"
        vtu_path = os.path.join(output_dir, vtu_filename)
        mesh.cast_to_unstructured_grid().save(vtu_path)
        
        # Enregistrement pour le fichier manifeste .pvd
        processed_steps.append((t, vtu_filename))
        print(f" -> Étape {step}/{max_step} traitée ({csv_name} au temps t={t})")

    # 3. Écriture du fichier manifeste .pvd qui unifie tous les pas de temps dans ParaView
    with open(output_pvd, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t, vtu_file in processed_steps:
            f.write(f'    <DataSet timestep="{t}" group="" part="0" file="{vtu_file}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"[Succès] Série temporelle (déplacement seul) générée dans : {output_pvd}")


def interpolate_displacement_obs_mesh_to_cad_mesh_2D(
    mesh_obs: pv.DataSet, 
    mesh_cad: pv.DataSet, 
    tform_cad_to_img_4D: np.ndarray
) -> None:
    """Interpolates the 'displacement' point data from mesh_obs to mesh_cad
    using spatial mapping and transforms the vectors back to the CAD space.
    """
    if "displacement" not in mesh_obs.point_data:
        raise ValueError("The observed mesh does not contain a 'displacement' array.")

    # 1. Get the spatial coordinates of the CAD mesh points
    points_cad = mesh_cad.points  # shape (N, 3)
    
    # Transform CAD coordinates into the Observation (Image) Coordinate space
    points_cad_hom = np.hstack([points_cad, np.ones((len(points_cad), 1))])
    points_img_hom = (tform_cad_to_img_4D @ points_cad_hom.T).T
    points_img = points_img_hom[:, :3]

    # 2. Build a fast KDTree on the observed mesh points for interpolation
    # (In 2D we map coordinates in the X-Y plane)
    points_obs_2d = mesh_obs.points[:, :2]
    points_img_2d = points_img[:, :2]
    
    tree_obs = KDTree(points_obs_2d)
    
    # Query nearest neighbors on the observed mesh
    distances, indices = tree_obs.query(points_img_2d, k=1)

    # Grab the closest displacements
    disp_obs = mesh_obs.point_data["displacement"][indices]  # Shape (N, 3)

    # 3. Transform the displacement vectors back from Image Space to CAD space
    # Using the inverse jacobian (top-left 3x3 of cad_to_img, inverted)
    tform_jac_cad_to_img_3D = tform_cad_to_img_4D[:3, :3]
    inv_jacobian = np.linalg.inv(tform_jac_cad_to_img_3D)
    
    # Apply rotation/scaling transform to vectors
    disp_cad_transformed = (inv_jacobian @ disp_obs.T).T

    # Save directly back to the CAD mesh point data
    mesh_cad.point_data["displacement_projected"] = disp_cad_transformed


import meshio

def read_msh_safely(msh_path: str) -> pv.UnstructuredGrid:
    mesh = meshio.read(msh_path)
    mesh.cell_sets = {}
    pv_mesh = pv.from_meshio(mesh)
    
    # --- AJOUT : Ne conserver que les tétraèdres (Type 10) ---
    # VTK_TETRA est le type 10
    pv_mesh = pv_mesh.extract_cells(pv_mesh.celltypes == 10)
    
    return pv_mesh

def project_vtu_series_to_cad_mesh_mask(
    input_pvd_path: str, 
    mesh_cad_path: str, 
    tform_h5_to_cad_4D: np.ndarray, 
    output_pvd_path: str
) -> None:
    """Reads a time series of PyVista VTU/PVD meshes containing displacements,
    reconstructs the geometry, projects/transforms the vectors onto a masked sub-region 
    of a CAD mesh, and writes a new PyVista temporal PVD/VTU series.
    """
    # =========================================================================
    # 1. PARSING INPUT PVD / IDENTIFYING TIMESTEPS
    # =========================================================================
    print(f"[1/4] Parsing input PVD metadata: {input_pvd_path}")
    if not os.path.exists(input_pvd_path):
        raise FileNotFoundError(f"Input PVD file {input_pvd_path} not found.")

    input_dir = os.path.dirname(input_pvd_path) or "."
    steps = [] # Will hold tuples of (time, absolute_vtu_path)

    tree = ET.parse(input_pvd_path)
    root = tree.getroot()
    for dataset in root.findall(".//DataSet"):
        timestep = float(dataset.get("timestep"))
        vtu_rel_path = dataset.get("file")
        vtu_abs_path = os.path.join(input_dir, vtu_rel_path)
        steps.append((timestep, vtu_abs_path))

    if not steps:
        raise ValueError("No timesteps could be located in the input PVD file.")
    
    steps.sort(key=lambda x: x[0])
    print(f" -> {len(steps)} timesteps successfully identified.")

    # =========================================================================
    # 2. CHARGEMENT DU MAILLAGE CAD ET CALCUL DES MASQUES
    # =========================================================================
    print(f"[2/4] Loading CAD reference mesh: {mesh_cad_path}")
    
    if mesh_cad_path.lower().endswith(".msh"):
        mesh_cad = read_msh_safely(mesh_cad_path)
    else:
        mesh_cad = pv.read(mesh_cad_path)
    
    # Read the first timestep's mesh geometry to determine physical bounds
    first_obs_mesh = pv.read(steps[0][1])
    points_obs = first_obs_mesh.points

    # Transform observed points into CAD space to find the bounding region
    points_hom = np.hstack([points_obs, np.ones((points_obs.shape[0], 1))])
    points_obs_in_cad = (tform_h5_to_cad_4D @ points_hom.T).T[:, :3]

    # Build KDTree using CAD-space projected coordinates
    points_obs_2d = points_obs_in_cad[:, :2]
    tree_obs_2d = KDTree(points_obs_2d)

    # Query nearest neighbors for CAD cell centers to find active regions
    cell_centers = mesh_cad.cell_centers().points
    DISTANCE_THRESHOLD = 1.0

    distances, _ = tree_obs_2d.query(cell_centers[:, :2], distance_upper_bound=DISTANCE_THRESHOLD)
    cell_mask_initial = distances <= DISTANCE_THRESHOLD

    # Extend selection along the full width of the Y slices matching bounds
    if np.any(cell_mask_initial):
        active_y = cell_centers[cell_mask_initial, 1]
        y_min, y_max = np.min(active_y), np.max(active_y)
        
        print(f" -> Extending active CAD submesh along Y range: [{y_min:.2f}, {y_max:.2f}]")
        
        # Identify all cells residing inside this Y window
        tol = 1e-5
        cell_mask_extended = (cell_centers[:, 1] >= y_min - tol) & (cell_centers[:, 1] <= y_max + tol)
        cell_indices = np.where(cell_mask_extended)[0]
    else:
        raise ValueError("No matching cells found on CAD mesh within the spatial distance threshold.")

    # Extract the physical CAD submesh volumetrically
    submesh_volume = mesh_cad.extract_cells(cell_indices)

    # Compute static 2D proximity mask to save as 'is_imported' (0.1 inside boundary, 0.0 outside)
    submesh_points_2d = submesh_volume.points[:, :2]
    submesh_distances, _ = tree_obs_2d.query(submesh_points_2d, distance_upper_bound=DISTANCE_THRESHOLD)
    is_imported_mask = np.where(submesh_distances <= DISTANCE_THRESHOLD, 0.1, 0.0)
    submesh_volume.point_data["is_imported"] = is_imported_mask

    # =========================================================================
    # 3. INVERTING SYSTEM TRANSFORMS
    # =========================================================================
    print("[3/4] Inverting spatial transformation matrices...")
    tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

    # =========================================================================
    # 4. TEMPORAL PROJECTION AND SAVING NEW SERIES
    # =========================================================================
    output_dir = os.path.dirname(output_pvd_path) or "."
    pvd_filename = os.path.basename(output_pvd_path)
    pvd_name_no_ext, _ = os.path.splitext(pvd_filename)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[4/4] Projecting timesteps onto CAD submesh -> saving to {output_pvd_path}")
    
    processed_steps = []

    for i, (t, vtu_file_path) in enumerate(steps):
        # Load timestep observation dataset
        obs_step_mesh = pv.read(vtu_file_path)

        # Interpolate displacement variables to CAD submesh
        interpolate_displacement_obs_mesh_to_cad_mesh_2D(
            mesh_obs=obs_step_mesh,
            mesh_cad=submesh_volume,
            tform_cad_to_img_4D=tform_cad_to_img_4D
        )

        # Save this step as a standalone .vtu file
        vtu_out_name = f"{pvd_name_no_ext}_{i:04d}.vtu"
        vtu_out_path = os.path.join(output_dir, vtu_out_name)
        submesh_volume.save(vtu_out_path)

        processed_steps.append((t, vtu_out_name))
        print(f" -> Time t={t:.2f} successfully projected.")

    # Save final unifying PVD index file
    with open(output_pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t_val, relative_vtu_file in processed_steps:
            f.write(f'    <DataSet timestep="{t_val}" group="" part="0" file="{relative_vtu_file}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"[Success] Time series successfully generated: {output_pvd_path}")


def resample_h5_time_series(
    input_h5_path: str,
    output_xdmf_path: str,
    target_num: int,
    kind: str = "linear"
) -> None:
    """Lit une série temporelle depuis un fichier H5, interpole le champ de déplacement
    dans le temps pour atteindre exactement N pas de temps (target_num_steps), 
    et sauvegarde le résultat dans un nouveau couple de fichiers XDMF/H5 natifs FEniCSx.
    Les pas de temps sont enregistrés sous forme d'index entiers (0, 1, 2...).
    """
    # =========================================================================
    # 1. LECTURE DE LA STRUCTURE ET COLLECTE DES DONNÉES D'ORIGINE
    # =========================================================================
    target_num_steps = target_num + 1
    xdmf_in_path = input_h5_path.replace(".h5", ".xdmf")
    steps = []
    geom_path = "/Mesh/mesh/geometry"
    topo_path = "/Mesh/mesh/topology"
    
    if os.path.exists(xdmf_in_path):
        print(f"[1/4] Analyse de {xdmf_in_path}...")
        tree = ET.parse(xdmf_in_path)
        root = tree.getroot()
        try:
            geom_path = next(root.iter("Geometry")).find("DataItem").text.split(":")[-1].strip()
            topo_path = next(root.iter("Topology")).find("DataItem").text.split(":")[-1].strip()
        except Exception:
            pass

        for grid in root.findall(".//Grid"):
            time_node = grid.find("Time")
            attr_node = None
            for attr in grid.findall(".//Attribute"):
                if "displacement" in attr.get("Name", ""):
                    attr_node = attr
                    break
                    
            if time_node is not None and attr_node is not None:
                t_val = float(time_node.get("Value"))
                di = attr_node.find("DataItem")
                if di is not None:
                    steps.append((t_val, di.text.split(":")[-1].strip()))
    else:
        raise FileNotFoundError(f"Le fichier de métadonnées associé {xdmf_in_path} est requis.")

    if not steps:
        raise ValueError(f"Aucun dataset de déplacement n'a été trouvé dans {xdmf_in_path}.")

    steps.sort(key=lambda x: x[0])
    t_orig = np.array([s[0] for s in steps])
    print(f" -> {len(t_orig)} pas de temps d'origine détectés (de t={t_orig[0]} à t={t_orig[-1]})")

    # =========================================================================
    # 2. CHARGEMENT ET DÉTECTION DYNAMIQUE DES DIMENSIONS
    # =========================================================================
    with h5py.File(input_h5_path, "r") as f:
        try:
            points_obs = f[geom_path][:]
            cells_obs = f[topo_path][:]
        except KeyError as e:
            print(f"Disponible sous les clés : {list(f[geom_path].keys())}")
            raise KeyError(f"Erreur lors de la lecture des chemins de maillage : {e}")
        # Détection de la dimension géométrique (ex: 3 pour X, Y, Z)
        gdim = points_obs.shape[1]
        
        # Détection automatique du type de cellule selon le nombre de nœuds par maille
        nodes_per_cell = cells_obs.shape[1]
        if nodes_per_cell == 4:
            cell_type = "tetrahedron"
        elif nodes_per_cell == 3:
            cell_type = "triangle"
        else:
            raise ValueError(f"Type de cellule non supporté : {nodes_per_cell} nœuds par élément.")
        
        # Détection de la dimension du champ de déplacement (ex: 2 pour UX, UY)
        first_ds_path = steps[0][1]
        dim_disp = f[first_ds_path].shape[1]
        num_nodes = f[first_ds_path].shape[0]
        
        print(f"[2/4] Structure détectée : Éléments '{cell_type}' ({gdim}D) | Déplacement {dim_disp}D")
        print(f"      Chargement de la matrice globale en RAM...")
        
        all_disp_orig = np.zeros((len(t_orig), num_nodes, dim_disp))
        for idx, (_, ds_path) in enumerate(steps):
            all_disp_orig[idx, :, :] = f[ds_path][:, :dim_disp]

    # =========================================================================
    # 3. INTERPOLATION TEMPORELLE
    # =========================================================================
    print(f"[3/4] Calcul de l'interpolation temporelle ({kind}) vers {target_num_steps} pas...")
    t_target = np.linspace(t_orig[0], t_orig[-1], target_num_steps)
    interpolator = interp1d(t_orig, all_disp_orig, axis=0, kind=kind, bounds_error=False, fill_value="extrapolate")
    all_disp_target = interpolator(t_target)

    # =========================================================================
    # 4. RECONSTRUCTION DU MAILLAGE ET EXPORT SÉCURISÉ
    # =========================================================================
    print(f"[4/4] Écriture de la nouvelle série temporelle : {output_xdmf_path}")
    
    # Instanciation avec le type de cellule et la dimension correcte
    coord_element = basix.ufl.element("Lagrange", cell_type, 1, shape=(gdim,))
    domain_obs = ufl.Mesh(coord_element)
    mesh_obs = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells=cells_obs, x=points_obs, e=domain_obs)

    # Espace de fonction adapté à la dimension du déplacement (2D) sur le maillage (3D)
    V_obs = fem.functionspace(mesh_obs, ("CG", 1, (dim_disp,)))
    u_obs = fem.Function(V_obs, name="displacement_projected")

    # Arbre spatial pour faire correspondre les coordonnées physiques sans dépendre de l'ordre des DOFs
    tree = KDTree(points_obs)

    with dolfinx.io.XDMFFile(mesh_obs.comm, output_xdmf_path, "w") as xdmf_out:
        xdmf_out.write_mesh(mesh_obs)
        
        for i, t in enumerate(t_target):
            disp_step = np.nan_to_num(all_disp_target[i, :, :], nan=0.0)

            # Le callback reçoit les coordonnées x sous la forme (3, num_points)
            def interpolate_callback(x):
                points_to_query = x[:gdim, :].T
                _, indices = tree.query(points_to_query)
                # Retourne une forme (dim_disp, num_points) attendue par DOLFINx
                return disp_step[indices].T

            u_obs.interpolate(interpolate_callback)
            u_obs.x.scatter_forward()
            xdmf_out.write_function(u_obs, float(i))

    print(f"[Succès] Nouveau fichier temporel généré avec succès ({target_num_steps} pas).")



# if __name__ == "__main__":
#     dossier_csv = "/home/pmarechal/Documents/synthetic_csv/fenicsx_surface_z0_csv"
#     file_prefix = "FE_z0_step_"

#     process_csv_series_pyvista(
#         folder_path=dossier_csv, 
#         output_pvd="MAINTEST/pyvista_exports/csv_imports/dic_series.pvd", 
#         file_prefix=file_prefix, 
#         alpha=20.0,
#         ech=1,
#         start_idx = 0,
#         end_idx = 52,
#     )



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
        project_vtu_series_to_cad_mesh_mask(
            input_pvd_path = "MAINTEST/pyvista_exports/csv_imports/dic_series.pvd", 
            mesh_cad_path="Flat_specimen_refined.msh", 
            tform_h5_to_cad_4D = np.identity(4), 
            output_pvd_path = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
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




