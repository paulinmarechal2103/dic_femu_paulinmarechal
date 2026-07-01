import pandas as pd
import numpy as np
import pyvista as pv
import meshio
import dolfinx.fem as fem
import dolfinx.io
from mpi4py import MPI
from numpy.typing import NDArray
from plasticity_simu import load_and_write_mesh
from image_calibration import calibrate_2d_manual,check_calibration_2d
import os
import ufl
import skimage
import h5py
import basix
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

# =========================================================================
# 1. CONVERSION ET TRIANGULATION NETTOYÉE
# =========================================================================
def csv_to_fenicsx_xdmf(csv_file: str, xdmf_path: str, alpha: float = 0.2) -> None:
    """Lit le CSV de la DIC, filtre les points valides, triangule 

    et écrit le fichier XDMF en ajoutant explicitement les positions x et y.
    """

    data = pd.read_csv(csv_file)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    d = {names[i]: data.values[:, i] for i in range(len(names))}

    # Masque de filtrage : points bien corrélés (u != 0 et non NaN)
    valid_mask = (d["u"] != 0) & (~np.isnan(d["u"]))
    
    x_coords = d["x"][valid_mask]
    y_coords = d["y"][valid_mask]
    u_values = d["u"][valid_mask]
    v_values = d["v"][valid_mask]

    # 1. Construction des points géométriques (N, 3)
    points = np.stack([x_coords, y_coords, np.zeros_like(x_coords)], axis=-1)
    
    # Triangulation avec PyVista
    cloud = pv.PolyData(points)
    volume = cloud.delaunay_2d(alpha=alpha)
    if volume.n_cells == 0:
        print(f"[Attention] Aucun triangle généré avec alpha={alpha}. Bascule sur Delaunay standard...")
        volume = cloud.delaunay_2d(alpha=0.0)

    # Extraction des triangles
    faces_raw = volume.faces
    triangles = faces_raw.reshape(-1, 4)[:, 1:]

    print(f"-> Maillage créé : {len(points)} nœuds et {len(triangles)} triangles.")

    # 2. Formatage de TOUTES les données en colonnes (N, 1) pour Meshio
    u_values_2d = np.array(u_values).reshape(-1, 1)
    v_values_2d = np.array(v_values).reshape(-1, 1)
    x_values_2d = np.array(x_coords).reshape(-1, 1)
    y_values_2d = np.array(y_coords).reshape(-1, 1)

    # 3. Construction du maillage avec déplacements ET positions explicites
    mesh_meshio = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        point_data={
            "x": x_values_2d, # Position X initiale de la DIC rattachée au point
            "y": y_values_2d, # Position Y initiale de la DIC rattachée au point
            "u": u_values_2d,
            "v": v_values_2d,
        }
    )

    # Écriture finale (XDMF + H5)
    meshio.write(xdmf_path, mesh_meshio)
    print(f"[Succès] Fichier {xdmf_path} généré avec les champs ['u', 'v', 'x', 'y'].")

# =========================================================================
# 2. PERMUTATION ET CHARGEMENT DANS FENICSX
# =========================================================================
def permute_array(array: NDArray, perm_indices: NDArray) -> NDArray:
    """Réaligne l'ordre du tableau avec la numérotation dolfinx."""
    # .squeeze() transforme un tableau (N, 1) en un vecteur plat (N,)
    array_flat = np.asarray(array).squeeze() 
    
    # SÉCURITÉ CRITIQUE : Si dolfinx ne fournit pas d'indices de permutation 
    # (tableau vide), cela signifie qu'aucune permutation n'est nécessaire !
    if perm_indices is None or perm_indices.size == 0:
        return array_flat
        
    output = np.zeros_like(array_flat)
    for i in range(len(array_flat)):
        output[i] = array_flat[perm_indices[i]]
    return output


def load_mesh_and_displacement_field(xdmf_path: str):
    """Charge le maillage et crée le champ de déplacement vectoriel u.

    Compatible avec dolfinx v0.10+.
    """
    # A. Lecture de la géométrie par dolfinx
    mesh_XDMF = dolfinx.io.XDMFFile(MPI.COMM_SELF, xdmf_path, "r")
    mesh_dolfinx = mesh_XDMF.read_mesh(name="Grid")
    mesh_dolfinx.topology.create_connectivity(
        mesh_dolfinx.topology.dim - 1, mesh_dolfinx.topology.dim
    )

    # B. Lecture des champs par meshio
    mesh_meshio = meshio.read(xdmf_path)

    print("Champs trouvés par Meshio dans le fichier :", list(mesh_meshio.point_data.keys()))

    if "u" not in mesh_meshio.point_data:
        raise KeyError(f"Le champ 'u' est introuvable dans le fichier XDMF.")

    # C. Application de la permutation sécurisée
    global_indices = mesh_dolfinx.geometry.input_global_indices
    permuted_ux = permute_array(mesh_meshio.point_data["u"], global_indices)
    permuted_uy = permute_array(mesh_meshio.point_data["v"], global_indices)

    # D. Création de la fonction FEniCSx (Vecteur P1 continu)
    # Syntaxe v0.10 : fem.functionspace
    CG1_vector = fem.functionspace(mesh_dolfinx, ("CG", 1, (2,)))
    u_obs = fem.Function(CG1_vector, name="displacement_femu")

    # E. Remplissage du tableau sous dolfinx v0.10
    # On accède directement à .x.array qui est un tableau NumPy plat
    u_array = u_obs.x.array
    u_array[:] = 0.0                  # Initialisation à zéro
    u_array[0::2] = permuted_ux       # Composantes X aux dofs pairs
    u_array[1::2] = permuted_uy       # Composantes Y aux dofs impairs

    # Partage des valeurs si calcul parallèle (remplace ghostUpdate)
    u_obs.x.scatter_forward()

    return mesh_dolfinx, u_obs

import ufl

def compute_strain_tensor(mesh_dolfinx, u_obs):
    """Calcule le tenseur des déformations epsilon dans le plan (2D)

    à partir d'un maillage dont les points ont 3 coordonnées.
    """
    # 1. Calcul du gradient complet (génère une forme 2x3)
    grad_full = ufl.grad(u_obs)

    # 2. Extraction de la sous-matrice carrée 2x2 (uniquement les dérivées par rapport à x et y)
    # grad_full[i, j] où i est la composante de u (0 ou 1) et j est la coordonnée (0, 1 ou 2)
    grad_2d = ufl.as_matrix([
        [grad_full[0, 0], grad_full[0, 1]],
        [grad_full[1, 0], grad_full[1, 1]]
    ])

    # 3. Calcul de la partie symétrique sur la matrice carrée 2x2
    epsilon_expr_ufl = ufl.sym(grad_2d)

    # 4. Création de l'espace de fonction Tensoriel 2D (matrice 2x2 à chaque nœud)
    CG1_tensor = fem.functionspace(mesh_dolfinx, ("CG", 1, (2, 2)))
    eps_obs = fem.Function(CG1_tensor, name="strain_tensor")

    # 5. Évaluation et interpolation aux nœuds
    local_expr = fem.Expression(epsilon_expr_ufl, CG1_tensor.element.interpolation_points)
    eps_obs.interpolate(local_expr)

    print("[Succès] Le tenseur des déformations epsilon (2D) a été calculé avec succès.")
    return eps_obs

def export_results_to_xdmf(mesh_dolfinx, u_obs, eps_obs, output_path: str):
    """Exporte le maillage dolfinx, le champ de déplacement et le tenseur epsilon

    dans un fichier XDMF pour ParaView. Compatible dolfinx v0.10.
    """
    # Création du fichier d'exportation avec dolfinx.io
    with dolfinx.io.XDMFFile(mesh_dolfinx.comm, output_path, "w") as xdmf:
        # 1. Écriture de la topologie et de la géométrie du maillage
        # (ParaView y trouvera automatiquement les coordonnées spatiales x, y, z des nœuds)
        xdmf.write_mesh(mesh_dolfinx)
        
        # 2. Écriture du champ de déplacement vectoriel u (contient u et v)
        # On spécifie le temps t=0.0 pour initialiser la série temporelle dans ParaView
        xdmf.write_function(u_obs, 0.0)
        
        # 3. Écriture du tenseur des déformations epsilon
        xdmf.write_function(eps_obs, 0.0)
        
    print(f"[Succès] Les résultats ont été exportés avec succès dans : {output_path}")


# global_coords_map servira à stocker les positions (x,y) strictes retenues au pas 1
REFERENCE_COORDS = None 

# =========================================================================
# 1. GÉNÉRATION DU MAILLAGE DE RÉFÉRENCE (FIXÉ AU PAS 1)
# =========================================================================
def create_reference_mesh_from_csv(csv_file: str, alpha: float = 20.0):
    """Lit le fichier 0001, fixe les coordonnées de référence globales,

    génère la triangulation et renvoie le maillage dolfinx.
    """
    global REFERENCE_COORDS
    
    data = pd.read_csv(csv_file)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    d = {names[i]: data.values[:, i] for i in range(len(names))}

    # Filtrage des points valides au pas 1 (déplacement non nul et non NaN)
    valid_mask = (d["sigma"] != -1)
    
    ref_x = np.round(d["x"][valid_mask], 4)
    ref_y = np.round(d["y"][valid_mask], 4)
    REFERENCE_COORDS = list(zip(ref_x, ref_y))
    
    points = np.stack([d["x"][valid_mask], d["y"][valid_mask], np.zeros_like(d["x"][valid_mask])], axis=-1)
    
    # Triangulation PyVista
    cloud = pv.PolyData(points)
    volume = cloud.delaunay_2d(alpha=alpha)
    if volume.n_cells == 0:
        print(f"[Attention] Aucun triangle avec alpha={alpha}, bascule sur Delaunay standard.")
        volume = cloud.delaunay_2d(alpha=0.0)

    triangles = volume.faces.reshape(-1, 4)[:, 1:]
    print(f"-> Maillage de référence créé : {len(points)} nœuds et {len(triangles)} triangles.")

    # Passage temporaire par meshio pour instancier proprement le maillage dolfinx
    tmp_path = "tmp_ref_mesh.xdmf"
    mesh_meshio = meshio.Mesh(points=points, cells=[("triangle", triangles)])
    meshio.write(tmp_path, mesh_meshio)

    with dolfinx.io.XDMFFile(MPI.COMM_SELF, tmp_path, "r") as xdmf_in:
        mesh_dolfinx = xdmf_in.read_mesh(name="Grid")
    
    mesh_dolfinx.topology.create_connectivity(mesh_dolfinx.topology.dim - 1, mesh_dolfinx.topology.dim)
    
    if os.path.exists(tmp_path): os.remove(tmp_path)
    if os.path.exists("tmp_ref_mesh.h5"): os.remove("tmp_ref_mesh.h5")
        
    return mesh_dolfinx


# =========================================================================
# 2. MISE À JOUR ALIGNÉE SUR LE PAS DE RÉFÉRENCE
# =========================================================================
def update_displacement_field(csv_file: str, mesh_dolfinx, u_obs):
    """Lit un CSV, extrait le déplacement uniquement pour les nœuds

    qui correspondent aux positions du maillage de référence.
    """
    global REFERENCE_COORDS
    
    data = pd.read_csv(csv_file)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    d = {names[i]: data.values[:, i] for i in range(len(names))}

    current_x = np.round(d["x"], 4)
    current_y = np.round(d["y"], 4)
    
    current_data_map = {
        (cx, cy): (cu, cv) for cx, cy, cu, cv in zip(current_x, current_y, d["u"], d["v"])
    }

    u_values = []
    v_values = []
    for coord in REFERENCE_COORDS:
        if coord in current_data_map:
            u_val, v_val = current_data_map[coord]
            u_values.append(u_val if not np.isnan(u_val) else 0.0)
            v_values.append(v_val if not np.isnan(v_val) else 0.0)
        else:
            u_values.append(0.0)
            v_values.append(0.0)

    u_values = np.array(u_values)
    v_values = np.array(v_values)

    # Permutation dolfinx standard
    global_indices = mesh_dolfinx.geometry.input_global_indices
    if global_indices is not None and global_indices.size > 0:
        permuted_ux = u_values[global_indices]
        permuted_uy = v_values[global_indices]
    else:
        permuted_ux = u_values
        permuted_uy = v_values

    # Injection
    u_array = u_obs.x.array
    u_array[0::2] = permuted_ux
    u_array[1::2] = permuted_uy
    u_obs.x.scatter_forward()


def update_displacement_field_2(csv_file: str, mesh_dolfinx, u_obs, k: int = 3):
    """Lit un CSV, extrait le déplacement et remplace les valeurs manquantes

    ou NaN par une interpolation IDW (Inverse Distance Weighting) via un KDTree.
    """
    global REFERENCE_COORDS
    
    data = pd.read_csv(csv_file)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    d = {names[i]: data.values[:, i] for i in range(len(names))}

    # 1. Filtrer uniquement les points VALIDES du pas actuel (non NaN et non nuls)
    u_curr = d["u"]
    v_curr = d["v"]
    valid_mask = ~np.isnan(u_curr) & ~np.isnan(v_curr) & (u_curr != 0)
    
    # Sécurité : si aucun point n'est valide dans tout le CSV
    if not np.any(valid_mask):
        print(f"[Attention] Aucun point valide dans {csv_file}. Remplissage par zéros.")
        u_values = np.zeros(len(REFERENCE_COORDS))
        v_values = np.zeros(len(REFERENCE_COORDS))
    else:
        valid_coords = np.stack([d["x"][valid_mask], d["y"][valid_mask]], axis=-1)
        valid_u = u_curr[valid_mask]
        valid_v = v_curr[valid_mask]

        # 2. Construire le KDTree avec les coordonnées valides actuelles
        tree = KDTree(valid_coords)

        # 3. Convertir les coordonnées de référence en tableau NumPy pour traitement vectoriel
        ref_coords_arr = np.array(REFERENCE_COORDS)
        
        # Ajuster k si jamais on a moins de points valides que le k demandé
        k_neighbors = min(k, len(valid_coords))
        
        # Chercher les k voisins les plus proches pour chaque nœud de référence
        distances, indices = tree.query(ref_coords_arr, k=k_neighbors)

        # 4. Interpolation Inverse Distance Weighting (IDW)
        if k_neighbors == 1:
            # Cas dégradé : un seul voisin
            u_values = valid_u[indices]
            v_values = valid_v[indices]
        else:
            # Éviter la division par zéro pour les points qui se superposent exactement
            eps = 1e-10
            weights = 1.0 / (distances + eps)
            
            # Normalisation des poids (la somme des poids pour un point doit valoir 1)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights /= weights_sum

            # Calcul de la moyenne pondérée (produit scalaire matriciel pour la vitesse)
            u_values = np.sum(valid_u[indices] * weights, axis=1)
            v_values = np.sum(valid_v[indices] * weights, axis=1)

    # 5. Permutation dolfinx standard
    global_indices = mesh_dolfinx.geometry.input_global_indices
    if global_indices is not None and global_indices.size > 0:
        permuted_ux = u_values[global_indices]
        permuted_uy = v_values[global_indices]
    else:
        permuted_ux = u_values
        permuted_uy = v_values

    # 6. Injection dans la fonction FEniCSx
    u_array = u_obs.x.array
    u_array[0::2] = permuted_ux
    u_array[1::2] = permuted_uy
    u_obs.x.scatter_forward()



def update_strain_field_from_csv(csv_file: str, mesh_dolfinx, E_obs, k: int = 3):
    """Lit un CSV, extrait les composantes de déformation (exx, eyy, exy) et remplace

    les valeurs manquantes ou NaN par une interpolation IDW via un KDTree.
    """
    global REFERENCE_COORDS
    
    data = pd.read_csv(csv_file)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    d = {names[i]: data.values[:, i] for i in range(len(names))}

    # 1. Filtrer uniquement les points VALIDES du pas actuel (non NaN pour les déformations)
    exx_curr = d["exx"]
    eyy_curr = d["eyy"]
    exy_curr = d["exy"]
    
    # On valide les points où aucune des trois composantes n'est NaN
    valid_mask = ~np.isnan(exx_curr) & ~np.isnan(eyy_curr) & ~np.isnan(exy_curr)
    
    # Sécurité : si aucun point n'est valide dans tout le CSV
    if not np.any(valid_mask):
        print(f"[Attention] Aucun point valide pour les déformations dans {csv_file}. Remplissage par zéros.")
        exx_values = np.zeros(len(REFERENCE_COORDS))
        eyy_values = np.zeros(len(REFERENCE_COORDS))
        exy_values = np.zeros(len(REFERENCE_COORDS))
    else:
        valid_coords = np.stack([d["x"][valid_mask], d["y"][valid_mask]], axis=-1)
        valid_exx = exx_curr[valid_mask]
        valid_eyy = eyy_curr[valid_mask]
        valid_exy = exy_curr[valid_mask]

        # 2. Construire le KDTree avec les coordonnées valides actuelles
        tree = KDTree(valid_coords)

        # 3. Convertir les coordonnées de référence en tableau NumPy
        ref_coords_arr = np.array(REFERENCE_COORDS)
        
        # Ajuster k si besoin
        k_neighbors = min(k, len(valid_coords))
        
        # Chercher les k voisins les plus proches
        distances, indices = tree.query(ref_coords_arr, k=k_neighbors)

        # 4. Interpolation Inverse Distance Weighting (IDW)
        if k_neighbors == 1:
            u_ind = indices.ravel() if indices.ndim > 1 else indices
            exx_values = valid_exx[u_ind]
            eyy_values = valid_eyy[u_ind]
            exy_values = valid_exy[u_ind]
        else:
            eps = 1e-10
            weights = 1.0 / (distances + eps)
            
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights /= weights_sum

            # Calcul de la moyenne pondérée pour chaque composante tensorielle
            exx_values = np.sum(valid_exx[indices] * weights, axis=1)
            eyy_values = np.sum(valid_eyy[indices] * weights, axis=1)
            exy_values = np.sum(valid_exy[indices] * weights, axis=1)

    # 5. Permutation dolfinx standard basée sur la géométrie du maillage
    global_indices = mesh_dolfinx.geometry.input_global_indices
    if global_indices is not None and global_indices.size > 0:
        permuted_exx = exx_values[global_indices]
        permuted_eyy = eyy_values[global_indices]
        permuted_exy = exy_values[global_indices]
    else:
        permuted_exx = exx_values
        permuted_eyy = eyy_values
        permuted_exy = exy_values

    # 6. Injection dans la fonction FEniCSx (Espace Tensoriel 2D -> 4 composantes par nœud)
    # L'ordre dolfinx standard pour un tenseur (2, 2) au nœud est : [T00, T01, T10, T11]
    # Soit : [exx, exy, eyx, eyy]
    E_array = E_obs.x.array
    E_array[0::4] = permuted_exx  # Composante (0,0) -> exx
    E_array[1::4] = permuted_exy  # Composante (0,1) -> exy
    E_array[2::4] = permuted_exy  # Composante (1,0) -> eyx (égal à exy par symétrie)
    E_array[3::4] = permuted_eyy  # Composante (1,1) -> eyy
    
    E_obs.x.scatter_forward()
# =========================================================================
# 3. BOUCLE PRINCIPALE AVEC SÉRIE TEMPORELLE FENICSX
# =========================================================================
def process_csv_series_fenicsx(folder_path: str, output_xdmf: str, file_prefix: str, alpha: float = 0.2):
    """Boucle sur les fichiers CSV en utilisant le moteur dolfinx pour l'écriture

    de la série temporelle et UFL pour le calcul exact d'epsilon.
    """
    # Utilisation du pas 0001 comme référence géométrique
    first_csv = os.path.join(folder_path, f"{file_prefix}0001.csv")
    if not os.path.exists(first_csv):
        raise FileNotFoundError(f"Le fichier de référence initial {first_csv} est introuvable.")
        
    print(f"[Info] Création du maillage de référence depuis {first_csv}...")
    mesh = create_reference_mesh_from_csv(first_csv, alpha=alpha)

    # Préparation des espaces fonctionnels dolfinx v0.10
    CG1_vector = fem.functionspace(mesh, ("CG", 1, (2,)))
    u_field = fem.Function(CG1_vector, name="displacement")

    CG1_tensor = fem.functionspace(mesh, ("CG", 1, (2, 2)))
    eps_field = fem.Function(CG1_tensor, name="strain_tensor")

    # Opérateur UFL pour Epsilon (2D + Symétrie)
    grad_full = ufl.grad(u_field)
    grad_2d = ufl.as_matrix([[grad_full[0, 0], grad_full[0, 1]], [grad_full[1, 0], grad_full[1, 1]]])
    epsilon_expr_ufl = ufl.sym(grad_2d)
    local_expr = fem.Expression(epsilon_expr_ufl, CG1_tensor.element.interpolation_points)

    print(f"[Info] Début du traitement de la série dans {output_xdmf}...")
    with dolfinx.io.XDMFFile(mesh.comm, output_xdmf, "w") as xdmf:
        # Écriture initiale obligatoire du maillage complet (non vide !)
        xdmf.write_mesh(mesh)

        # Traitement chronologique de la série (de 1 à 100)
        for step in range(0, 1237, 20):
            csv_name = f"{file_prefix}{step:04d}.csv"
            csv_path = os.path.join(folder_path, csv_name)

            if not os.path.exists(csv_path):
                print(f"[Attention] Pas de temps {step} manquant ({csv_name}). Arrêt de la boucle.")
                break

            t = float(step)

            # Mise à jour des valeurs et calcul d'epsilon par projection UFL
            update_displacement_field_2(csv_path, mesh, u_field)
            eps_field.interpolate(local_expr)

            # Écriture dans la structure temporelle XDMF
            xdmf.write_function(u_field, t)
            xdmf.write_function(eps_field, t)

            if step % 10 == 0:
                print(f" -> Étape {step}/100 traitée ({csv_name} au temps t={t})")

    print(f"[Succès] Série temporelle complète générée avec dolfinx dans : {output_xdmf}")

import os
import dolfinx
import ufl
from dolfinx import fem


def process_csv_series_fenicsx(folder_path: str, output_xdmf: str, file_prefix: str, alpha: float = 0.2):
    """Boucle sur les fichiers CSV en utilisant dolfinx pour l'écriture de la série temporelle.

    Calcule et compare les différents tenseurs de déformation (Infinitésimal, Green-Lagrange, 
    Hencky et CSV).
    """
    # Utilisation du pas 0001 comme référence géométrique
    first_csv = os.path.join(folder_path, f"{file_prefix}0001.csv")
    if not os.path.exists(first_csv):
        raise FileNotFoundError(f"Le fichier de référence initial {first_csv} est introuvable.")
        
    print(f"[Info] Création du maillage de référence depuis {first_csv}...")
    mesh = create_reference_mesh_from_csv(first_csv, alpha=alpha)

    # Préparation des espaces fonctionnels dolfinx v0.10
    CG1_scalar = fem.functionspace(mesh, ("CG", 1))
    CG1_vector = fem.functionspace(mesh, ("CG", 1, (2,)))
    CG1_tensor = fem.functionspace(mesh, ("CG", 1, (2, 2)))

    # Déclaration des champs (Functions)
    u_field = fem.Function(CG1_vector, name="displacement")
    
    # 1. Tenseur de déformation linéarisé (Cauchy / Infinitésimal)
    eps_field = fem.Function(CG1_tensor, name="epsilon_ufl")
    
    # 2. Tenseur de Green-Lagrange (Grandes déformations UFL)
    E_ufl_field = fem.Function(CG1_tensor, name="E_ufl")
    
    # 3. Tenseur de déformation extrait du CSV (exx, eyy, exy)
    E_csv_field = fem.Function(CG1_tensor, name="E_csv")
    
    # 4. Tenseur de Hencky (Logarithmique, basé sur ta simplification C = 2E + I)
    H_field = fem.Function(CG1_tensor, name="Hencky_ufl")
    
    # 5. Tenseur d'écart / différence entre Hencky (UFL) et les données du CSV
    diff_H_csv_field = fem.Function(CG1_scalar, name="diff_Hencky_minus_Ecsv")

        # --- Opérateurs et Expressions UFL ---
        # grad_full a une forme (2, 3) : lignes = composantes de u, colonnes = dérivées (dx, dy, dz)
    grad_full = ufl.grad(u_field)

    # On extrait uniquement les dérivées par rapport à x (0) et y (1) pour avoir un tenseur (2, 2)
    grad_u = ufl.as_matrix([[grad_full[0, 0], grad_full[0, 1]], [grad_full[1, 0], grad_full[1, 1]]])
    I = ufl.Identity(2)
    
    # Équation d'Epsilon (linéaire)
    epsilon_expr_ufl = ufl.sym(grad_u)
    
    # Équation de Green-Lagrange (E)
    E_expr_ufl = 0.5 * (grad_u + ufl.transpose(grad_u) + ufl.dot(ufl.transpose(grad_u), grad_u))
    
    # Équation de Hencky via ta simplification : C = 2E + I  =>  C - I = 2E
    # Ce qui donne l'approximation : ln(C) ~= (2E) - 0.5 * (2E * 2E) = 2E - 2 * E^2
    C = 2.0 * E_expr_ufl + I
    X = C - I

    # # Termes de l'approximation de Padé pour ln(C)
    # # ln(I + X) ~= X - 1/2 X^2 + 1/3 X^3 ... ou via fraction rationnelle de Padé
    # # Pour plus de précision sans inversion de matrice en UFL, un Taylor étendu ou Padé explicite :
    # X2 = ufl.dot(X, X)
    # X3 = ufl.dot(X2, X)
    # X4 = ufl.dot(X3, X)

    # # Approximation de Taylor-Padé de ln(C)
    # lnC = X - 0.5*X2 + (1.0/3.0)*X3 - 0.25*X4
    # #ln_C_approx = C_minus_I - 0.5 * ufl.dot(C_minus_I, C_minus_I)
    Hencky_expr_ufl = 0.5 * ufl.dot(X, ufl.inv(I - 0.5 * X))  # Forme de Padé plus stable que la série de Taylor
    
    # Équation de la différence : Hencky_ufl - E_csv
    # On soustrait directement le champ dolfinx "E_csv_field" de l'expression symbolique de Hencky
    
    # diff_expr_ufl = Hencky_expr_ufl - E_csv_field
    diff_expr_ufl = ufl.sqrt(ufl.inner(Hencky_expr_ufl - E_csv_field, Hencky_expr_ufl - E_csv_field))
    # Compilation des expressions locales pour l'interpolation dolfinx v0.10
    interp_points = CG1_tensor.element.interpolation_points
    eps_local_expr = fem.Expression(epsilon_expr_ufl, interp_points)
    E_ufl_local_expr = fem.Expression(E_expr_ufl, interp_points)
    Hencky_local_expr = fem.Expression(Hencky_expr_ufl, interp_points)
    diff_local_expr = fem.Expression(diff_expr_ufl, interp_points)

    print(f"[Info] Début du traitement de la série dans {output_xdmf}...")
    with dolfinx.io.XDMFFile(mesh.comm, output_xdmf, "w") as xdmf:
        # Écriture initiale du maillage
        xdmf.write_mesh(mesh)

        # Traitement chronologique de la série (de 0 à 1236, pas de 20)
        for step in range(0, 1237, 20):
            csv_name = f"{file_prefix}{step:04d}.csv"
            csv_path = os.path.join(folder_path, csv_name)

            if not os.path.exists(csv_path):
                print(f"[Attention] Pas de temps {step} manquant ({csv_name}). Arrêt de la boucle.")
                break

            t = float(step)

            # 1. Mise à jour du déplacement depuis le CSV
            update_displacement_field_2(csv_path, mesh, u_field)
            
            # 2. Remplissage du tenseur E_csv à partir des colonnes "exx", "eyy", "exy"
            # (Note : assure-toi que cette fonction applique bien exy aux indices [0,1] et [1,0])
            update_strain_field_from_csv(csv_path, mesh, E_csv_field)

            # 3. Calculs cinématiques UFL par interpolation
            eps_field.interpolate(eps_local_expr)
            E_ufl_field.interpolate(E_ufl_local_expr)
            H_field.interpolate(Hencky_local_expr)
            
            # 4. Calcul de la différence (s'exécute après la mise à jour de H_field et E_csv_field)
            diff_H_csv_field.interpolate(diff_local_expr)

            # 5. Écriture des données temporelles isolées distinctement dans le fichier XDMF
            xdmf.write_function(u_field, t)
            xdmf.write_function(eps_field, t)
            xdmf.write_function(E_ufl_field, t)
            xdmf.write_function(E_csv_field, t)
            xdmf.write_function(H_field, t)
            xdmf.write_function(diff_H_csv_field, t)

            if step % 10 == 0:
                print(f" -> Étape {step}/1237 traitée ({csv_name} au temps t={t})")

    print(f"[Succès] Série temporelle complète générée avec dolfinx dans : {output_xdmf}")


def interpolate_displacement_obs_mesh_to_cad_mesh_2D(
    u_obs: fem.Function, u_cad: fem.Function, tform_cad_to_img_4D: NDArray
) -> None:
    V_obs = u_obs.function_space
    disp_obs_dim = V_obs.element.value_shape[0]

    V_cad = u_cad.function_space
    disp_cad_dim = V_cad.element.value_shape[0]
    dim = disp_cad_dim
    assert disp_obs_dim == dim

    tform_jac_cad_to_img_3D = tform_cad_to_img_4D[:3, :3]

    bb_tree_obs = dolfinx.geometry.bb_tree(V_obs.mesh, 2)
    midpoint_tree = dolfinx.geometry.create_midpoint_tree(
        V_obs.mesh,
        2,
        np.arange(V_obs.mesh.topology.index_map(2).size_local, dtype=np.int32),
    )

    def _u_cad_expr(x: NDArray) -> NDArray:
        x_4D = np.concatenate([x, [np.ones_like(x[0])]])
        x_img_4D = tform_cad_to_img_4D @ x_4D
        x_img = x_img_4D[:3, :]

        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree_obs, x_img.T)
        cells = dolfinx.geometry.compute_colliding_cells(V_obs.mesh, cell_candidates, x_img.T)
        closest_entity = dolfinx.geometry.compute_closest_entity(bb_tree_obs, midpoint_tree, V_obs.mesh, x_img.T)

        selected_cells = closest_entity
        disp_eval = u_obs.eval(x_img.T, selected_cells)

        disp_eval_scaled = np.linalg.inv(tform_jac_cad_to_img_3D) @ np.concatenate(
            [disp_eval.T, [np.zeros_like(disp_eval.T[0])]]
        )

        # CORRECTION ICI : on coupe à 'dim' (2 ou 3) pour respecter l'espace cible
        return disp_eval_scaled[:dim]

    u_cad.interpolate(_u_cad_expr)

# if __name__ == "__main__":
#     dossier_csv = "/home/pmarechal/Documents/DP_N_E/images_and_csv"
    
#     process_csv_series_fenicsx(
#         folder_path=dossier_csv, 
#         output_xdmf="dic_series_complete.xdmf", 
#         file_prefix="N_E_basler_", 
#         alpha=20.0
#     )

import xml.etree.ElementTree as ET

def project_h5_series_to_cad_mesh(
    h5_path: str, 
    mesh_cad: dolfinx.mesh.Mesh, 
    tform_h5_to_cad_4D: NDArray, 
    output_xdmf_path: str
) -> None:
    """Lit une série temporelle FEniCSx directement depuis son fichier H5 (via h5py),
    reconstruit le maillage source et projette le champ de déplacement vectoriel
    sur un maillage cible dolfinx.mesh.Mesh pour chaque pas de temps.
    """
    # =========================================================================
    # 1. PARSING DE LA STRUCTURE ET DES PAS DE TEMPS
    # =========================================================================
    xdmf_path = h5_path.replace(".h5", ".xdmf")
    steps = []
    geom_path = "/Mesh/Grid/geometry"
    topo_path = "/Mesh/Grid/topology"

    if os.path.exists(xdmf_path):
        print(f"[1/4] Analyse du fichier métadonnées {xdmf_path} pour identifier les datasets...")
        tree = ET.parse(xdmf_path)
        root = tree.getroot()
        
        try:
            geom_path = next(root.iter("Geometry")).find("DataItem").text.split(":")[-1].strip()
            topo_path = next(root.iter("Topology")).find("DataItem").text.split(":")[-1].strip()
        except Exception:
            pass 

        for grid in root.findall(".//Grid"):
            time_node = grid.find("Time")
            attr_node = grid.find(".//Attribute[@Name='displacement']")
            if time_node is not None and attr_node is not None:
                t_val = float(time_node.get("Value"))
                di = attr_node.find("DataItem")
                if di is not None:
                    ds_path = di.text.split(":")[-1].strip()
                    steps.append((t_val, ds_path))
    else:
        print(f"[1/4] Fichier XDMF non trouvé. Inspection directe du fichier H5 : {h5_path}")
        with h5py.File(h5_path, "r") as f:
            mesh_key = list(f["Mesh"].keys())[0]
            geom_path = f"/Mesh/{mesh_key}/geometry"
            topo_path = f"/Mesh/{mesh_key}/topology"
            func_group = f["Function/displacement"]
            for k in sorted(list(func_group.keys()), key=lambda x: int(x) if x.isdigit() else x):
                try:
                    t_val = float(k)
                except ValueError:
                    t_val = float(len(steps))
                steps.append((t_val, f"Function/displacement/{k}"))

    if not steps:
        raise ValueError("Aucun pas de temps ou champ de déplacement n'a pu être localisé.")
    
    steps.sort(key=lambda x: x[0])
    print(f" -> {len(steps)} pas de temps détectés.")

    # =========================================================================
    # 2. CHARGEMENT DU MAILLAGE SOURCE DEPUIS LE H5 ET INSTANCIATION DOLFINX
    # =========================================================================
    print(f"[2/4] Extraction des tableaux de maillage depuis le H5...")
    with h5py.File(h5_path, "r") as f:
        points_obs = f[geom_path][:]
        cells_obs = f[topo_path][:]

    # FIX 1 : shape=(3,) définit proprement la dimension géométrique (gdim=3) pour UFL
    coord_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    domain_obs = ufl.Mesh(coord_element)
    
    # FIX 2 : Mots-clés nommés explicites pour contourner l'inversion d'arguments v0.9/v0.10
    mesh_obs = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF, 
        cells=cells_obs, 
        x=points_obs, 
        e=domain_obs
    )
    mesh_obs.topology.create_connectivity(mesh_obs.topology.dim - 1, mesh_obs.topology.dim)

    # Initialisation des espaces de fonction (Déplacement 2D)
    dim_disp = 2
    V_obs = fem.functionspace(mesh_obs, ("CG", 1, (dim_disp,)))
    u_obs = fem.Function(V_obs, name="displacement")
    global_indices_obs = mesh_obs.geometry.input_global_indices

    # Préparation de l'espace sur le maillage cible (CAD) reçu en argument
    V_cad = fem.functionspace(mesh_cad, ("CG", 1, (dim_disp,)))
    u_cad = fem.Function(V_cad, name="displacement_projected")

    # 3. Définition du seuil de distance (ex: si la DIC a un pas de 2mm, un seuil à 2.0 ou 3.0 mm est cohérent)
    DISTANCE_THRESHOLD = 2.0  # À ajuster selon la taille de tes mailles DIC

    # =========================================================================
    # 3. TRANSFORMATION DE COORDONNÉES (INVERSION)
    # =========================================================================
    print("[3/4] Inversion de la matrice de transformation spatiale...")
    tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

    # =========================================================================
    # 4. BOUCLE TEMPORELLE DE LECTURE H5 ET PROJECTION
    # =========================================================================
    print(f"[4/4] Lancement de la projection temporelle brute vers : {output_xdmf_path}")
    
    with dolfinx.io.XDMFFile(mesh_cad.comm, output_xdmf_path, "w") as xdmf_out:
        xdmf_out.write_mesh(mesh_cad)

        with h5py.File(h5_path, "r") as f:
            for t, ds_path in steps:
                data_step = f[ds_path][:]
                
                ux = np.nan_to_num(data_step[:, 0], nan=0.0)
                uy = np.nan_to_num(data_step[:, 1], nan=0.0)

                if global_indices_obs is not None and global_indices_obs.size > 0:
                    permuted_ux = ux[global_indices_obs]
                    permuted_uy = uy[global_indices_obs]
                else:
                    permuted_ux = ux
                    permuted_uy = uy

                u_obs_array = u_obs.x.array
                u_obs_array[0::2] = permuted_ux
                u_obs_array[1::2] = permuted_uy
                u_obs.x.scatter_forward()

                # Appel à ta fonction externe d'interpolation géométrique
                interpolate_displacement_obs_mesh_to_cad_mesh_2D(u_obs, u_cad, tform_cad_to_img_4D)

                xdmf_out.write_function(u_cad, t)
                print(f" -> Pas t={t} projeté avec succès.")

    print(f"[Succès] Série temporelle complète exportée.")

def project_h5_series_to_cad_mesh_mask(
    h5_path: str, 
    mesh_cad: dolfinx.mesh.Mesh, 
    tform_h5_to_cad_4D: NDArray, 
    output_xdmf_path: str
) -> None:
    """Lit une série temporelle FEniCSx directement depuis son fichier H5 (via h5py),
    reconstruit le maillage source et projette le champ de déplacement vectoriel
    sur un maillage cible dolfinx.mesh.Mesh pour chaque pas de temps.
    
    Ajoute également une étiquette scalaire 'is_imported' (0 ou 1) sur le maillage CAD.
    """
    # =========================================================================
    # 1. PARSING DE LA STRUCTURE ET DES PAS DE TEMPS
    # =========================================================================
    xdmf_path = h5_path.replace(".h5", ".xdmf")
    steps = []
    geom_path = "/Mesh/Grid/geometry"
    topo_path = "/Mesh/Grid/topology"

    if os.path.exists(xdmf_path):
        print(f"[1/4] Analyse du fichier métadonnées {xdmf_path} pour identifier les datasets...")
        tree = ET.parse(xdmf_path)
        root = tree.getroot()
        
        try:
            geom_path = next(root.iter("Geometry")).find("DataItem").text.split(":")[-1].strip()
            topo_path = next(root.iter("Topology")).find("DataItem").text.split(":")[-1].strip()
        except Exception:
            pass 

        for grid in root.findall(".//Grid"):
            time_node = grid.find("Time")
            attr_node = grid.find(".//Attribute[@Name='displacement']")
            if time_node is not None and attr_node is not None:
                t_val = float(time_node.get("Value"))
                di = attr_node.find("DataItem")
                if di is not None:
                    ds_path = di.text.split(":")[-1].strip()
                    steps.append((t_val, ds_path))
    else:
        print(f"[1/4] Fichier XDMF non trouvé. Inspection directe du fichier H5 : {h5_path}")
        with h5py.File(h5_path, "r") as f:
            mesh_key = list(f["Mesh"].keys())[0]
            geom_path = f"/Mesh/{mesh_key}/geometry"
            topo_path = f"/Mesh/{mesh_key}/topology"
            func_group = f["Function/displacement"]
            for k in sorted(list(func_group.keys()), key=lambda x: int(x) if x.isdigit() else x):
                try:
                    t_val = float(k)
                except ValueError:
                    t_val = float(len(steps))
                steps.append((t_val, f"Function/displacement/{k}"))

    if not steps:
        raise ValueError("Aucun pas de temps ou champ de déplacement n'a pu être localisé.")
    
    steps.sort(key=lambda x: x[0])
    print(f" -> {len(steps)} pas de temps détectés.")

    # =========================================================================
    # 2. CHARGEMENT DU MAILLAGE SOURCE DEPUIS LE H5 ET INSTANCIATION DOLFINX
    # =========================================================================
    print(f"[2/4] Extraction des tableaux de maillage depuis le H5...")
    with h5py.File(h5_path, "r") as f:
        points_obs = f[geom_path][:]
        cells_obs = f[topo_path][:]

    coord_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    domain_obs = ufl.Mesh(coord_element)
    
    mesh_obs = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF, 
        cells=cells_obs, 
        x=points_obs, 
        e=domain_obs
    )
    mesh_obs.topology.create_connectivity(mesh_obs.topology.dim - 1, mesh_obs.topology.dim)

    # Initialisation des espaces de fonction (Déplacement 2D)
    dim_disp = 2
    V_obs = fem.functionspace(mesh_obs, ("CG", 1, (dim_disp,)))
    u_obs = fem.Function(V_obs, name="displacement")
    global_indices_obs = mesh_obs.geometry.input_global_indices

    # Préparation de l'espace sur le maillage cible (CAD) reçu en argument
    V_cad = fem.functionspace(mesh_cad, ("CG", 1, (dim_disp,)))
    u_cad = fem.Function(V_cad, name="displacement_projected")
    
    # Seuil de distance pour valider si un point reçoit une donnée H5
    DISTANCE_THRESHOLD = 2.0  

    # =========================================================================
    # NOUVEAU : CALCUL GÉOMÉTRIQUE DE L'ÉTIQUETTE (MASQUE STATIQUE)
    # =========================================================================
    print("[Mise à jour] Calcul de l'étiquette géométrique 'is_imported' (Cylindre Z)...")
    if points_obs.shape[1] == 2:
        points_3d = np.hstack([points_obs, np.zeros((points_obs.shape[0], 1))])
    else:
        points_3d = points_obs[:, :3]
        
    points_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_obs_in_cad_space = (tform_h5_to_cad_4D @ points_hom.T).T[:, :3]

    # CHANGEMENT 1 : On ne garde que X et Y pour projeter sur le plan (O, X, Y)
    points_obs_2d = points_obs_in_cad_space[:, :2]
    tree_obs_2d = KDTree(points_obs_2d)

    V_mask = fem.functionspace(mesh_cad, ("CG", 1))
    u_mask = fem.Function(V_mask, name="is_imported")

    # Seuil de distance (le rayon du cylindre)
    DISTANCE_THRESHOLD = 2.0  

    def mask_interpolation_callback(x):
        # x a une forme (3, num_points_cad) -> [x_cad, y_cad, z_cad]
        # CHANGEMENT 2 : On ignore x[2, :] (le Z du CAD) pour la requête
        nodes_cad_2d = x[:2, :].T
        
        # La distance calculée ici est purement en 2D (dans le plan XY)
        distances, _ = tree_obs_2d.query(nodes_cad_2d, distance_upper_bound=DISTANCE_THRESHOLD)
        
        return np.where(distances <= DISTANCE_THRESHOLD, 1.0, 0.0)

    u_mask.interpolate(mask_interpolation_callback)
    u_mask.x.scatter_forward()
    mask_values = u_mask.x.array
    # 1. On évalue le masque au centre de chaque cellule (Approximation DG0)
    tdim = mesh_cad.topology.dim
    fdim = tdim - 1

    # Trouver la connectivité Cellules -> Facettes
    mesh_cad.topology.create_connectivity(tdim, fdim)
    mesh_cad.topology.create_connectivity(fdim, tdim)

    cell_to_facets = mesh_cad.topology.connectivity(tdim, fdim)
    facet_to_cells = mesh_cad.topology.connectivity(fdim, tdim)

    # Calculer une valeur par cellule (moyenne des sommets de la cellule)
    cell_to_vertices = mesh_cad.topology.connectivity(tdim, 0)
    num_cells = mesh_cad.topology.index_map(tdim).size_local + mesh_cad.topology.index_map(tdim).num_ghosts

    cell_values = np.zeros(num_cells)
    for cell in range(num_cells):
        vertices = cell_to_vertices.links(cell)
        cell_values[cell] = 1.0 if np.mean(mask_values[vertices]) > 0.5 else 0.0

    # 2. Une facette est sur le contour si elle sépare une cellule à 1 d'une cellule à 0
    boundary_facets = []
    num_facets = mesh_cad.topology.index_map(fdim).size_local + mesh_cad.topology.index_map(fdim).num_ghosts

    for facet in range(num_facets):
        cells = facet_to_cells.links(facet)
        
        if len(cells) == 2:
            # Facette interne : sépare deux cellules
            val1 = cell_values[cells[0]]
            val2 = cell_values[cells[1]]
            if val1 != val2: # L'une est à 1, l'autre à 0
                boundary_facets.append(facet)
        elif len(cells) == 1:
            # Facette sur la frontière physique du maillage CAD
            val = cell_values[cells[0]]
            if val == 1.0: 
                # Si la zone importée touche le bord du CAD, c'est aussi un contour
                boundary_facets.append(facet)

    # 3. Création du MeshTag
    boundary_facets = np.array(boundary_facets, dtype=np.int32)
    values = np.full_like(boundary_facets, 1, dtype=np.int32) # Tag "1" pour le contour

    contour_meshtags = dolfinx.mesh.meshtags(mesh_cad, fdim, boundary_facets, values)
    contour_meshtags.name = "contour_is_imported"
    # =========================================================================
    # 3. TRANSFORMATION DE COORDONNÉES (INVERSION)
    # =========================================================================
    print("[3/4] Inversion de la matrice de transformation spatiale...")
    tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

    # =========================================================================
    # 4. BOUCLE TEMPORELLE DE LECTURE H5 ET PROJECTION
    # =========================================================================
    print(f"[4/4] Lancement de la projection temporelle brute vers : {output_xdmf_path}")
    
    with dolfinx.io.XDMFFile(mesh_cad.comm, output_xdmf_path, "w") as xdmf_out:
        xdmf_out.write_mesh(mesh_cad)

        R_h5_to_cad = tform_h5_to_cad_4D[:3, :3]

        with h5py.File(h5_path, "r") as f:
            for t, ds_path in steps:
                data_step = f[ds_path][:]
                num_nodes = data_step.shape[0]
                
                # 1. On crée un tableau 3D pour les déplacements d'origine [Ux, Uy, 0]
                disp_orig_3d = np.zeros((num_nodes, 3))
                disp_orig_3d[:, :2] = data_step[:, :2]
                
                # 2. Transformation des vecteurs par la matrice 3x3 (Rotation + Échelle)
                # (R_h5_to_cad @ disp_orig_3d.T).T permet d'opérer efficacement sur tous les nœuds d'un coup
                disp_transformed_3d = (R_h5_to_cad @ disp_orig_3d.T).T
                
                # 3. Extraction des nouvelles composantes et gestion des NaNs
                ux = np.nan_to_num(disp_transformed_3d[:, 0], nan=0.0)
                uy = np.nan_to_num(disp_transformed_3d[:, 1], nan=0.0)

                # [Le reste de ton code de permutation et d'affectation reste inchangé]
                if global_indices_obs is not None and global_indices_obs.size > 0:
                    permuted_ux = ux[global_indices_obs]
                    permuted_uy = uy[global_indices_obs]
                else:
                    permuted_ux = ux
                    permuted_uy = uy

                u_obs_array = u_obs.x.array
                u_obs_array[0::2] = permuted_ux
                u_obs_array[1::2] = permuted_uy
                u_obs.x.scatter_forward()

                # Interpolation vers le maillage CAD
                interpolate_displacement_obs_mesh_to_cad_mesh_2D(u_obs, u_cad, tform_cad_to_img_4D)

                xdmf_out.write_function(u_cad, t)
                xdmf_out.write_function(u_mask, t)
                mesh_cad.topology.create_connectivity(fdim, tdim)
                xdmf_out.write_meshtags(contour_meshtags, mesh_cad.geometry)
                print(f" -> Pas t={t} projeté avec succès.")

    print(f"[Succès] Série temporelle complète exportée.")


# def project_h5_series_to_cad_mesh_mask_2(
#     h5_path: str, 
#     mesh_cad: dolfinx.mesh.Mesh, 
#     tform_h5_to_cad_4D: NDArray, 
#     output_xdmf_path: str
# ) -> None:
#     """Lit une série temporelle FEniCSx directement depuis son fichier H5 (via h5py),
#     reconstruit le maillage source et projette le champ de déplacement vectoriel
#     sur un maillage cible dolfinx.mesh.Mesh pour chaque pas de temps.
    
#     Calcule et exporte également le contour de la zone interpolée via des MeshTags.
#     """
#     # =========================================================================
#     # 1. PARSING DE LA STRUCTURE ET DES PAS DE TEMPS
#     # =========================================================================
#     xdmf_path = h5_path.replace(".h5", ".xdmf")
#     steps = []
#     geom_path = "/Mesh/Grid/geometry"
#     topo_path = "/Mesh/Grid/topology"

#     if os.path.exists(xdmf_path):
#         print(f"[1/4] Analyse du fichier métadonnées {xdmf_path} pour identifier les datasets...")
#         tree = ET.parse(xdmf_path)
#         root = tree.getroot()
        
#         try:
#             geom_path = next(root.iter("Geometry")).find("DataItem").text.split(":")[-1].strip()
#             topo_path = next(root.iter("Topology")).find("DataItem").text.split(":")[-1].strip()
#         except Exception:
#             pass 

#         for grid in root.findall(".//Grid"):
#             time_node = grid.find("Time")
#             attr_node = grid.find(".//Attribute[@Name='displacement']")
#             if time_node is not None and attr_node is not None:
#                 t_val = float(time_node.get("Value"))
#                 di = attr_node.find("DataItem")
#                 if di is not None:
#                     ds_path = di.text.split(":")[-1].strip()
#                     steps.append((t_val, ds_path))
#     else:
#         print(f"[1/4] Fichier XDMF non trouvé. Inspection directe du fichier H5 : {h5_path}")
#         with h5py.File(h5_path, "r") as f:
#             mesh_key = list(f["Mesh"].keys())[0]
#             geom_path = f"/Mesh/{mesh_key}/geometry"
#             topo_path = f"/Mesh/{mesh_key}/topology"
#             func_group = f["Function/displacement"]
#             for k in sorted(list(func_group.keys()), key=lambda x: int(x) if x.isdigit() else x):
#                 try:
#                     t_val = float(k)
#                 except ValueError:
#                     t_val = float(len(steps))
#                 steps.append((t_val, f"Function/displacement/{k}"))

#     if not steps:
#         raise ValueError("Aucun pas de temps ou champ de déplacement n'a pu être localisé.")
    
#     steps.sort(key=lambda x: x[0])
#     print(f" -> {len(steps)} pas de temps détectés.")

#     # =========================================================================
#     # 2. CHARGEMENT DU MAILLAGE SOURCE DEPUIS LE H5 ET INSTANCIATION DOLFINX
#     # =========================================================================
#     print(f"[2/4] Extraction des tableaux de maillage depuis le H5...")
#     with h5py.File(h5_path, "r") as f:
#         points_obs = f[geom_path][:]
#         cells_obs = f[topo_path][:]

#     coord_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
#     domain_obs = ufl.Mesh(coord_element)
    
#     mesh_obs = dolfinx.mesh.create_mesh(
#         MPI.COMM_SELF, 
#         cells=cells_obs, 
#         x=points_obs, 
#         e=domain_obs
#     )
#     mesh_obs.topology.create_connectivity(mesh_obs.topology.dim - 1, mesh_obs.topology.dim)

#     dim_disp = 2
#     V_obs = fem.functionspace(mesh_obs, ("CG", 1, (dim_disp,)))
#     u_obs = fem.Function(V_obs, name="displacement")
#     global_indices_obs = mesh_obs.geometry.input_global_indices

#     V_cad = fem.functionspace(mesh_cad, ("CG", 1, (dim_disp,)))
#     u_cad = fem.Function(V_cad, name="displacement_projected")
    
#     # =========================================================================
#     # CONFIGURATION DU MASQUE ET DU CONTOUR (DG0 pour éviter les conflits de DOFs)
#     # =========================================================================
#     print("[Mise à jour] Calcul de l'étiquette géométrique 'is_imported' (Cylindre Z)...")
#     if points_obs.shape[1] == 2:
#         points_3d = np.hstack([points_obs, np.zeros((points_obs.shape[0], 1))])
#     else:
#         points_3d = points_obs[:, :3]
        
#     points_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
#     points_obs_in_cad_space = (tform_h5_to_cad_4D @ points_hom.T).T[:, :3]

#     points_obs_2d = points_obs_in_cad_space[:, :2]
#     tree_obs_2d = KDTree(points_obs_2d)

#     # Utilisation directe de DG0 : une valeur par cellule, indexée sur les cellules locales
#     V_cell = fem.functionspace(mesh_cad, ("DG", 0))
#     u_mask_cell = fem.Function(V_cell, name="is_imported_cell")
#     DISTANCE_THRESHOLD = 2.0  

#     def cell_mask_interpolation_callback(x):
#         # x a une forme (3, num_points) correspondant aux centres d'évaluation (ici les centres des cellules)
#         nodes_cad_2d = x[:2, :].T
#         distances, _ = tree_obs_2d.query(nodes_cad_2d, distance_upper_bound=DISTANCE_THRESHOLD)
#         return np.where(distances <= DISTANCE_THRESHOLD, 1.0, 0.0)

#     u_mask_cell.interpolate(cell_mask_interpolation_callback)
#     u_mask_cell.x.scatter_forward()
#     cell_values = u_mask_cell.x.array  # Directement mappé sur l'index des cellules locales

#     # Construction du MeshTag de contour
#     tdim = mesh_cad.topology.dim
#     fdim = tdim - 1
#     mesh_cad.topology.create_connectivity(tdim, fdim)
#     mesh_cad.topology.create_connectivity(fdim, tdim)

#     facet_to_cells = mesh_cad.topology.connectivity(fdim, tdim)
#     num_facets = mesh_cad.topology.index_map(fdim).size_local + mesh_cad.topology.index_map(fdim).num_ghosts

#     boundary_facets = []
#     for facet in range(num_facets):
#         cells = facet_to_cells.links(facet)
#         if len(cells) == 2:
#             # Facette interne : changement d'état entre 1 et 0
#             if cell_values[cells[0]] != cell_values[cells[1]]:
#                 boundary_facets.append(facet)
#         elif len(cells) == 1:
#             # Facette frontière du domaine : si la cellule associée est importée
#             if cell_values[cells[0]] == 1.0:
#                 boundary_facets.append(facet)

#     boundary_facets = np.array(boundary_facets, dtype=np.int32)
#     values = np.full_like(boundary_facets, 1, dtype=np.int32)

#     contour_meshtags = dolfinx.mesh.meshtags(mesh_cad, fdim, boundary_facets, values)
#     contour_meshtags.name = "contour_is_imported"

#     # Optionnel : Si tu as absolument besoin de stocker u_mask en CG1 pour l'affichage continu
#     V_mask_cg1 = fem.functionspace(mesh_cad, ("CG", 1))
#     u_mask = fem.Function(V_mask_cg1, name="is_imported")
#     def cg1_mask_callback(x):
#         nodes_cad_2d = x[:2, :].T
#         distances, _ = tree_obs_2d.query(nodes_cad_2d, distance_upper_bound=DISTANCE_THRESHOLD)
#         return np.where(distances <= DISTANCE_THRESHOLD, 1.0, 0.0)
#     u_mask.interpolate(cg1_mask_callback)
#     u_mask.x.scatter_forward()

#     # =========================================================================
#     # 3. TRANSFORMATION DE COORDONNÉES (INVERSION)
#     # =========================================================================
#     print("[3/4] Inversion de la matrice de transformation spatiale...")
#     tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

#     # =========================================================================
#     # 4. BOUCLE TEMPORELLE DE LECTURE H5 ET PROJECTION
#     # =========================================================================
#     print(f"[4/4] Lancement de la projection temporelle brute vers : {output_xdmf_path}")
    
#     with dolfinx.io.XDMFFile(mesh_cad.comm, output_xdmf_path, "w") as xdmf_out:
#         xdmf_out.write_mesh(mesh_cad)
#         # On écrit les meshtags du contour une seule fois au début
#         xdmf_out.write_meshtags(contour_meshtags, mesh_cad.geometry)

#         R_h5_to_cad = tform_h5_to_cad_4D[:3, :3]

#         with h5py.File(h5_path, "r") as f:
#             for t, ds_path in steps:
#                 data_step = f[ds_path][:]
#                 num_nodes = data_step.shape[0]
                
#                 disp_orig_3d = np.zeros((num_nodes, 3))
#                 disp_orig_3d[:, :2] = data_step[:, :2]
                
#                 disp_transformed_3d = (R_h5_to_cad @ disp_orig_3d.T).T
                
#                 ux = np.nan_to_num(disp_transformed_3d[:, 0], nan=0.0)
#                 uy = np.nan_to_num(disp_transformed_3d[:, 1], nan=0.0)

#                 if global_indices_obs is not None and global_indices_obs.size > 0:
#                     permuted_ux = ux[global_indices_obs]
#                     permuted_uy = uy[global_indices_obs]
#                 else:
#                     permuted_ux = ux
#                     permuted_uy = uy

#                 u_obs_array = u_obs.x.array
#                 u_obs_array[0::2] = permuted_ux
#                 u_obs_array[1::2] = permuted_uy
#                 u_obs.x.scatter_forward()

#                 # Ton interpolation externe personnalisée vers le maillage CAD
#                 interpolate_displacement_obs_mesh_to_cad_mesh_2D(u_obs, u_cad, tform_cad_to_img_4D)

#                 xdmf_out.write_function(u_cad, t)
#                 xdmf_out.write_function(u_mask, t)
#                 print(f" -> Pas t={t} projeté avec succès.")
#     # =========================================================================
#     # EXPORT DU SOUS-MAILLAGE DÉDIÉ POUR LE CONTOUR
#     # =========================================================================
#     print("[Nouveau] Extraction et export du sous-maillage de contour...")
    
#     try:
#         # CORRECTION : On récupère uniquement le premier élément [0] qui est le Mesh
#         submesh_contour = dolfinx.mesh.create_submesh(
#             mesh_cad, 
#             fdim, 
#             boundary_facets
#         )[0]
        
#         # Définition du chemin d'accès pour le fichier de contour
#         contour_xdmf_path = output_xdmf_path.replace(".xdmf", "_contour.xdmf")
        
#         # Écriture du maillage de contour seul
#         with dolfinx.io.XDMFFile(mesh_cad.comm, contour_xdmf_path, "w") as xdmf_contour:
#             xdmf_contour.write_mesh(submesh_contour)
#         print(f" -> Sous-maillage de contour exporté avec succès dans : {contour_xdmf_path}")
        
#     except Exception as e:
#         print(f"[Erreur] Impossible de créer ou d'exporter le sous-maillage : {e}")

def project_h5_series_to_cad_mesh_mask_2(
    h5_path: str, 
    mesh_cad: dolfinx.mesh.Mesh, 
    tform_h5_to_cad_4D: NDArray, 
    output_xdmf_path: str
) -> None:
    """Lit une série temporelle FEniCSx directement depuis son fichier H5 (via h5py),
    reconstruit le maillage source et projette le champ de déplacement vectoriel
    sur un maillage cible dolfinx.mesh.Mesh pour chaque pas de temps.
    
    Calcule et exporte également le sous-maillage volumique (tétraèdres) de la zone interpolée.
    """
    # =========================================================================
    # 1. PARSING DE LA STRUCTURE ET DES PAS DE TEMPS
    # =========================================================================
    xdmf_path = h5_path.replace(".h5", ".xdmf")
    steps = []
    geom_path = "/Mesh/Grid/geometry"
    topo_path = "/Mesh/Grid/topology"

    if os.path.exists(xdmf_path):
        print(f"[1/4] Analyse du fichier métadonnées {xdmf_path} pour identifier les datasets...")
        tree = ET.parse(xdmf_path)
        root = tree.getroot()
        
        try:
            geom_path = next(root.iter("Geometry")).find("DataItem").text.split(":")[-1].strip()
            topo_path = next(root.iter("Topology")).find("DataItem").text.split(":")[-1].strip()
        except Exception:
            pass 

        for grid in root.findall(".//Grid"):
            time_node = grid.find("Time")
            attr_node = grid.find(".//Attribute[@Name='displacement']")
            if time_node is not None and attr_node is not None:
                t_val = float(time_node.get("Value"))
                di = attr_node.find("DataItem")
                if di is not None:
                    ds_path = di.text.split(":")[-1].strip()
                    steps.append((t_val, ds_path))
    else:
        print(f"[1/4] Fichier XDMF non trouvé. Inspection directe du fichier H5 : {h5_path}")
        with h5py.File(h5_path, "r") as f:
            mesh_key = list(f["Mesh"].keys())[0]
            geom_path = f"/Mesh/{mesh_key}/geometry"
            topo_path = f"/Mesh/{mesh_key}/topology"
            func_group = f["Function/displacement"]
            for k in sorted(list(func_group.keys()), key=lambda x: int(x) if x.isdigit() else x):
                try:
                    t_val = float(k)
                except ValueError:
                    t_val = float(len(steps))
                steps.append((t_val, f"Function/displacement/{k}"))

    if not steps:
        raise ValueError("Aucun pas de temps ou champ de déplacement n'a pu être localisé.")
    
    steps.sort(key=lambda x: x[0])
    print(f" -> {len(steps)} pas de temps détectés.")

    # =========================================================================
    # 2. CHARGEMENT DU MAILLAGE SOURCE DEPUIS LE H5 ET INSTANCIATION DOLFINX
    # =========================================================================
    print(f"[2/4] Extraction des tableaux de maillage depuis le H5...")
    with h5py.File(h5_path, "r") as f:
        points_obs = f[geom_path][:]
        cells_obs = f[topo_path][:]

    coord_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    domain_obs = ufl.Mesh(coord_element)
    
    mesh_obs = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF, 
        cells=cells_obs, 
        x=points_obs, 
        e=domain_obs
    )
    mesh_obs.topology.create_connectivity(mesh_obs.topology.dim - 1, mesh_obs.topology.dim)

    dim_disp = 2
    V_obs = fem.functionspace(mesh_obs, ("CG", 1, (dim_disp,)))
    u_obs = fem.Function(V_obs, name="displacement")
    global_indices_obs = mesh_obs.geometry.input_global_indices

    V_cad = fem.functionspace(mesh_cad, ("CG", 1, (dim_disp,)))
    u_cad = fem.Function(V_cad, name="displacement_projected")
    
    # =========================================================================
    # CONFIGURATION DU MASQUE ET SÉLECTION DES CELLULES
    # =========================================================================
    print("[Mise à jour] Calcul de l'étiquette géométrique 'is_imported' (Cylindre Z)...")
    if points_obs.shape[1] == 2:
        points_3d = np.hstack([points_obs, np.zeros((points_obs.shape[0], 1))])
    else:
        points_3d = points_obs[:, :3]
        
    points_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_obs_in_cad_space = (tform_h5_to_cad_4D @ points_hom.T).T[:, :3]

    points_obs_2d = points_obs_in_cad_space[:, :2]
    tree_obs_2d = KDTree(points_obs_2d)

    # Utilisation de DG0 : une valeur par cellule
    V_cell = fem.functionspace(mesh_cad, ("DG", 0))
    u_mask_cell = fem.Function(V_cell, name="is_imported_cell")
    DISTANCE_THRESHOLD = 2.0  

    def cell_mask_interpolation_callback(x):
        nodes_cad_2d = x[:2, :].T
        distances, _ = tree_obs_2d.query(nodes_cad_2d, distance_upper_bound=DISTANCE_THRESHOLD)
        return np.where(distances <= DISTANCE_THRESHOLD, 1.0, 0.0)

    u_mask_cell.interpolate(cell_mask_interpolation_callback)
    u_mask_cell.x.scatter_forward()
    cell_values = u_mask_cell.x.array  

    # Récupération de la dimension topologique du maillage CAD (3 pour des tétraèdres)
    tdim = mesh_cad.topology.dim

    # Identification directe des indices des cellules se trouvant dans la zone d'intérêt
    cell_indices = np.where(cell_values == 1.0)[0].astype(np.int32)

    # Création de MeshTags au niveau des cellules (optionnel, pour l'écrire dans le XDMF principal)
    volume_meshtags = dolfinx.mesh.meshtags(mesh_cad, tdim, cell_indices, np.full_like(cell_indices, 1, dtype=np.int32))
    volume_meshtags.name = "zone_interet_volume"

    # Si vous avez besoin de stocker u_mask en CG1 pour l'affichage continu
    V_mask_cg1 = fem.functionspace(mesh_cad, ("CG", 1))
    u_mask = fem.Function(V_mask_cg1, name="is_imported")
    def cg1_mask_callback(x):
        nodes_cad_2d = x[:2, :].T
        distances, _ = tree_obs_2d.query(nodes_cad_2d, distance_upper_bound=DISTANCE_THRESHOLD)
        return np.where(distances <= DISTANCE_THRESHOLD, 1.0, 0.0)
    u_mask.interpolate(cg1_mask_callback)
    u_mask.x.scatter_forward()

    # =========================================================================
    # 3. TRANSFORMATION DE COORDONNÉES (INVERSION)
    # =========================================================================
    print("[3/4] Inversion de la matrice de transformation spatiale...")
    tform_cad_to_img_4D = np.linalg.inv(tform_h5_to_cad_4D)

    # =========================================================================
    # 4. BOUCLE TEMPORELLE DE LECTURE H5 ET PROJECTION
    # =========================================================================
    print(f"[4/4] Lancement de la projection temporelle brute vers : {output_xdmf_path}")
    
    with dolfinx.io.XDMFFile(mesh_cad.comm, output_xdmf_path, "w") as xdmf_out:
        xdmf_out.write_mesh(mesh_cad)
        # On écrit les meshtags volumiques une seule fois au début
        xdmf_out.write_meshtags(volume_meshtags, mesh_cad.geometry)

        R_h5_to_cad = tform_h5_to_cad_4D[:3, :3]

        with h5py.File(h5_path, "r") as f:
            for t, ds_path in steps:
                data_step = f[ds_path][:]
                num_nodes = data_step.shape[0]
                
                disp_orig_3d = np.zeros((num_nodes, 3))
                disp_orig_3d[:, :2] = data_step[:, :2]
                
                disp_transformed_3d = (R_h5_to_cad @ disp_orig_3d.T).T
                
                ux = np.nan_to_num(disp_transformed_3d[:, 0], nan=0.0)
                uy = np.nan_to_num(disp_transformed_3d[:, 1], nan=0.0)

                if global_indices_obs is not None and global_indices_obs.size > 0:
                    permuted_ux = ux[global_indices_obs]
                    permuted_uy = uy[global_indices_obs]
                else:
                    permuted_ux = ux
                    permuted_uy = uy

                u_obs_array = u_obs.x.array
                u_obs_array[0::2] = permuted_ux
                u_obs_array[1::2] = permuted_uy
                u_obs.x.scatter_forward()

                # Interpolation externe personnalisée vers le maillage CAD
                interpolate_displacement_obs_mesh_to_cad_mesh_2D(u_obs, u_cad, tform_cad_to_img_4D)

                xdmf_out.write_function(u_cad, t)
                xdmf_out.write_function(u_mask, t)
                print(f" -> Pas t={t} projeté avec succès.")

    # =========================================================================
    # EXPORT DU SOUS-MAILLAGE DÉDIÉ POUR LE VOLUME DE L'INTERSECTION
    # =========================================================================
    print("[Nouveau] Extraction et export du sous-maillage volumique de la zone d'intérêt...")
    
    try:
        # Extraction du sous-maillage à partir de tdim (dimension 3 -> tétraèdres)
        submesh_volume = dolfinx.mesh.create_submesh(
            mesh_cad, 
            tdim, 
            cell_indices
        )[0]
        
        # Définition du chemin d'accès pour le fichier volumique isolé
        volume_xdmf_path = output_xdmf_path.replace(".xdmf", "_volume_interet.xdmf")
        
        # Écriture du maillage tétraédrique seul
        with dolfinx.io.XDMFFile(mesh_cad.comm, volume_xdmf_path, "w") as xdmf_vol:
            xdmf_vol.write_mesh(submesh_volume)
        print(f" -> Sous-maillage volumique exporté avec succès dans : {volume_xdmf_path}")
        
    except Exception as e:
        print(f"[Erreur] Impossible de créer ou d'exporter le sous-maillage volumique : {e}")


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
    geom_path = "/Mesh/Grid/geometry"
    topo_path = "/Mesh/Grid/topology"

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
        points_obs = f[geom_path][:]
        cells_obs = f[topo_path][:]
        
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
    u_obs = fem.Function(V_obs, name="displacement_resampled")

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

            # MODIFICATION ICI : On passe l'index entier `float(i)` au lieu du temps `t`
            xdmf_out.write_function(u_obs, float(i))

    print(f"[Succès] Nouveau fichier temporel généré avec succès ({target_num_steps} pas).")


# if __name__ == "__main__":
#     resample_h5_time_series("results/projection_cad_temporelle.h5","results/resampling_time_series_linear.xdmf",50)


if __name__ == "__main__":
    import os

    # =========================================================================
    # 1. CONFIGURATION DES CHEMINS (À MODIFIER)
    # =========================================================================
    # Mettez ici les vrais chemins vers vos fichiers de test
    H5_FILE = "dic_series_complete.h5"
    GMSH_FILE = "astar_6mm.xdmf"
    OUTPUT_XDMF = "results/projection_cad_temporelle_mask.xdmf"
    
    # Paramètre alpha pour la triangulation Delaunay du nuage de points DIC
    ALPHA_TRIANGULATION = 20.0 
    domain = load_and_write_mesh(GMSH_FILE)
    # =========================================================================
    # 2. CONFIGURATION DE LA MATRICE DE TRANSFORMATION (H5 -> CAD)
    # =========================================================================
    ref_image = skimage.io.imread("N_E_basler_0000.tif", as_gray=True)
    tform_cad_to_img_4d = calibrate_2d_manual(domain,ref_image)
    tform_h5_to_cad = np.linalg.inv(tform_cad_to_img_4d)
    # Astuce : Si vous avez une translation connue (ex: +10mm en X et -5mm en Y) :
    # tform_h5_to_cad[0, 3] = 10.0
    # tform_h5_to_cad[1, 3] = -5.0

    # =========================================================================
    # 3. SÉCURITÉ ET LANCEMENT DU TRAITEMENT
    # =========================================================================
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
            # Exécution de la fonction globale de traitement
            project_h5_series_to_cad_mesh_mask_2(
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