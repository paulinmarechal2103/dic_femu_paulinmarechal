import pandas as pd
import numpy as np
import pyvista as pv
import meshio
import dolfinx.fem as fem
import dolfinx.io
from mpi4py import MPI
from numpy.typing import NDArray

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
    
    x_coords = d["x_c"][valid_mask]
    y_coords = d["y_c"][valid_mask]
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

import os
import pandas as pd
import numpy as np
import pyvista as pv
import meshio
import ufl
import dolfinx.fem as fem
import dolfinx.io
from mpi4py import MPI
from scipy.spatial import KDTree

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

# =========================================================================
# EXÉCUTION
# =========================================================================
if __name__ == "__main__":
    dossier_csv = "/home/pmarechal/Documents/DP_N_E/images_and_csv"
    
    process_csv_series_fenicsx(
        folder_path=dossier_csv, 
        output_xdmf="dic_series_complete.xdmf", 
        file_prefix="N_E_basler_", 
        alpha=20.0
    )