import os
import numpy as np
import pandas as pd
import pyvista as pv
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
from scipy.interpolate import NearestNDInterpolator

def import_vic2d_csv_to_pv(csv_path: str, alpha: float = 20.0) -> pv.UnstructuredGrid:
    """
    Étape 1 : Lit le fichier CSV de Vic-2D en suivant la logique de vic.py.
    Filtre les points non corrélés (eyy == 0) et construit le maillage PyVista.
    """
    # 1. Lecture directe (sans sauter de ligne comme dans ton fichier vic.py)
    data = pd.read_csv(csv_path)
    
    # Nettoyage des en-têtes (suppression des espaces et des guillemets doubles)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    
    # Reconstruction du dictionnaire de données temporaire
    d = {}
    for i, name in enumerate(names):
        d[name] = data.values[:, i]
        
    # 2. Nettoyage des points non corrélés (copie conforme de ton vic.py)
    if "eyy" in d:
        num_values = np.shape(data.values)[0]
        retained_indices = list(filter(lambda k: d["eyy"][k] != 0, np.arange(num_values)))
        for name in names:
            d[name] = d[name][retained_indices]
    else:
        print("[WARNING] Colonne 'eyy' manquante pour filtrer les points non corrélés.")

    # 3. Identification robuste des axes géométriques (gère X ou x selon l'export Vic)
    if "X" in names:
        col_x, col_y = "X", "Y"
    elif "x" in names:
        col_x, col_y = "x", "y"
    else:
        raise ValueError('Le fichier CSV ne contient pas de colonne de coordonnées valide ("X" ou "x")')

    # Identification robuste des colonnes de déplacement (U ou u)
    if "U" in names:
        col_u, col_v = "U", "V"
    elif "u" in names:
        col_u, col_v = "u", "v"
    else:
        raise ValueError('Le fichier CSV ne contient pas de colonne de déplacement valide ("U" ou "u")')

    # 4. Création du nuage de points et triangulation Delaunay
    points = np.stack([d[col_x], d[col_y], 0 * d[col_x]], axis=-1)
    cloud = pv.PolyData(points)
    
    print(f"[INFO] Triangulation des points DIC avec alpha={alpha}...")
    volume = cloud.delaunay_2d(alpha=alpha)
    
    # Injection de tous les champs d'origine dans le maillage PyVista
    for name in names:
        volume[name] = d[name]
        
    mesh_pv = pv.UnstructuredGrid(volume)
    
    # On stocke des pointeurs explicites pour l'étape d'extraction DOLFINx
    mesh_pv.point_data["u_meas"] = d[col_u]
    mesh_pv.point_data["v_meas"] = d[col_v]
    
    return mesh_pv

def create_dolfinx_mesh_from_pv(pv_mesh: pv.UnstructuredGrid) -> dolfinx.mesh.Mesh:
    """
    Étape 2 : Extrait la topologie du maillage PyVista pour générer un objet
    dolfinx.mesh.Mesh natif et compatible avec MPI (FeniCSx v0.10).
    """
    print("[INFO] Conversion du maillage PyVista vers DOLFINx (v0.10)...")
    
    # Extraction des nœuds géométriques 2D
    coords = pv_mesh.points[:, :2].astype(np.float64)
    
    # Extraction sécurisée de la connectivité des triangles via cells_dict
    if pv.CellType.TRIANGLE in pv_mesh.cells_dict:
        cells = pv_mesh.cells_dict[pv.CellType.TRIANGLE].astype(np.int32)
    else:
        num_cells = pv_mesh.n_cells
        cells = pv_mesh.cells.reshape((num_cells, 4))[:, 1:].astype(np.int32)
    
    # Définition de l'élément géométrique requis par l'API FeniCSx v0.10
    geom_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(geom_element)
    
    # RECTIFICATION : 'domain' (3e argument) et 'coords' (4e argument)
    dolfinx_mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, domain, coords)
    
    return dolfinx_mesh


def extract_u_to_dolfinx(pv_mesh: pv.UnstructuredGrid, dolfinx_mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Function:
    """
    Étape 3 : Crée l'espace fonctionnel vectoriel et transfère proprement
    le déplacement 'u' de la DIC sur les degrés de liberté (DoF) de FeniCSx.
    """
    print("[INFO] Extraction et transfert du champ de déplacement vectoriel 'u'...")
    
    # 1. Définition de l'espace vectoriel P1 (Lagrange Continu Degré 1, Dimension 2)
    V = dolfinx.fem.functionspace(dolfinx_mesh, ("Lagrange", 1, (2,)))
    u_function = dolfinx.fem.Function(V, name="u_obs")
    
    # 2. Création des interpolateurs basés sur les coordonnées de la DIC
    interp_u = NearestNDInterpolator(pv_mesh.points[:, :2], pv_mesh.point_data["u_meas"])
    interp_v = NearestNDInterpolator(pv_mesh.points[:, :2], pv_mesh.point_data["v_meas"])
    
    # 3. Fonction d'évaluation pour la méthode .interpolate() de FeniCSx
    def u_expression(x):
        # x contient les coordonnées des DoFs fournies par FeniCSx au format (3, num_points)
        values = np.zeros((2, x.shape[1]))
        values[0, :] = interp_u(x[0, :], x[1, :])  # Composante horizontale (U)
        values[1, :] = interp_v(x[0, :], x[1, :])  # Composante verticale (V)
        return values

    # 4. Interpolation et synchronisation MPI des vecteurs fantômes
    u_function.interpolate(u_expression)
    u_function.x.scatter_forward()
    
    return u_function


# =============================================================================
# SCRIPT DE TEST PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Remplace par le nom exact ou le chemin de ton fichier CSV de Vic-2D
    CSV_VIC2D = "VK03-1-16-0201_0.csv" 
    
    if not os.path.exists(CSV_VIC2D):
        print(f"[ERREUR] Le fichier {CSV_VIC2D} est introuvable dans le répertoire courant.")
    else:
        # Exécution séquentielle des 3 étapes
        pv_grid = import_vic2d_csv_to_pv(CSV_VIC2D, alpha=20.0)
        mesh_fx = create_dolfinx_mesh_from_pv(pv_grid)
        u_meas_fx = extract_u_to_dolfinx(pv_grid, mesh_fx)
        
        print("\n[SUCCÈS] Pipeline d'importation validé !")
        print(f" -> Nombre de triangles générés dans FeniCSx : {mesh_fx.topology.index_map(mesh_fx.topology.dim).size_global}")
        
        # Petit calcul de contrôle : calcul de la norme L2 du déplacement importé
        norm_l2 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u_meas_fx, u_meas_fx) * ufl.dx))
        print(f" -> Norme L2 du déplacement dans FeniCSx : {norm_l2:.4e}")