import basix
import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import pyvista as pv
import skimage.transform
from numpy.typing import NDArray
from petsc4py import PETSc
from plasticity_simu import load_and_write_mesh
import matplotlib.pyplot as plt
import cv2

def rotation_matrix_2D(theta: float) -> NDArray:
    """Create a 2D rotation matrix."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

def get_xy_bounding_box(x: NDArray) -> NDArray:
    """Compute a 2D bounding box for a given set of points."""
    x_min = np.min(x[:, 0])
    y_min = np.min(x[:, 1])
    x_max = np.max(x[:, 0])
    y_max = np.max(x[:, 1])
    return np.array([[x_min, y_min], [x_max, y_max]])


def get_xyz_bounding_box(x: NDArray):
    """Compute a 3D bounding box for a given set of points."""
    x_min = np.min(x[:, 0])
    y_min = np.min(x[:, 1])
    z_min = np.min(x[:, 2])
    x_max = np.max(x[:, 0])
    y_max = np.max(x[:, 1])
    z_max = np.max(x[:, 2])
    return np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])


def affine_transform_bounding_boxes_2D(
    bb0: NDArray, bb1: NDArray
) -> tuple[NDArray, NDArray]:
    """Construct an affine transformation that maps bb0 to bb1.

    In the sense that the mapped bb0 is centered
    inside bb1 while having a maximal size
    and keeping original aspect ratio.
    """
    w0 = np.linalg.norm(bb0[1, 0] - bb0[0, 0])
    h0 = np.linalg.norm(bb0[1, 1] - bb0[0, 1])

    w1 = np.linalg.norm(bb1[1, 0] - bb1[0, 0])
    h1 = np.linalg.norm(bb1[1, 1] - bb1[0, 1])

    w_ratio = w1 / w0
    h_ratio = h1 / h0

    resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio

    center0 = (bb0[0] + bb0[1]) / 2.0
    center1 = (bb1[0] + bb1[1]) / 2.0

    translation = (center1 / resize_ratio) - center0

    return np.array([resize_ratio, resize_ratio]), translation

def affine_transform_2D_to_4D(scaling: NDArray, translation: NDArray) -> NDArray:
    """Create a 4D transformation matrix from 2D components."""
    assert len(scaling) == 2
    assert len(translation) == 2

    return np.array(
        [
            [scaling[0], 0, 0, scaling[0] * translation[0]],
            [0, scaling[1], 0, scaling[1] * translation[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def transform_4D_to_3D(transform_4D: NDArray) -> NDArray:
    """Convert a 4D affine transformation to a 3D affine transformation."""
    transform_3D = np.zeros((3, 3), dtype=transform_4D.dtype)
    transform_3D[:2, :2] = transform_4D[:2, :2]
    transform_3D[:2, 2] = transform_4D[:2, 3]
    transform_3D[2, :2] = transform_4D[3, :2]
    transform_3D[2, 2] = transform_4D[3, 3]
    return transform_3D


def transform_3D_to_4D(transform_3D: NDArray) -> NDArray:
    """Convert a 3D affine transformation to a 4D affine transformation."""
    transform_4D = np.zeros((4, 4), dtype=transform_3D.dtype)
    transform_4D[:2, :2] = transform_3D[:2, :2]
    transform_4D[:2, 3] = transform_3D[:2, 2]
    transform_4D[3, :2] = transform_3D[2, :2]
    transform_4D[3, 3] = transform_3D[2, 2]
    transform_4D[2, 2] = 1.0
    return transform_4D


def img_uniform_grid(img_shape: tuple[int, int]) -> pv.ImageData:
    """Construct a uniform grid that represents an image."""
    h, w = img_shape
    dims = (w, h, 1)
    spacing = (1, 1, 1)
    origin = (0, 0, 0)
    return pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)


def dolfinx_mesh_to_pv_mesh(
    dolfinx_mesh: dolfinx.mesh.Mesh,
) -> pv.UnstructuredGrid:
    """Convert a dolfinx mesh to a pyvista mesh."""
    topology, cell_types, x = dolfinx.plot.vtk_mesh(
        dolfinx_mesh, dim=dolfinx_mesh.topology.dim
    )
    return pv.UnstructuredGrid(topology, cell_types, x)


def construct_reference_cad_image(
    cad_mesh: dolfinx.mesh.Mesh, ref_img_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """Construct a reference CAD image clamped to the grid."""
    pv_mesh = dolfinx_mesh_to_pv_mesh(cad_mesh)

    hauteur, largeur = ref_img_shape
    probe_grid = img_uniform_grid((hauteur, largeur))

    bounding_box_xy = get_xy_bounding_box(cad_mesh.geometry.x)
    bounding_box_img = np.array([[0.0, 0.0], [largeur, hauteur]])
    
    scaling, translation = affine_transform_bounding_boxes_2D(
        bounding_box_xy, bounding_box_img
    )

    transformation_4D = affine_transform_2D_to_4D(scaling, translation)
    pv_mesh.transform(transformation_4D, inplace=True)

    DG0 = dolfinx.fem.functionspace(cad_mesh, ("DG", 0))
    dummy_fn = dolfinx.fem.Function(DG0)
    dummy_fn.x.array[:] = 1.0
    pv_mesh.cell_data["dummy"] = dummy_fn.x.array
    pv_mesh.set_active_scalars("dummy")

    probed_grid = probe_grid.sample(pv_mesh)
    probed_grid.set_active_scalars("dummy")

    cad_ref_img = np.reshape(probed_grid.active_scalars, (hauteur, largeur))
    cad_ref_img = np.nan_to_num(cad_ref_img, nan=0.0)
    return cad_ref_img, transformation_4D


def register_imgs_manual(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """Perform manual image registration using point-clicking with OpenCV.

    Args:
    ----
        img0 (NDArray): Reference CAD image (source)
        img1 (NDArray): Real DIC image (destination)

    Returns:
    -------
        NDArray: 4x4 homogeneous affine transformation matrix
    """
    h, w = img0.shape[:2]

    # Normalisation et conversion en uint8 pour OpenCV
    vis0_base = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0.copy()
    vis1_base = (img1 * 255).astype(np.uint8) if img1.dtype != np.uint8 else img1.copy()

    pts0 = []
    pts1 = []

    window_name = "Calibration Manuelle - ENTRER pour valider, ECHAP pour annuler"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Gestionnaire d'événements de la souris
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < w:
                pts0.append([x, y])
                print(f"Point CAO sélectionné : [{x}, {y}]")
            else:
                pts1.append([x - w, y])
                print(f"Point Image sélectionné : [{x - w}, {y}]")

    cv2.setMouseCallback(window_name, click_event)

    print("\n--- INSTRUCTIONS POUR LE POINTAGE ---")
    print("1. Cliquez sur un point structurel marquant à GAUCHE (Image CAO).")
    print("2. Cliquez sur son homologue à DROITE (Image Réelle).")
    print("3. Répétez l'opération pour au moins 3 ou 4 paires de points.")
    print("4. Appuyez sur ENTRÉE dans la fenêtre pour valider, ou sur ECHAP pour quitter.\n")

    while True:
        # Copie fraîche pour rafraîchir le tracé des cercles et index
        vis0 = cv2.cvtColor(vis0_base, cv2.COLOR_GRAY2BGR)
        vis1 = cv2.cvtColor(vis1_base, cv2.COLOR_GRAY2BGR)

        # Dessin des points CAO (Rouge)
        for idx, pt in enumerate(pts0):
            cv2.circle(vis0, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.putText(vis0, str(idx + 1), (pt[0] + 8, pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Dessin des points réels (Vert)
        for idx, pt in enumerate(pts1):
            cv2.circle(vis1, (pt[0], pt[1]), 5, (0, 255, 0), -1)
            cv2.putText(vis1, str(idx + 1), (pt[0] + 8, pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Concaténation horizontale pour l'affichage côte à côte
        combined = np.hstack((vis0, vis1))
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Touche Entrée
            break
        elif key == 27:  # Touche Échap
            print("Calibration annulée par l'utilisateur.")
            cv2.destroyAllWindows()
            return np.identity(4)

    cv2.destroyAllWindows()

    # Vérification de la cohérence des listes de points
    min_pts = min(len(pts0), len(pts1))
    if min_pts < 3:
        print(f"Erreur : Pas assez de points appariés (Requis : >= 3, Reçu : {min_pts}).")
        return np.identity(4)

    if len(pts0) != len(pts1):
        print(f"Attention : Déséquilibre du nombre de points. Alignement effectué sur les {min_pts} premières paires.")

    np_pts0 = np.array(pts0[:min_pts], dtype=np.float32)
    np_pts1 = np.array(pts1[:min_pts], dtype=np.float32)

    # Estimation robuste de la matrice affine 2D (2x3)
    M, inliers = cv2.estimateAffine2D(np_pts0, np_pts1)

    if M is None:
        print("Erreur de calcul mathématique lors de l'estimation de la matrice affine. Retour Matrice Identité.")
        return np.identity(4)

    # Conversion de la matrice affine 2D (2x3) en matrice homogène 4D (4x4)
    tform_ref_to_img_4d = np.identity(4)
    tform_ref_to_img_4d[:2, :2] = M[:2, :2]
    tform_ref_to_img_4d[:2, 3] = M[:2, 2]

    return tform_ref_to_img_4d


def calibrate_2d_manual(mesh: dolfinx.mesh.Mesh, ref_img: NDArray) -> NDArray:
    """Perform 2d DIC calibration via manual point selection."""
    cad_ref_img, tform_cad_to_ref_4d = construct_reference_cad_image(mesh, np.shape(ref_img))
    
    # Appel de la nouvelle routine manuelle OpenCV
    tform_ref_to_img_4d = register_imgs_manual(cad_ref_img, ref_img)
    
    return tform_ref_to_img_4d @ tform_cad_to_ref_4d


def check_calibration_2d(
    mesh: dolfinx.mesh.Mesh, ref_img: NDArray, tform_cad_to_img_4d: NDArray
) -> None:
    """Display calibration results using PyVista."""
    h, w = np.shape(ref_img)
    img_pv = pv.ImageData(dimensions=(w + 1, h + 1, 1))
    img_pv.cell_data["gray_level"] = ref_img.flatten()

    mesh_pv = dolfinx_mesh_to_pv_mesh(mesh)
    mesh_pv = mesh_pv.transform(tform_cad_to_img_4d, inplace=False)
    mesh_pv = mesh_pv.translate([0.0, 0.0, 1.0], inplace=False)

    p = pv.Plotter()
    p.add_mesh(img_pv, cmap="gray")
    p.add_mesh(mesh_pv, show_edges=True, color="red", opacity=0.5)
    p.show()


if __name__ == "__main__":
    # Chargement du maillage et de l'image de référence
    domain = load_and_write_mesh("3specimen.msh")
    ref_image = skimage.io.imread("speckle_3specimen.png", as_gray=True)
    
    # Calcul de la calibration par pointage manuel
    tfo = calibrate_2d_manual(domain, ref_image)
    
    print("\nMatrice de transformation finale 4x4 obtenue :")
    print(tfo)
    
    # Visualisation 3D finale du recalage
    check_calibration_2d(domain, ref_image, tfo)
