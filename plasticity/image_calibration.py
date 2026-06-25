import basix
import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import imreg_dft
#import numba
import numpy as np
import pyvista as pv
import skimage.transform
from numpy.typing import NDArray
from petsc4py import PETSc
from skimage.filters import threshold_multiotsu, threshold_otsu, gaussian
from plasticity_simu import load_and_write_mesh
import matplotlib.pyplot as plt
import cv2

def rotation_matrix_2D(theta: float) -> NDArray:
    """Create a 2D rotation matrix.

    Args:
    ----
        theta (float): rotation angle

    Returns:
    -------
        (2, 2) NDArray: the rotation matrix

    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

def get_xy_bounding_box(x: NDArray) -> NDArray:
    """Compute a 2D bounding box for a given set of points.

    Args:
    ----
        x (NDArray): set of points

    Returns:
    -------
        (2, 2) NDArray: bounding box

    """
    x_min = np.min(x[:, 0])
    y_min = np.min(x[:, 1])
    x_max = np.max(x[:, 0])
    y_max = np.max(x[:, 1])
    return np.array([[x_min, y_min], [x_max, y_max]])


def get_xyz_bounding_box(x: NDArray):
    """Compute a 3D bounding box for a given set of points.

    Args:
    ----
        x (NDArray): set of points

    Returns:
    -------
        (2, 3) NDArray: bounding box

    """
    x_min = np.min(x[:, 0])
    y_min = np.min(x[:, 1])
    z_min = np.min(x[:, 2])
    x_max = np.max(x[:, 0])
    y_max = np.max(x[:, 1])
    z_max = np.max(x[:, 2])
    return np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])


# def affine_transform_bounding_boxes_2D(
#     bb0: NDArray, bb1: NDArray
# ) -> tuple[NDArray, NDArray]:
#     """Construct an affine transformation that maps bb0 to bb1.

#     In the sense that the mapped bb0 is placed
#     on the bottom left of bb1 while having a maximal size
#     and keeping original aspect ratio.

#     Bounding boxes format: NDArray [[x_min, y_min], [x_max, y_max]]

#     Args:
#     ----
#         bb0 ((2, 2) NDArray): bounding box to transform
#         bb1 (NDArray): destination bounding box

#     Returns:
#     -------
#         tuple[NDArray, NDArray]: scaling, translation

#     """
#     w0 = np.linalg.norm(bb0[0, 0] - bb0[1, 0])
#     h0 = np.linalg.norm(bb0[0, 1] - bb0[1, 1])

#     w1 = np.linalg.norm(bb1[0, 0] - bb1[1, 0])
#     h1 = np.linalg.norm(bb1[0, 1] - bb1[1, 1])

#     w_ratio = w1 / w0
#     h_ratio = h1 / h0

#     # select transformation ratio
#     resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio

#     # compute translation to origin
#     translation = bb1[0] - bb0[0]

#     return np.array([resize_ratio, resize_ratio]), translation

def affine_transform_bounding_boxes_2D(
    bb0: NDArray, bb1: NDArray
) -> tuple[NDArray, NDArray]:
    """Construct an affine transformation that maps bb0 to bb1.

    In the sense that the mapped bb0 is centered
    inside bb1 while having a maximal size
    and keeping original aspect ratio.

    Bounding boxes format: NDArray [[x_min, y_min], [x_max, y_max]]

    Args:
    ----
        bb0 ((2, 2) NDArray): bounding box to transform (CAD)
        bb1 (NDArray): destination bounding box (Image)

    Returns:
    -------
        tuple[NDArray, NDArray]: scaling, translation

    """
    w0 = np.linalg.norm(bb0[1, 0] - bb0[0, 0])
    h0 = np.linalg.norm(bb0[1, 1] - bb0[0, 1])

    w1 = np.linalg.norm(bb1[1, 0] - bb1[0, 0])
    h1 = np.linalg.norm(bb1[1, 1] - bb1[0, 1])

    w_ratio = w1 / w0
    h_ratio = h1 / h0

    # Sélection du ratio maximal respectant l'aspect ratio
    resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio

    # Calcul des centres des deux boîtes englobantes
    center0 = (bb0[0] + bb0[1]) / 2.0
    center1 = (bb1[0] + bb1[1]) / 2.0

    # La transformation appliquée ensuite est : x' = scaling * (x + translation)
    # Pour que center0 devienne center1, on ajuste la translation :
    translation = (center1 / resize_ratio) - center0

    return np.array([resize_ratio, resize_ratio]), translation

def affine_transform_2D_to_4D(scaling: NDArray, translation: NDArray) -> NDArray:
    """Create a 4D transformation matrix from 2D components.

    Args:
    ----
        scaling ((2) NDArray): scaling of the transformation
        translation ((2) NDArray): translation of the transformation

    Returns:
    -------
        (4, 4) NDArray: 4D representation of the affine transform

    """
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
    """Convert a 4D affine transformation to a 3D affine transformation.

    Args:
    ----
        transform_4D (NDArray): 4D affine transformation

    Returns:
    -------
        NDArray: 3D affine transformation

    """
    transform_3D = np.zeros((3, 3), dtype=transform_4D.dtype)
    transform_3D[:2, :2] = transform_4D[:2, :2]
    transform_3D[:2, 2] = transform_4D[:2, 3]
    transform_3D[2, :2] = transform_4D[3, :2]
    transform_3D[2, 2] = transform_4D[3, 3]
    return transform_3D


def transform_3D_to_4D(transform_3D: NDArray) -> NDArray:
    """Convert a 3D affine transformation to a 4D affine transformation.

    Args:
    ----
        transform_3D (NDArray): 4D affine transformation

    Returns:
    -------
        NDArray: 4D affine transformation

    """
    transform_4D = np.zeros((4, 4), dtype=transform_3D.dtype)
    transform_4D[:2, :2] = transform_3D[:2, :2]
    transform_4D[:2, 3] = transform_3D[:2, 2]
    transform_4D[3, :2] = transform_3D[2, :2]
    transform_4D[3, 3] = transform_3D[2, 2]
    transform_4D[2, 2] = 1.0
    return transform_4D


def img_uniform_grid(img_shape: tuple[int, int]) -> pv.ImageData:
    """Construct a uniform grid that represents an image.

        This is useful for interpolation from a mesh to an image.

    The image can be recovered by accessing the `active_scalars`
    attribute of the underlying pyvista object.

    Args:
    ----
        img_shape (tuple[int, int]): shape of the image to represent

    Returns:
    -------
        pv.ImageData: structured mesh representing the image

    """
    h, w = img_shape
    dims = (w, h, 1)
    spacing = (1, 1, 1)
    origin = (0, 0, 0)
    return pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)


def dolfinx_mesh_to_pv_mesh(
    dolfinx_mesh: dolfinx.mesh.Mesh,
) -> pv.UnstructuredGrid:
    """Convert a dolfinx mesh to a pyvista mesh.

    Args:
    ----
        dolfinx_mesh (dolfinx.mesh.Mesh): a dolfinx mesh

    Returns:
    -------
        a pyvista mesh

    """
    topology, cell_types, x = dolfinx.plot.vtk_mesh(
        dolfinx_mesh, dim=dolfinx_mesh.topology.dim
    )
    return pv.UnstructuredGrid(topology, cell_types, x)


def construct_reference_cad_image(
    cad_mesh: dolfinx.mesh.Mesh, ref_img_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """Construct a reference CAD image.

    The CAD mesh is drawn in a reference position clamped
    to the bottom left of the image. This serves as an
    intermediate representation of the CAD mesh to run
    a DFT correlation algorithm to align the CAD mesh with
    the reference DIC image.

    Args:
    ----
        cad_mesh (dolfinx.mesh.Mesh): CAD mesh
        ref_img_shape (tuple[int, int]): shape of the constructed image

    Returns:
    -------
        tuple[NDArray, NDArray]:
            reference CAD image (gray level), and the
            4D transformation matrix that maps the cad coordinates to
            the cad reference coordinates.

    """
    # convert dolfinx mesh to pyvista
    pv_mesh = dolfinx_mesh_to_pv_mesh(cad_mesh)

    # ref_img_shape est (Hauteur, Largeur)
    hauteur, largeur = ref_img_shape
    probe_grid = img_uniform_grid((hauteur, largeur))

    # transform dolfinx mesh to pixel coordinates
    bounding_box_xy = get_xy_bounding_box(cad_mesh.geometry.x)
    
    # En coordonnées géométriques (X, Y), le coin max est [Largeur, Hauteur]
    bounding_box_img = np.array([[0.0, 0.0], [largeur, hauteur]])
    
    scaling, translation = affine_transform_bounding_boxes_2D(
        bounding_box_xy, bounding_box_img
    )

    transformation_4D = affine_transform_2D_to_4D(scaling, translation)

    # apply transformation
    pv_mesh.transform(transformation_4D, inplace=True)

    # create dummy DG0 function to paint the image
    DG0 = dolfinx.fem.functionspace(cad_mesh, ("DG", 0))
    dummy_fn = dolfinx.fem.Function(DG0)
    dummy_fn.x.array[:] = 1.0
    pv_mesh.cell_data["dummy"] = dummy_fn.x.array
    pv_mesh.set_active_scalars("dummy")

    # paint the uniform grid
    probed_grid = probe_grid.sample(pv_mesh)
    probed_grid.set_active_scalars("dummy")

    # On reshape au format standard NumPy : (Hauteur, Largeur)
    cad_ref_img = np.reshape(probed_grid.active_scalars, (hauteur, largeur))
    cad_ref_img = np.nan_to_num(cad_ref_img, nan=0.0)
    return cad_ref_img, transformation_4D



def register_imgs(
    img0: np.ndarray,
    img1: np.ndarray,
    max_angle: float = 10,
    min_scale: float = 0.6,
    max_scale: float = 0.7,
    rescale_factor: float = 1.0,
    sigma: float = 3.0,  # <-- Nouveau paramètre pour régler la force du flou
) -> np.ndarray:
    assert np.shape(img0) == np.shape(img1)

    # ----------------------------------------------------
    # ETAPE 1 : Traitement img0 (Silhouette standard)
    # ----------------------------------------------------
    threshold0 = threshold_otsu(img0)
    binary0 = np.array(img0 >= threshold0, dtype=int)
    binary0_downsampled = skimage.transform.rescale(
        (255 * binary0).astype(np.uint8), rescale_factor, anti_aliasing=False
    )

    # ----------------------------------------------------
    # ETAPE 2 : Traitement img1 (Mouchetis + Flou + Multi-Otsu)
    # ----------------------------------------------------
    # On applique le flou gaussien pour "mélanger" le mouchetis et obtenir une surface lisse
    img1_blurred = gaussian(img1, sigma=sigma, preserve_range=True)

    # On calcule le Multi-Otsu sur l'image FLOUTÉE
    thresholds1 = threshold_multiotsu(img1_blurred)
    binary1 = np.array(img1_blurred >= thresholds1[0], dtype=int)
    binary1_downsampled = skimage.transform.rescale(
        (255 * binary1).astype(np.uint8), rescale_factor, anti_aliasing=False
    )

    # --- VISUALISATION DES FILTRES ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Ligne 0 : Image source 0 et sa binarisation
    axes[0, 0].imshow(img0, cmap="gray")
    axes[0, 0].set_title("Original img0")
    axes[0, 1].axis("off")  # Case vide pour équilibrer la grille
    axes[0, 2].imshow(binary0_downsampled, cmap="gray")
    axes[0, 2].set_title(f"Binary img0 (Rescaled x{rescale_factor})")

    # Ligne 1 : Image mouchetée, étape de flou, puis binarisation finale
    axes[1, 0].imshow(img1, cmap="gray")
    axes[1, 0].set_title("Original img1 (Mouchetis)")
    axes[1, 1].imshow(img1_blurred, cmap="gray")
    axes[1, 1].set_title(f"Img1 après Flou Gaussien (sigma={sigma})")
    axes[1, 2].imshow(binary1_downsampled, cmap="gray")
    axes[1, 2].set_title("Binary img1 (Multi-Otsu sur flou)")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # ETAPE 3 : Alignement imreg_dft
    # ----------------------------------------------------
    constraints = {"angle": [0, max_angle], "scale": [min_scale, max_scale]}

    tform = imreg_dft.similarity(
        binary0_downsampled.transpose(),
        binary1_downsampled.transpose(),
        order=2,
        numiter=3,
        constraints=constraints,
    )

    # Recalcul des matrices d'origine
    trans = 1 / rescale_factor * tform["tvec"]
    scale = tform["scale"]
    angle = tform["angle"]
    theta = angle * np.pi / 180
    scale_mat = scale * np.identity(2)
    rot_mat = rotation_matrix_2D(theta)
    rot_scale_mat = rot_mat @ scale_mat
    h, w = np.shape(img0)
    img_center = np.array([w / 2, h / 2])
    rot_scale_trans = trans + (np.identity(2) - rot_scale_mat) @ img_center
    trans_4d = np.array(
        [
            [rot_scale_mat[0, 0], rot_scale_mat[0, 1], 0, rot_scale_trans[0]],
            [rot_scale_mat[1, 0], rot_scale_mat[1, 1], 0, rot_scale_trans[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    inv_trans_4d = np.linalg.inv(trans_4d)

    return inv_trans_4d

def calibrate_2d(
    mesh: dolfinx.mesh.Mesh,
    ref_img: NDArray,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
) -> NDArray:
    """Perform 2d DIC calibration.

    In 2d, no proper camera model is used. Instead, a single projection matrix is built
    that maps real world coordinates to pixel coordinates.

    Args:
    ----
        mesh (dolfinx.mesh.Mesh): CAD mesh (world coordinates)
        ref_img (NDArray): a reference image where the specimen
                              is visible without deformation
        min_scale (float): low scaling bounds for calibration
        max_scale (float): high scaling bounds for calibration

    Returns:
    -------
        A 4x4 transformation matrix in homogeneous coordinates that maps
        real world coordinates to image coordinates

    """
    (
        cad_ref_img,
        tform_cad_to_ref_4d,
    ) = construct_reference_cad_image(mesh, np.shape(ref_img))
    tform_ref_to_img_4d = register_imgs(
        cad_ref_img, ref_img, min_scale=min_scale, max_scale=max_scale
    )
    return tform_ref_to_img_4d @ tform_cad_to_ref_4d


def check_calibration_2d(
    mesh: dolfinx.mesh.Mesh, ref_img: NDArray, tform_cad_to_img_4d: NDArray
) -> None:
    """Display calibration results.

    Plot the calibrated CAD mesh next to the reference image.

    Args:
    ----
        mesh (dolfinx.mesh.Mesh): CAD mesh
        ref_img (NDArray): DIC reference image
        tform_cad_to_img_4d ((4, 4) NDArray): calibration

    """
    h, w = np.shape(ref_img)
    img_pv = pv.ImageData(dimensions=(w + 1, h + 1, 1))
    img_pv.cell_data["gray_level"] = ref_img.flatten()

    mesh_pv = dolfinx_mesh_to_pv_mesh(mesh)
    mesh_pv = mesh_pv.transform(tform_cad_to_img_4d, inplace=False)
    mesh_pv = mesh_pv.translate([0.0, 0.0, 1.0], inplace=False)

    p = pv.Plotter()
    p.add_mesh(img_pv, cmap="gray")
    p.add_mesh(mesh_pv, show_edges=True)
    p.show()

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




if __name__ == "__main__":
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, io, log, mesh
    with io.XDMFFile(MPI.COMM_WORLD, "astar_6mm.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        ref_img = skimage.io.imread("femu_files/csv_imgs/VK03-1-16-0001_0.tif", as_gray=True)
        tfo = calibrate_2d(domain, ref_img)
        print(tfo)
        check_calibration_2d(domain, ref_img, tfo)

    # Chargement du maillage et de l'image de référence
    domain = load_and_write_mesh("3specimen.msh")
    ref_image = skimage.io.imread("speckle_3specimen.png", as_gray=True)
    
    # Calcul de la calibration par pointage manuel
    tform_cad_to_img_4d = calibrate_2d_manual(domain, ref_image)
    
    print("\nMatrice de transformation finale 4x4 obtenue :")
    print(tform_cad_to_img_4d)
    
    # Visualisation 3D finale du recalage
    check_calibration_2d(domain, ref_image, tform_cad_to_img_4d)
