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
from skimage.filters import threshold_multiotsu, threshold_otsu

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


def affine_transform_bounding_boxes_2D(
    bb0: NDArray, bb1: NDArray
) -> tuple[NDArray, NDArray]:
    """Construct an affine transformation that maps bb0 to bb1.

    In the sense that the mapped bb0 is placed
    on the bottom left of bb1 while having a maximal size
    and keeping original aspect ratio.

    Bounding boxes format: NDArray [[x_min, y_min], [x_max, y_max]]

    Args:
    ----
        bb0 ((2, 2) NDArray): bounding box to transform
        bb1 (NDArray): destination bounding box

    Returns:
    -------
        tuple[NDArray, NDArray]: scaling, translation

    """
    w0 = np.linalg.norm(bb0[0, 0] - bb0[1, 0])
    h0 = np.linalg.norm(bb0[0, 1] - bb0[1, 1])

    w1 = np.linalg.norm(bb1[0, 0] - bb1[1, 0])
    h1 = np.linalg.norm(bb1[0, 1] - bb1[1, 1])

    w_ratio = w1 / w0
    h_ratio = h1 / h0

    # select transformation ratio
    resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio

    # compute translation to origin
    translation = bb1[0] - bb0[0]

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
    return cad_ref_img, transformation_4D


def register_imgs(
    img0: NDArray,
    img1: NDArray,
    max_angle: float = 10,
    min_scale: float = 0.6,
    max_scale: float = 0.7,
    rescale_factor: float = 1.0,
) -> NDArray:
    """Compute an affine transformation from img0 to img1.

    The optimized transformation includes scaling, rotation and
    translation (rigid body motion).

    Args:
    ----
        img0 (NDArray): source image
        img1 (NDArray): destination image
        max_angle (int, optional): max rotation angle. Defaults to 10.
        min_scale (float, optional): min scaling angle. Defaults to 0.6.
        max_scale (float, optional): max scaling angle. Defaults to 0.7.
        rescale_factor (float): image rescaling factor for alignment

    Returns:
    -------
        (4, 4) NDArray: 4D transformation matrix

    """
    assert np.shape(img0) == np.shape(img1)

    # make img binary for segmentation
    threshold0 = threshold_otsu(img0)
    binary0 = np.array(img0 >= threshold0, dtype=int)
    binary0_downsampled = skimage.transform.rescale(
        (255 * binary0).astype(np.uint8), rescale_factor, anti_aliasing=False
    )

    thresholds1 = threshold_multiotsu(img1)
    binary1 = np.array(img1 >= thresholds1[0], dtype=int)
    binary1_downsampled = skimage.transform.rescale(
        (255 * binary1).astype(np.uint8), rescale_factor, anti_aliasing=False
    )

    # constrain sought transform to avoid algorithm collapse
    constraints = {"angle": [0, max_angle], "scale": [min_scale, max_scale]}

    # run DFT optimizer
    # transpose images as imreg_dft uses reversed axes
    tform = imreg_dft.similarity(
        binary0_downsampled.transpose(),
        binary1_downsampled.transpose(),
        order=2,
        numiter=3,
        constraints=constraints,
    )
    print(f"{tform=}")

    trans = 1 / rescale_factor * tform["tvec"]
    scale = tform["scale"]

    # DFT optimizer angle is in degrees
    angle = tform["angle"]
    theta = angle * np.pi / 180

    # construct transformation matrices
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

    # imreg_dft constructs the inverse of the transformation
    return np.linalg.inv(trans_4d)


def calibrate_2d(
    mesh,
    ref_img,
    min_scale = 0.7,
    max_scale = 1.3,
):
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
    print("np.shape(ref_img)=", np.shape(ref_img))
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
    hauteur, largeur = np.shape(ref_img)
    
    # PyVista attend les dimensions de la grille sous la forme (NX, NY, NZ) -> (Largeur, Hauteur, 1)
    img_pv = pv.ImageData(dimensions=(largeur + 1, hauteur + 1, 1))
    img_pv.cell_data["gray_level"] = ref_img.flatten()

    mesh_pv = dolfinx_mesh_to_pv_mesh(mesh)
    mesh_pv = mesh_pv.transform(tform_cad_to_img_4d, inplace=False)
    mesh_pv = mesh_pv.translate([0.0, 0.0, 1.0], inplace=False)

    p = pv.Plotter()
    p.add_mesh(img_pv, cmap="gray")
    p.add_mesh(mesh_pv, show_edges=True)
    p.show()