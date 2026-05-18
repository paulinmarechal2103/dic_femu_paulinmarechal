"""Calibration and meshing of a specimen from VIC DIC experimental data.

Combines:
    - 2D-DIC metric calibration
    - Stereo-DIC calibration (VIC-3D .z3d project files)
    - CAD geometry calibration and alignment in camera coordinates
    - Specimen meshing utilities (pixel-to-cell mapping, interpolation matrices)

Author: Armand Touminet (armand.touminet@minesparis.psl.eu)
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import basix
import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import imreg_dft
#import numba
import numpy as np
import pyvista as pv
import scipy.optimize
import skimage
import skimage.transform
import vtk
import pandas as pd
from lxml import etree
from numpy.typing import NDArray
from petsc4py import PETSc
from scipy.spatial.transform import Rotation
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import binary_closing, disk
from dolfinx.io import XDMFFile
from mpi4py import MPI

"""
import dlf_dic
import dlf_dic.dic.p1_interpolation_jit
from dlf_stiffness.elasticity import rotation_matrix_2D
"""

if TYPE_CHECKING:
    from dlf_dic import StereoDicTrial


def csv_to_xdmf(csv_file: str, save_path: str, alpha: float = 0.2) -> None:
    """Read a Vic2D exported CSV file and export the underlying mesh to a xdmf file.

    Args:
    ----
        csv_file (str): input CSV file
        save_path (str): path to write the xdmf mesh
        alpha (float): delaunay triangulation parameter (see pv.delaunay_2d)

    """
    data = pd.read_csv(csv_file)
    names = [
        s.replace('"', "").replace(" ", "") for s in data.columns
    ]  # trim quote and whitespace

    d = {}

    for i in range(len(names)):
        d[names[i]] = data.values[:, i]

    # remove uncorrelated points from dict
    num_values = np.shape(data.values)[0]
    retained_indices = list(filter(lambda k: d["eyy"][k] != 0, np.arange(num_values)))
    for i in range(len(names)):
        d[names[i]] = d[names[i]][retained_indices]

    points = np.stack([d["x"], d["y"], 0 * d["x"]], axis=-1)
    cloud = pv.PolyData(points)
    volume = cloud.delaunay_2d(alpha=alpha)

    for n in names:
        volume[n] = d[n]

    mesh = pv.UnstructuredGrid(volume)
    mesh.save(save_path)



# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_xy_bounding_box(x: NDArray) -> NDArray:
    """Compute a 2D bounding box for a given set of points.

    Args:
    ----
        x (NDArray): set of points

    Returns:
    -------
        (2, 2) NDArray: bounding box [[x_min, y_min], [x_max, y_max]]

    """
    return np.array(
        [[np.min(x[:, 0]), np.min(x[:, 1])],
         [np.max(x[:, 0]), np.max(x[:, 1])]]
    )


def get_xyz_bounding_box(x: NDArray) -> NDArray:
    """Compute a 3D bounding box for a given set of points.

    Args:
    ----
        x (NDArray): set of points

    Returns:
    -------
        (2, 3) NDArray: bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]

    """
    return np.array(
        [[np.min(x[:, 0]), np.min(x[:, 1]), np.min(x[:, 2])],
         [np.max(x[:, 0]), np.max(x[:, 1]), np.max(x[:, 2])]]
    )


# ---------------------------------------------------------------------------
# Affine transform helpers
# ---------------------------------------------------------------------------

def affine_transform_bounding_boxes_2D(
    bb0: NDArray, bb1: NDArray
) -> tuple[NDArray, NDArray]:
    """Construct an affine transformation that maps bb0 into bb1.

    The mapped bb0 is placed on the bottom-left of bb1 with maximum size
    while preserving the original aspect ratio.

    Bounding-box format: NDArray [[x_min, y_min], [x_max, y_max]]

    Args:
    ----
        bb0 ((2, 2) NDArray): bounding box to transform
        bb1 ((2, 2) NDArray): destination bounding box

    Returns:
    -------
        tuple[NDArray, NDArray]: (scaling, translation)

    """
    w0 = np.linalg.norm(bb0[0, 0] - bb0[1, 0])
    h0 = np.linalg.norm(bb0[0, 1] - bb0[1, 1])
    w1 = np.linalg.norm(bb1[0, 0] - bb1[1, 0])
    h1 = np.linalg.norm(bb1[0, 1] - bb1[1, 1])

    w_ratio = w1 / w0
    h_ratio = h1 / h0
    resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio
    translation = bb1[0] - bb0[0]
    return np.array([resize_ratio, resize_ratio]), translation


def affine_transform_2D_to_4D(scaling: NDArray, translation: NDArray) -> NDArray:
    """Create a homogeneous 4x4 transformation matrix from 2D components.

    Args:
    ----
        scaling ((2,) NDArray): x and y scaling factors
        translation ((2,) NDArray): x and y translation

    Returns:
    -------
        (4, 4) NDArray

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
    """Convert a 4x4 homogeneous transform to a 3x3 affine transform.

    Args:
    ----
        transform_4D ((4, 4) NDArray): homogeneous affine transformation

    Returns:
    -------
        (3, 3) NDArray

    """
    transform_3D = np.zeros((3, 3), dtype=transform_4D.dtype)
    transform_3D[:2, :2] = transform_4D[:2, :2]
    transform_3D[:2, 2] = transform_4D[:2, 3]
    transform_3D[2, :2] = transform_4D[3, :2]
    transform_3D[2, 2] = transform_4D[3, 3]
    return transform_3D


def transform_3D_to_4D(transform_3D: NDArray) -> NDArray:
    """Convert a 3x3 affine transform to a 4x4 homogeneous transform.

    Args:
    ----
        transform_3D ((3, 3) NDArray): affine transformation

    Returns:
    -------
        (4, 4) NDArray

    """
    transform_4D = np.zeros((4, 4), dtype=transform_3D.dtype)
    transform_4D[:2, :2] = transform_3D[:2, :2]
    transform_4D[:2, 3] = transform_3D[:2, 2]
    transform_4D[3, :2] = transform_3D[2, :2]
    transform_4D[3, 3] = transform_3D[2, 2]
    transform_4D[2, 2] = 1.0
    return transform_4D


def rotation_matrix_euler(alpha: float, beta: float, gamma: float) -> NDArray:
    """Construct a rotation matrix from three Euler angles (extrinsics convention).

    Rotation sequence (axes fixed):
        1) gamma around Z
        2) beta around Y
        3) alpha around X

    Args:
    ----
        alpha (float): rotation around X axis (degrees)
        beta (float): rotation around Y axis (degrees)
        gamma (float): rotation around Z axis (degrees)

    Returns:
    -------
        (3, 3) NDArray

    """
    return Rotation.from_euler("ZYX", [alpha, beta, gamma], degrees=True).as_matrix()


def vtk_to_numpy(vtk_mat: vtk.vtkMatrix4x4) -> NDArray:
    """Convert a VTK 4x4 matrix to a numpy array."""
    np_mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            np_mat[i, j] = vtk_mat.GetElement(i, j)
    return np_mat


# ---------------------------------------------------------------------------
# Mesh / image conversion helpers
# ---------------------------------------------------------------------------

def dolfinx_mesh_to_pv_mesh(dolfinx_mesh: dolfinx.mesh.Mesh) -> pv.UnstructuredGrid:
    """Convert a dolfinx mesh to a pyvista UnstructuredGrid.

    Args:
    ----
        dolfinx_mesh (dolfinx.mesh.Mesh): input mesh

    Returns:
    -------
        pv.UnstructuredGrid

    """
    topology, cell_types, x = dolfinx.plot.vtk_mesh(
        dolfinx_mesh, dim=dolfinx_mesh.topology.dim
    )
    return pv.UnstructuredGrid(topology, cell_types, x)


def img_uniform_grid(img_shape: tuple[int, int]) -> pv.ImageData:
    """Construct a uniform pyvista grid that represents an image.

    The reconstructed image can be accessed via `grid.active_scalars`.

    Args:
    ----
        img_shape (tuple[int, int]): (width, height) of the target image

    Returns:
    -------
        pv.ImageData

    """
    w, h = img_shape
    return pv.ImageData(dimensions=(w, h, 1), spacing=(1, 1, 1), origin=(0, 0, 0))


def apply_mesh_transform(
    mesh: pv.UnstructuredGrid, tform: NDArray
) -> pv.UnstructuredGrid:
    """Apply a rigid body transform [Tx, Ty, Tz, Rx, Ry, Rz] to a pyvista mesh.

    Args:
    ----
        mesh (pv.UnstructuredGrid): mesh to transform
        tform ((6,) NDArray): [Tx, Ty, Tz, Rx, Ry, Rz]

    Returns:
    -------
        pv.UnstructuredGrid: transformed mesh (copy)

    """
    assert np.shape(tform) == (6,)
    *T, Rx, Ry, Rz = tform
    mesh = mesh.translate(T, inplace=False)
    mesh = mesh.rotate_x(Rx, inplace=False)
    mesh = mesh.rotate_y(Ry, inplace=False)
    return mesh.rotate_z(Rz, inplace=False)


# ---------------------------------------------------------------------------
# 2D DIC calibration
# ---------------------------------------------------------------------------

def construct_reference_cad_image(
    cad_mesh: dolfinx.mesh.Mesh, ref_img_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """Construct a reference CAD image for 2D DIC alignment.

    The CAD mesh is rendered in a canonical position (bottom-left of the image)
    to serve as the source image for a DFT-based registration against the real
    DIC reference image.

    Args:
    ----
        cad_mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates
        ref_img_shape (tuple[int, int]): (width, height) of the target image

    Returns:
    -------
        tuple[NDArray, NDArray]:
            - cad_ref_img: rendered grey-level image of the CAD mesh
            - transformation_4D: (4, 4) matrix mapping CAD coords to pixel coords

    """
    pv_mesh = dolfinx_mesh_to_pv_mesh(cad_mesh)
    w, h = ref_img_shape
    probe_grid = img_uniform_grid((h, w))

    bounding_box_xy = get_xy_bounding_box(cad_mesh.geometry.x)
    bounding_box_img = np.array([[0.0, 0.0], [h, w]])
    scaling, translation = affine_transform_bounding_boxes_2D(
        bounding_box_xy, bounding_box_img
    )
    transformation_4D = affine_transform_2D_to_4D(scaling, translation)
    pv_mesh.transform(transformation_4D, inplace=True)

    DG0 = dolfinx.fem.functionspace(cad_mesh, ("DG", 0))
    dummy_fn = dolfinx.fem.Function(DG0)
    dummy_fn.interpolate(lambda x: np.full(x.shape[1], 1.0))
    pv_mesh.cell_data["dummy"] = dummy_fn.x.array
    pv_mesh.set_active_scalars("dummy")

    probed_grid = probe_grid.sample(pv_mesh)
    probed_grid.set_active_scalars("dummy")
    cad_ref_img = np.reshape(probed_grid.active_scalars, (w, h))
    return cad_ref_img, transformation_4D

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

def register_imgs(
    img0: NDArray,
    img1: NDArray,
    max_angle: float = 10,
    min_scale: float = 0.6,
    max_scale: float = 0.7,
    rescale_factor: float = 1.0,
) -> NDArray:
    """Compute an affine transformation that maps img0 onto img1.

    The optimised transform includes scaling, rotation and translation.
    Uses a DFT-based (imreg_dft) similarity registration on binarised images.

    Args:
    ----
        img0 (NDArray): source image (CAD reference)
        img1 (NDArray): destination image (DIC reference)
        max_angle (float): maximum allowed rotation angle in degrees
        min_scale (float): minimum allowed scaling factor
        max_scale (float): maximum allowed scaling factor
        rescale_factor (float): downsampling factor applied before registration

    Returns:
    -------
        (4, 4) NDArray: homogeneous transformation matrix

    """
    assert np.shape(img0) == np.shape(img1)

    threshold0 = threshold_otsu(img0)
    binary0 = np.array(img0 >= threshold0, dtype=int)
    binary0_ds = skimage.transform.rescale(
        (255 * binary0).astype(np.uint8), rescale_factor, anti_aliasing=False
    )

    thresholds1 = threshold_multiotsu(img1)
    binary1 = np.array(img1 >= thresholds1[0], dtype=int)
    binary1_ds = skimage.transform.rescale(
        (255 * binary1).astype(np.uint8), rescale_factor, anti_aliasing=False
    )

    constraints = {"angle": [0, max_angle], "scale": [min_scale, max_scale]}
    tform = imreg_dft.similarity(
        binary0_ds.transpose(),
        binary1_ds.transpose(),
        order=2,
        numiter=3,
        constraints=constraints,
    )
    print(f"{tform=}")

    trans = 1 / rescale_factor * tform["tvec"]
    scale = tform["scale"]
    angle = tform["angle"]
    theta = angle * np.pi / 180

    scale_mat = scale * np.identity(2)
    rot_mat = rotation_matrix_2D(theta)
    rot_scale_mat = rot_mat @ scale_mat

    w, h = np.shape(img0)
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
    # imreg_dft returns the inverse transform
    return np.linalg.inv(trans_4d)


def calibrate_2d(
    mesh: dolfinx.mesh.Mesh,
    ref_img: NDArray,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
) -> NDArray:
    """Perform 2D DIC calibration.

    Builds a single projection matrix that maps real-world coordinates to
    pixel coordinates (no full camera model is used in 2D).

    Args:
    ----
        mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates
        ref_img (NDArray): reference DIC image (specimen at rest)
        min_scale (float): lower bound for scale search
        max_scale (float): upper bound for scale search

    Returns:
    -------
        (4, 4) NDArray: homogeneous projection from world to image coordinates

    """
    cad_ref_img, tform_cad_to_ref_4d = construct_reference_cad_image(
        mesh, np.shape(ref_img)
    )
    tform_ref_to_img_4d = register_imgs(
        cad_ref_img, ref_img, min_scale=min_scale, max_scale=max_scale
    )
    return tform_ref_to_img_4d @ tform_cad_to_ref_4d


def load_xdmf(
    filename: str, comm: MPI.Comm = MPI.COMM_SELF, grid_name: str = "Grid"
) -> dolfinx.mesh.Mesh:
    """Read a mesh from a XDMF file.

    Args:
    ----
        filename (str): file to read
        comm (MPI communicator): how to distribute read data
        grid_name (str): name of the mesh to read as present in the XDMF header

    Returns:
    -------
        dolfinx.mesh.Mesh

    """
    with XDMFFile(comm, filename, "r") as file:
        return file.read_mesh(name=grid_name)

def calibrate_2d_cached(
    mesh_path: str,
    ref_img_path: str,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    rescale_factor: float = 1.0,
) -> NDArray:
    """Perform 2D DIC calibration (disk-cached version).

    Args:
    ----
        mesh_path (str): path to the CAD mesh (.xdmf)
        ref_img_path (str): path to the DIC reference image
        min_scale (float): lower bound for scale search
        max_scale (float): upper bound for scale search
        rescale_factor (float): downsampling factor to speed up registration

    Returns:
    -------
        (4, 4) NDArray: homogeneous projection from world to image coordinates

    """
    mesh = load_xdmf(mesh_path)
    ref_img = skimage.io.imread(ref_img_path, as_gray=True)

    if len(ref_img_shape := np.shape(ref_img)) != 2:
        raise ValueError("Reference image must be 2-dimensional.")

    cad_ref_img, tform_cad_to_ref_4d = construct_reference_cad_image(
        mesh, ref_img_shape
    )
    tform_ref_to_img_4d = register_imgs(
        cad_ref_img,
        ref_img,
        min_scale=min_scale,
        max_scale=max_scale,
        rescale_factor=rescale_factor,
    )
    return tform_ref_to_img_4d @ tform_cad_to_ref_4d


def check_calibration_2d(
    mesh: dolfinx.mesh.Mesh, ref_img: NDArray, tform_cad_to_img_4d: NDArray
) -> None:
    """Visually verify a 2D DIC calibration result.

    Overlays the projected CAD mesh on the DIC reference image.

    Args:
    ----
        mesh (dolfinx.mesh.Mesh): CAD mesh
        ref_img (NDArray): DIC reference image
        tform_cad_to_img_4d ((4, 4) NDArray): calibration matrix

    """
    w, h = np.shape(ref_img)
    img_pv = pv.ImageData(dimensions=(w + 1, h + 1, 1))
    img_pv.cell_data["gray_level"] = ref_img.flatten()

    mesh_pv = dolfinx_mesh_to_pv_mesh(mesh)
    mesh_pv = mesh_pv.transform(tform_cad_to_img_4d, inplace=False)
    mesh_pv = mesh_pv.translate([0.0, 0.0, 1.0], inplace=False)

    p = pv.Plotter()
    p.add_mesh(img_pv, cmap="gray")
    p.add_mesh(mesh_pv, show_edges=True)
    p.show()


# ---------------------------------------------------------------------------
# Camera model helpers
# ---------------------------------------------------------------------------

def pinhole_intrinsics_matrix(
    center_x: float,
    center_y: float,
    focal_x: float,
    focal_y: float,
    skew: float,
) -> NDArray:
    """Construct the intrinsics matrix of a pinhole camera.

    Args:
    ----
        center_x (float): principal point x coordinate (pixels)
        center_y (float): principal point y coordinate (pixels)
        focal_x (float): focal length along x (pixels)
        focal_y (float): focal length along y (pixels)
        skew (float): skew coefficient

    Returns:
    -------
        (3, 4) NDArray: homogeneous intrinsics matrix

    """
    return np.array(
        [
            [focal_x, skew, center_x, 0.0],
            [0, focal_y, center_y, 0.0],
            [0, 0, 1.0, 0.0],
        ]
    )


def extrinsics_matrix(rot: NDArray, trans: NDArray) -> NDArray:
    """Construct the homogeneous extrinsics matrix of a pinhole camera.

    P = [ R  T ]
        [ 0  1 ]

    Args:
    ----
        rot ((3, 3) NDArray): rotation matrix (world → camera)
        trans ((3,) NDArray): translation vector (world → camera)

    Returns:
    -------
        (4, 4) NDArray: homogeneous extrinsics matrix

    """
    assert np.shape(rot) == (3, 3)
    assert np.shape(np.squeeze(trans)) == (3,)
    trans = np.reshape(trans, (3, 1))
    return np.block([[rot, trans], 3 * [0.0] + [1.0]])


def principal_point_image_center(
    img_width: int, img_height: int, center_x: float, center_y: float
) -> tuple[float, float]:
    """Convert the camera principal point to normalised image-centre coordinates.

    Args:
    ----
        img_width (int): image width in pixels
        img_height (int): image height in pixels
        center_x (float): principal point x coordinate
        center_y (float): principal point y coordinate

    Returns:
    -------
        (img_cx, img_cy): normalised window-centre coordinates

    """
    img_cx = -2 * (center_x - float(img_width) / 2) / img_width
    img_cy = -2 * (center_y - float(img_height) / 2) / img_height
    return img_cx, img_cy


def create_vtk_camera(intrinsics: NDArray, img_shape: tuple[int, int]) -> pv.Camera:
    """Create a pyvista/VTK camera from Vic3D intrinsic parameters.

    Args:
    ----
        intrinsics ((5,) NDArray): [cx, cy, fx, fy, skew]
        img_shape (tuple[int, int]): (width, height) in pixels

    Returns:
    -------
        pv.Camera

    """
    assert np.shape(intrinsics) == (5,)
    w, h = img_shape
    cx, cy, fx, fy = intrinsics[:4]

    camera = pv.Camera()
    wcx = -2 * (cx - float(w) / 2) / w
    wcy = 2 * (cy - float(h) / 2) / h
    camera.SetWindowCenter(wcx, wcy)
    view_angle = 180 / math.pi * (2.0 * math.atan2(h / 2.0, fy))
    camera.SetViewAngle(view_angle)
    return camera


def orient_vtk_camera(camera: pv.Camera, orientation: NDArray) -> pv.Camera:
    """Orient a pyvista camera according to Vic3D extrinsic parameters.

    Args:
    ----
        camera (pv.Camera): camera to orient
        orientation ((6,) NDArray): [Tx, Ty, Tz, alpha, beta, gamma]

    Returns:
    -------
        pv.Camera: oriented camera

    """
    assert np.shape(orientation) == (6,)
    camera.SetPosition(orientation[3], orientation[4], 0.0)
    camera.SetFocalPoint(orientation[3], orientation[4], orientation[5])
    camera.azimuth = orientation[1]
    camera.roll = orientation[0]
    camera.elevation = orientation[2]
    camera.SetViewUp(0, 1, 0)
    camera.clipping_range = (1, 850)
    return camera


def camera_projection_matrix(camera: pv.Camera, img_shape: tuple[int, int]) -> NDArray:
    """Construct the world-to-pixel projection matrix of a VTK camera.

    Args:
    ----
        camera (pv.Camera): pyvista camera
        img_shape (tuple[int, int]): (width, height) in pixels

    Returns:
    -------
        (3, 4) NDArray: homogeneous projection matrix

    """
    w, h = img_shape
    M = np.array(
        [[w / 2, 0.0, 0.0, w / 2],
         [0, h / 2, 0, h / 2],
         [0.0, 0.0, 0.0, 1.0]]
    )
    C = camera.GetCompositeProjectionTransformMatrix(
        w / h, camera.clipping_range[0], camera.clipping_range[1]
    )
    return M @ vtk_to_numpy(C)


def kinematic_sensitivity(projection_matrix: NDArray) -> Callable[[NDArray], NDArray]:
    """Return the kinematic sensitivity operator (Jacobian of the projection).

    Computes the Jacobian of  Π: (x, y, z) → (u, v)  where
        (s·u, s·v, s) = P · (x, y, z, 1)ᵀ

    Args:
    ----
        projection_matrix ((3, 4) NDArray): world-to-camera projection matrix

    Returns:
    -------
        Callable mapping a (3,) point to a (2, 3) Jacobian matrix

    """
    def _eval(x: NDArray) -> NDArray:
        z = projection_matrix @ x
        s = z[2]
        P23 = projection_matrix[:2, :3]
        P3 = projection_matrix[2, :3]
        return 1 / s * (P23 - z / s @ P3.T)

    return _eval


# ---------------------------------------------------------------------------
# Stereo-DIC calibration (Vic3D)
# ---------------------------------------------------------------------------

class Vic3dStereoCalibration:
    """Stereo calibration loaded from a Vic3D project."""

    def __init__(
        self,
        intrinsics_1: NDArray,
        distortion_1: NDArray,
        orientation_1: NDArray,
        intrinsics_2: NDArray,
        distortion_2: NDArray,
        orientation_2: NDArray,
        pixel_size: float,
    ) -> None:
        """Instantiate from Vic3D exported calibration arrays.

        Args:
        ----
            intrinsics_1 ((5,) NDArray): [cx, cy, fx, fy, skew] for camera 1
            distortion_1 ((11,) NDArray): distortion coefficients for camera 1
            orientation_1 ((6,) NDArray): [alpha, beta, gamma, Tx, Ty, Tz] for camera 1
            intrinsics_2 ((5,) NDArray): same for camera 2
            distortion_2 ((11,) NDArray): same for camera 2
            orientation_2 ((6,) NDArray): same for camera 2
            pixel_size (float): physical size of one pixel on the specimen (mm/px)

        """
        assert np.shape(intrinsics_1) == (5,)
        assert np.shape(distortion_1) == (11,)
        assert np.shape(orientation_1) == (6,)
        assert np.shape(intrinsics_2) == (5,)
        assert np.shape(distortion_2) == (11,)
        assert np.shape(orientation_2) == (6,)

        self.intrinsics_1 = intrinsics_1
        self.distortion_1 = distortion_1
        self.orientation_1 = orientation_1
        self.intrinsics_2 = intrinsics_2
        self.distortion_2 = distortion_2
        self.orientation_2 = orientation_2
        self.pixel_size = pixel_size

    def construct_cameras(
        self, img_shape: tuple[int, int]
    ) -> tuple[pv.Camera, pv.Camera]:
        """Build VTK cameras for this calibration.

        Args:
        ----
            img_shape (tuple[int, int]): (width, height) in pixels

        Returns:
        -------
            (camera_1, camera_2): pair of oriented pv.Camera objects

        """
        camera_1 = orient_vtk_camera(
            create_vtk_camera(self.intrinsics_1, img_shape), self.orientation_1
        )
        camera_2 = orient_vtk_camera(
            create_vtk_camera(self.intrinsics_2, img_shape), self.orientation_2
        )
        return camera_1, camera_2

    def compute_projection_matrices(
        self, img_shape: tuple[int, int]
    ) -> tuple[NDArray, NDArray]:
        """Compute the two world-to-pixel projection matrices.

        Args:
        ----
            img_shape (tuple[int, int]): (width, height) in pixels

        Returns:
        -------
            (P1, P2): pair of (3, 4) projection matrices

        """
        camera_1, camera_2 = self.construct_cameras(img_shape)
        return (
            camera_projection_matrix(camera_1, img_shape),
            camera_projection_matrix(camera_2, img_shape),
        )

    def compute_specimen_depth(self) -> float:
        """Estimate specimen depth from the pinhole formula (d = f / pixel_size)."""
        return self.intrinsics_1[2] / self.pixel_size


def load_calibration_from_z3d(
    project_file_path: str, pixel_size: float
) -> Vic3dStereoCalibration:
    """Load a stereo-DIC calibration from a Vic3D project file (.z3d).

    Args:
    ----
        project_file_path (str): path to the .z3d archive
        pixel_size (float): physical pixel size on the specimen (mm/px).
                            Must be measured manually.

    Returns:
    -------
        Vic3dStereoCalibration

    """
    with zipfile.ZipFile(project_file_path) as zf, zf.open("project.xml") as xml_file:
        xml = xml_file.read()

    root = etree.fromstring(xml)
    cameras = root.find("calibration").findall("camera")

    if len(cameras) != 2:
        raise RuntimeError(
            f"Expected 2 camera entries in {project_file_path}, found {len(cameras)}."
        )

    camera_1, camera_2 = cameras
    assert camera_1.attrib["id"] == "0"
    assert camera_2.attrib["id"] == "1"

    def _parse(s: str) -> NDArray:
        return np.array([float(v) for v in s.split()])

    return Vic3dStereoCalibration(
        intrinsics_1=_parse(camera_1.find("intrinsics").text),
        distortion_1=_parse(camera_1.find("distortion").text),
        orientation_1=_parse(camera_1.find("orientation").text),
        intrinsics_2=_parse(camera_2.find("intrinsics").text),
        distortion_2=_parse(camera_2.find("distortion").text),
        orientation_2=_parse(camera_2.find("orientation").text),
        pixel_size=pixel_size,
    )


# ---------------------------------------------------------------------------
# Stereo-DIC scene and CAD mesh alignment
# ---------------------------------------------------------------------------

class StereoDICScene:
    """Represent a stereo-DIC calibration as a VTK rendering scene."""

    def __init__(
        self,
        camera_1: pv.Camera,
        camera_2: pv.Camera,
        mesh: pv.UnstructuredGrid,
    ) -> None:
        """Construct a StereoDICScene from two cameras and a mesh."""
        self.camera_1 = camera_1
        self.camera_2 = camera_2
        self.mesh = mesh

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def capture_scene(
        self,
        mesh: pv.UnstructuredGrid,
        img_shape: tuple[int, int],
        show_edges: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Render the scene from both cameras and return grey-level images.

        Args:
        ----
            mesh (pv.UnstructuredGrid): mesh to render
            img_shape (tuple[int, int]): output image shape
            show_edges (bool): render mesh edges instead of filled faces

        Returns:
        -------
            (frame_1, frame_2): grey-level images for camera 1 and 2

        """
        frames = []
        for cam in (self.camera_1, self.camera_2):
            pl = pv.Plotter(off_screen=True, window_size=img_shape)
            pl.background_color = "black"
            pl.camera = cam.copy()
            if show_edges:
                pl.add_mesh(mesh, color="white", style="wireframe", line_width=3.0)
            else:
                pl.add_mesh(mesh, color="white")
            pl.show()
            pl.close()
            frames.append(np.linalg.norm(pl.last_image, axis=2))
        return frames[0], frames[1]

    def plot_scene(self, *meshes: pv.UnstructuredGrid, **pv_kwargs) -> None:
        """Interactive pyvista plot showing cameras and supplied meshes."""
        plotter = pv.Plotter()
        plotter.background_color = "white"
        for cam, label, color in zip(
            (self.camera_1, self.camera_2),
            ("Camera 1", "Camera 2"),
            ("blue", "red"),
        ):
            frustum = cam.view_frustum(1.0)
            plotter.add_mesh(
                frustum, style="wireframe", color=color, ambient=1.0, line_width=3.0
            )
            plotter.add_point_labels(
                [cam.position], [label],
                margin=0, fill_shape=False, font_size=14,
                shape_color="white", point_color="red", text_color="black",
            )
        for mesh in meshes:
            plotter.add_mesh(mesh, **pv_kwargs)
        plotter.show()

    # ------------------------------------------------------------------
    # Mesh transform
    # ------------------------------------------------------------------

    def transform_mesh(self, transform: NDArray) -> None:
        """Apply a rigid body transform [Tx, Ty, Tz, Rx, Ry, Rz] to self.mesh in place.

        Args:
        ----
            transform ((6,) NDArray): [Tx, Ty, Tz, Rx, Ry, Rz]

        """
        assert np.shape(transform)[0] == 6
        self.mesh = apply_mesh_transform(self.mesh, transform)

    def optimize_mesh_transform(
        self, initial_guess: NDArray, img_1: NDArray, img_2: NDArray
    ) -> NDArray:
        """Find the rigid transform that best aligns the CAD mesh with both camera images.

        Minimises a binary-mask overlap loss using Powell's method.

        Args:
        ----
            initial_guess ((6,) NDArray): starting point [Tx, Ty, Tz, Rx, Ry, Rz]
            img_1 (NDArray): grey-level image from camera 1
            img_2 (NDArray): grey-level image from camera 2

        Returns:
        -------
            (6,) NDArray: optimal transform [Tx, Ty, Tz, Rx, Ry, Rz]

        """
        thresholds_1 = threshold_multiotsu(img_1)
        binary_1 = binary_closing(
            np.array(img_1 >= thresholds_1[0], dtype=int), disk(7)
        )
        thresholds_2 = threshold_multiotsu(img_2)
        binary_2 = binary_closing(
            np.array(img_2 >= thresholds_2[0], dtype=int), disk(7)
        )

        def _loss(transform: NDArray) -> float:
            tform = initial_guess + transform
            transformed_mesh = apply_mesh_transform(self.mesh, tform)
            frame_1, frame_2 = self.capture_scene(transformed_mesh, np.shape(img_1))

            bframe_1 = np.array(frame_1 > threshold_otsu(frame_1), dtype=int)
            bframe_2 = np.array(frame_2 > threshold_otsu(frame_2), dtype=int)

            loss_val = 0.5 * (
                np.mean((binary_1 - bframe_1) ** 2)
                + np.mean((binary_2 - bframe_2) ** 2)
            )
            print(f"{loss_val=}, {transform=}")
            return loss_val

        x_opt = scipy.optimize.minimize(
            _loss,
            np.array([0, 0, 0, 0, 0, 0.0]),
            method="powell",
            constraints=[
                (None, None), (None, None), (None, None),
                (-10, 10), (-10, 10), (-10, 10),
            ],
            tol=1e-2,
            options={"maxiter": 40},
        )
        tform = x_opt.x + initial_guess
        print(f"optimal transform: {tform}")
        return tform


def optimize_mesh_transform_cached(
    z3d_file: str,
    img_0_path: str,
    img_1_path: str,
    cad_mesh_file: str,
    img_shape: tuple[int, int],
    pixel_size: float,
) -> NDArray:
    """Compute the CAD mesh spatial transform to align with a stereo-DIC calibration.

    Args:
    ----
        z3d_file (str): path to the Vic3D .z3d project file
        img_0_path (str): path to the reference image of camera 0
        img_1_path (str): path to the reference image of camera 1
        cad_mesh_file (str): path to the CAD mesh (.xdmf)
        img_shape (tuple[int, int]): (width, height) of the camera images
        pixel_size (float): physical pixel size on the specimen (mm/px)

    Returns:
    -------
        (6,) NDArray: optimal transform [Tx, Ty, Tz, Rx, Ry, Rz]

    """
    assert np.shape(img_shape)[0] == 2
    stereo_cal = load_calibration_from_z3d(z3d_file, pixel_size)
    img_0 = skimage.io.imread(img_0_path, as_gray=True)
    img_1 = skimage.io.imread(img_1_path, as_gray=True)
    cameras = stereo_cal.construct_cameras(img_shape)
    cad_mesh = dlf_dic.load_xdmf(cad_mesh_file)
    cad_mesh_pv = dolfinx_mesh_to_pv_mesh(cad_mesh)
    scene = StereoDICScene(*cameras, cad_mesh_pv)
    initial_depth_guess = stereo_cal.compute_specimen_depth()
    return scene.optimize_mesh_transform(
        np.array([0.0, 0, initial_depth_guess, 0, 0, 0.0]),
        img_0,
        img_1,
    )


# ---------------------------------------------------------------------------
# Pixel-to-cell mapping (meshing helpers)
# ---------------------------------------------------------------------------

def construct_pixel_to_cell_mapping(
    cad_mesh: dolfinx.mesh.Mesh,
    img_shape: tuple[int, int],
    tform_cad_to_img_4d: NDArray,
    displacement_shape: int = 2,
) -> NDArray:
    """Map each image pixel to the mesh cell it falls inside (2D / affine projection).

    Pixels outside the mesh are assigned index -1.

    Args:
    ----
        cad_mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates
        img_shape (tuple[int, int]): (width, height) of the image
        tform_cad_to_img_4d ((4, 4) NDArray): homogeneous world-to-pixel transform
        displacement_shape (int): number of displacement components

    Returns:
    -------
        (N,) NDArray[int]: cell index for each pixel (N = 2 * width * height)

    """
    img_mesh = dolfinx_mesh_to_pv_mesh(cad_mesh)
    img_mesh.transform(tform_cad_to_img_4d, inplace=True)

    probe_grid = img_uniform_grid(img_shape)
    DG0 = dolfinx.fem.functionspace(cad_mesh, ("DG", 0, (displacement_shape,)))
    num_cells = DG0.tabulate_dof_coordinates().shape[0]
    iota = np.arange(1, num_cells + 1, dtype=int)
    img_mesh.cell_data["idx"] = np.stack(
        [iota for _ in range(displacement_shape)], axis=1
    )

    idx_img = probe_grid.sample(img_mesh)
    cell_idx_vector = idx_img["idx"][:, :2].flatten() - 1
    return cell_idx_vector


def construct_pixel_to_cell_mapping_stereo(
    cad_mesh: dolfinx.mesh.Mesh,
    img_shape: tuple[int, int],
    P: NDArray,
    displacement_shape: int = 2,
) -> NDArray:
    """Map each image pixel to the mesh cell it falls inside (stereo projection).

    Uses the full perspective projection matrix P instead of an affine transform.
    Pixels outside the mesh are assigned index -1.

    Args:
    ----
        cad_mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates
        img_shape (tuple[int, int]): (width, height) of the image
        P ((3, 4) NDArray): homogeneous perspective projection matrix
        displacement_shape (int): number of displacement components

    Returns:
    -------
        (N,) NDArray[int]: cell index for each pixel

    """
    img_mesh = dolfinx_mesh_to_pv_mesh(cad_mesh)

    x = img_mesh.points.T
    x4 = np.vstack([x, np.ones_like(x[0])])
    y = P @ x4
    y /= y[2]
    y[2] = 0.0
    img_mesh.points = y.T

    probe_grid = img_uniform_grid(img_shape)
    DG0 = dolfinx.fem.functionspace(cad_mesh, ("DG", 0, (displacement_shape,)))
    num_cells = DG0.tabulate_dof_coordinates().shape[0]
    iota = np.arange(1, num_cells + 1, dtype=int)
    img_mesh.cell_data["idx"] = np.stack(
        [iota for _ in range(displacement_shape)], axis=1
    )

    idx_img = probe_grid.sample(img_mesh)
    cell_idx_vector = idx_img["idx"][:, :2].flatten() - 1
    return cell_idx_vector


# ---------------------------------------------------------------------------
# Interpolation matrices (mesh → image)
# ---------------------------------------------------------------------------

def img_mesh_to_img_interpolation_matrix_optimized_triangle(
    cad_mesh: dolfinx.mesh.Mesh,
    img_shape: tuple[int, int],
    tform_cad_to_img_4d: NDArray,
    displacement_shape: int = 2,
) -> PETSc.Mat:
    """Build the sparse mesh-to-image interpolation matrix (triangle P1, 2D affine).

    JIT-compiled (numba) implementation.

    Args:
    ----
        cad_mesh (dolfinx.mesh.Mesh): CAD mesh (triangle cells)
        img_shape (tuple[int, int]): (width, height) of the image
        tform_cad_to_img_4d ((4, 4) NDArray): world-to-pixel transform
        displacement_shape (int): number of displacement components

    Returns:
    -------
        PETSc.Mat: sparse interpolation matrix

    """
    assert cad_mesh.topology.cell_types[0] == dolfinx.mesh.CellType.triangle

    tform_img_to_cad_4D = np.linalg.inv(tform_cad_to_img_4d)
    cell_idx_vector = construct_pixel_to_cell_mapping(
        cad_mesh, img_shape, tform_cad_to_img_4d
    )

    V = dolfinx.fem.functionspace(cad_mesh, ("CG", 1, (displacement_shape,)))
    num_cells, *_ = np.shape(cad_mesh.geometry.dofmap)
    cell_to_dofs = np.array([V.dofmap.cell_dofs(i) for i in range(num_cells)])
    dofs_coords = V.tabulate_dof_coordinates()
    w, h = img_shape

    @numba.njit(fastmath=True)
    def _assembly_loop() -> tuple[NDArray, NDArray, NDArray]:
        row_ptr = np.zeros(len(cell_idx_vector) + 1, dtype=np.int64)
        nnz = 3 * np.shape(cell_idx_vector[cell_idx_vector != -1])[0]
        col_ind = np.zeros(nnz, dtype=np.int64)
        buffer = np.zeros(nnz, dtype=np.double)
        cur_idx = 0
        for i in range(len(cell_idx_vector)):
            if cell_idx_vector[i] == -1:
                row_ptr[i + 1] = row_ptr[i]
                continue
            row_ptr[i + 1] = 3 + row_ptr[i]
            dof_indices = cell_to_dofs[cell_idx_vector[i]]
            triangle_nodes = dofs_coords[dof_indices]
            x_img = np.array([(i // 2) % w, (i // 2) // h, 0, 1], dtype=np.float64)
            x_cad = tform_img_to_cad_4D @ x_img
            x_cad = x_cad[:displacement_shape]
            basis_values = dlf_dic.dic.p1_interpolation_jit.tabulate_P1_basis_triangle(
                x_cad[:2], np.ascontiguousarray(triangle_nodes[:, :2])
            )
            buffer[cur_idx: cur_idx + 3] = basis_values
            col_ind[cur_idx: cur_idx + 3] = (
                displacement_shape * dof_indices + i % displacement_shape
            )
            cur_idx += 3
        row_ptr[-1] = nnz
        return row_ptr, col_ind, buffer

    print("[info] constructing interpolation matrix...")
    row_ptr, col_ind, buffer = _assembly_loop()
    print("[info] done.")

    row_ptr = np.array(row_ptr, dtype=np.int32)
    col_ind = np.array(col_ind, dtype=np.int32)
    mat = PETSc.Mat().createAIJWithArrays(
        size=(len(cell_idx_vector), V.tabulate_dof_coordinates().shape[0] * displacement_shape),
        csr=(row_ptr, col_ind, buffer),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    return mat


def img_mesh_to_img_interpolation_matrix_optimized_triangle_stereo(
    cad_mesh: dolfinx.mesh.Mesh,
    img_shape: tuple[int, int],
    projection_matrix: NDArray,
    displacement_shape: int = 3,
) -> PETSc.Mat:
    """Build the sparse mesh-to-image interpolation matrix (triangle P1, stereo).

    JIT-compiled (numba) implementation using the full perspective projection.

    Args:
    ----
        cad_mesh (dolfinx.mesh.Mesh): CAD mesh (triangle cells)
        img_shape (tuple[int, int]): (width, height) of the image
        projection_matrix ((3, 4) NDArray): world-to-pixel perspective projection
        displacement_shape (int): number of displacement components (3 for stereo)

    Returns:
    -------
        PETSc.Mat: sparse interpolation matrix

    """
    assert cad_mesh.topology.cell_types[0] == dolfinx.mesh.CellType.triangle

    cell_idx_vector = construct_pixel_to_cell_mapping_stereo(
        cad_mesh, img_shape, projection_matrix
    )

    V = dolfinx.fem.functionspace(cad_mesh, ("CG", 1, (displacement_shape,)))
    num_cells, *_ = np.shape(cad_mesh.geometry.dofmap)
    cell_to_dofs = np.array([V.dofmap.cell_dofs(i) for i in range(num_cells)])
    dofs_coords = V.tabulate_dof_coordinates()
    w, h = img_shape

    @numba.njit(fastmath=True)
    def _assembly_loop() -> tuple[NDArray, NDArray, NDArray]:
        row_ptr = np.zeros(len(cell_idx_vector) + 1, dtype=np.int64)
        nnz = 3 * np.shape(cell_idx_vector[cell_idx_vector != -1])[0]
        col_ind = np.zeros(nnz, dtype=np.int64)
        buffer = np.zeros(nnz, dtype=np.double)
        cur_idx = 0
        for i in range(len(cell_idx_vector)):
            if cell_idx_vector[i] == -1:
                row_ptr[i + 1] = row_ptr[i]
                continue
            row_ptr[i + 1] = 3 + row_ptr[i]
            dof_indices = cell_to_dofs[cell_idx_vector[i]]
            triangle_nodes = dofs_coords[dof_indices]
            triangle_nodes_4d = np.ones((3, 4))
            triangle_nodes_4d[:, :3] = triangle_nodes
            proj = (projection_matrix @ triangle_nodes_4d.T).T
            proj /= proj[:, -1:]
            triangle_nodes_img = proj[:, :2]
            x_img = np.array([(i // 2) % w, (i // 2) // h], dtype=np.float64)
            basis_values = dlf_dic.dic.p1_interpolation_jit.tabulate_P1_basis_triangle(
                x_img, np.ascontiguousarray(triangle_nodes_img)
            )
            buffer[cur_idx: cur_idx + 3] = basis_values
            col_ind[cur_idx: cur_idx + 3] = (
                displacement_shape * dof_indices + i % displacement_shape
            )
            cur_idx += 3
        row_ptr[-1] = nnz
        return row_ptr, col_ind, buffer

    print("[info] constructing interpolation matrix...")
    row_ptr, col_ind, buffer = _assembly_loop()
    print("[info] done.")

    row_ptr = np.array(row_ptr, dtype=np.int32)
    col_ind = np.array(col_ind, dtype=np.int32)
    mat = PETSc.Mat().createAIJWithArrays(
        size=(len(cell_idx_vector), V.tabulate_dof_coordinates().shape[0] * displacement_shape),
        csr=(row_ptr, col_ind, buffer),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    return mat