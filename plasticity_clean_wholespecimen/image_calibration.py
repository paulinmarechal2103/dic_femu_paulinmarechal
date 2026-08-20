"""image_calibration.py – 2-D DIC camera calibration utilities.

This module builds the 4×4 homogeneous transformation matrix **T** that maps
every point expressed in CAD/world coordinates (metres or millimetres) to the
corresponding pixel position in the DIC camera image:

    x_img = T @ x_cad       (homogeneous column vectors)

The calibration pipeline is split into two independent stages:

1. **CAD → reference image** (``construct_reference_cad_image``):
   The CAD mesh is rasterised into a synthetic grey-level image that has the
   same pixel dimensions as the real DIC image.  The mesh is centred inside
   the image and scaled to fill it as much as possible while preserving its
   aspect ratio.  The resulting 4×4 matrix ``T_cad_ref`` is returned alongside
   the synthetic image.

2. **Reference image → real DIC image** (``register_imgs`` / ``register_imgs_manual``):
   The synthetic CAD image is aligned to the real speckle image.  Two
   strategies are available:

   * **Automatic** (``register_imgs``): a DFT-based similarity registration
     (translation + rotation + uniform scale) using ``imreg_dft``.  The speckle
     image is first blurred with a Gaussian filter and then thresholded with
     Multi-Otsu to isolate the specimen silhouette.

   * **Manual** (``register_imgs_manual``): the user clicks matching landmark
     pairs in an OpenCV side-by-side window.  An affine matrix is then
     estimated robustly from those pairs.

The final calibration matrix is:

    T = T_ref_img @ T_cad_ref

where ``T_ref_img`` is the inverse of the transform recovered by the
registration step (so that it maps *from* image pixels *to* CAD coords when
inverted, or equivalently, maps CAD coords *to* image pixels in the direct
form used here).

Public API
----------
``calibrate_2d``         – fully automatic calibration.
``calibrate_2d_manual``  – manual landmark-based calibration.
``check_calibration_2d`` – 3-D visualisation of the calibration result.
"""

from __future__ import annotations

import cv2
import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import imreg_dft
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import skimage.io
import skimage.transform
from numpy.typing import NDArray
from skimage.filters import gaussian, threshold_multiotsu, threshold_otsu

# ---------------------------------------------------------------------------
# Utility – 2-D geometry helpers
# ---------------------------------------------------------------------------


def rotation_matrix_2D(theta: float) -> NDArray:
    """Return the 2×2 rotation matrix for angle *theta* (radians).

    Args:
        theta (float): Counter-clockwise rotation angle in radians.

    Returns:
        NDArray: Shape ``(2, 2)`` rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def get_xy_bounding_box(x: NDArray) -> NDArray:
    """Compute the axis-aligned 2-D bounding box of a point cloud.

    Only the first two columns (X, Y) of *x* are considered; any Z column is
    ignored.  This is intentional: the CAD mesh lives in the XY plane.

    Args:
        x (NDArray): Point cloud, shape ``(N, ≥2)``.

    Returns:
        NDArray: Shape ``(2, 2)`` array ``[[x_min, y_min], [x_max, y_max]]``.
    """
    return np.array(
        [
            [np.min(x[:, 0]), np.min(x[:, 1])],
            [np.max(x[:, 0]), np.max(x[:, 1])],
        ]
    )


def get_xyz_bounding_box(x: NDArray) -> NDArray:
    """Compute the axis-aligned 3-D bounding box of a point cloud.

    Args:
        x (NDArray): Point cloud, shape ``(N, 3)``.

    Returns:
        NDArray: Shape ``(2, 3)`` array ``[[x_min, y_min, z_min],
        [x_max, y_max, z_max]]``.
    """
    return np.array(
        [
            [np.min(x[:, 0]), np.min(x[:, 1]), np.min(x[:, 2])],
            [np.max(x[:, 0]), np.max(x[:, 1]), np.max(x[:, 2])],
        ]
    )


# ---------------------------------------------------------------------------
# Utility – affine transform builders
# ---------------------------------------------------------------------------


def affine_transform_bounding_boxes_2D(
    bb0: NDArray, bb1: NDArray
) -> tuple[NDArray, NDArray]:
    """Build a 2-D affine that maps bounding box *bb0* into *bb1*.

    The transform scales *bb0* uniformly (preserving aspect ratio) so that it
    fits entirely inside *bb1*, then translates it to align their centres.

    The resulting transform is encoded as a **scale** vector and a
    **pre-translation** vector so that the actual mapping reads::

        x' = scale * (x + translation)

    i.e. first translate, then scale.  This convention is compatible with
    :func:`affine_transform_2D_to_4D`.

    Bounding-box format: ``NDArray[[x_min, y_min], [x_max, y_max]]``.

    Args:
        bb0 (NDArray): Source bounding box, shape ``(2, 2)`` (CAD space).
        bb1 (NDArray): Destination bounding box, shape ``(2, 2)`` (image
            space, in pixels).

    Returns:
        tuple[NDArray, NDArray]:
            * ``scale`` – uniform scale factor applied to *both* axes,
              shape ``(2,)``.
            * ``translation`` – pre-scale translation vector, shape ``(2,)``.
    """
    # Dimensions of each box
    w0 = abs(bb0[1, 0] - bb0[0, 0])
    h0 = abs(bb0[1, 1] - bb0[0, 1])
    w1 = abs(bb1[1, 0] - bb1[0, 0])
    h1 = abs(bb1[1, 1] - bb1[0, 1])

    # Uniform scale: pick the tighter ratio so the mesh fits completely
    w_ratio = w1 / w0
    h_ratio = h1 / h0
    resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio

    # Centres of both boxes
    center0 = (bb0[0] + bb0[1]) / 2.0
    center1 = (bb1[0] + bb1[1]) / 2.0

    # Solve for translation such that scale*(center0 + t) = center1
    translation = (center1 / resize_ratio) - center0

    return np.array([resize_ratio, resize_ratio]), translation


def affine_transform_2D_to_4D(scaling: NDArray, translation: NDArray) -> NDArray:
    """Lift a 2-D *scale-then-translate* transform into a 4×4 homogeneous matrix.

    The 2-D transform is defined as::

        x' = s * (x + t)   =>   x' = s*x + s*t

    which expands to the 4×4 matrix::

        [[s_x,  0,   0,  s_x*t_x],
         [ 0,  s_y,  0,  s_y*t_y],
         [ 0,   0,   1,    0    ],
         [ 0,   0,   0,    1    ]]

    The Z row/column is left as identity because CAD meshes reside in the XY
    plane (Z = 0).

    Args:
        scaling (NDArray): Shape ``(2,)`` – per-axis scale factors ``[s_x, s_y]``.
        translation (NDArray): Shape ``(2,)`` – pre-scale translation ``[t_x, t_y]``.

    Returns:
        NDArray: Shape ``(4, 4)`` homogeneous transformation matrix.
    """
    assert len(scaling) == 2, "scaling must have exactly 2 components"
    assert len(translation) == 2, "translation must have exactly 2 components"

    return np.array(
        [
            [scaling[0], 0, 0, scaling[0] * translation[0]],
            [0, scaling[1], 0, scaling[1] * translation[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def transform_4D_to_3D(transform_4D: NDArray) -> NDArray:
    """Extract a 3×3 homogeneous matrix from a 4×4 one (XY plane only).

    The mapping discards the Z row and column but preserves the translation
    components stored in column 3 of the 4×4 matrix.  This is the inverse of
    :func:`transform_3D_to_4D`.

    Args:
        transform_4D (NDArray): Shape ``(4, 4)`` homogeneous matrix.

    Returns:
        NDArray: Shape ``(3, 3)`` matrix equivalent in the XY plane.
    """
    t3 = np.zeros((3, 3), dtype=transform_4D.dtype)
    t3[:2, :2] = transform_4D[:2, :2]
    t3[:2, 2] = transform_4D[:2, 3]
    t3[2, :2] = transform_4D[3, :2]
    t3[2, 2] = transform_4D[3, 3]
    return t3


def transform_3D_to_4D(transform_3D: NDArray) -> NDArray:
    """Embed a 3×3 homogeneous matrix into a 4×4 one (XY plane only).

    The mapping is the inverse of :func:`transform_4D_to_3D`.

    Args:
        transform_3D (NDArray): Shape ``(3, 3)`` homogeneous matrix.

    Returns:
        NDArray: Shape ``(4, 4)`` matrix with Z row/column set to identity.
    """
    t4 = np.zeros((4, 4), dtype=transform_3D.dtype)
    t4[:2, :2] = transform_3D[:2, :2]
    t4[:2, 3] = transform_3D[:2, 2]
    t4[3, :2] = transform_3D[2, :2]
    t4[3, 3] = transform_3D[2, 2]
    t4[2, 2] = 1.0
    return t4


# ---------------------------------------------------------------------------
# Utility – mesh / image conversion helpers
# ---------------------------------------------------------------------------


def img_uniform_grid(img_shape: tuple[int, int]) -> pv.ImageData:
    """Build a PyVista uniform grid matching an image's pixel layout.

    Each cell of the grid corresponds to one pixel.  The grid spans from
    ``(0, 0, 0)`` to ``(width, height, 1)`` with unit spacing, matching the
    NumPy image convention (columns = X, rows = Y).

    This grid is used to *probe* (interpolate) a mesh onto pixel positions via
    :meth:`pv.DataSet.sample`.

    Args:
        img_shape (tuple[int, int]): ``(height, width)`` in pixels – the
            standard NumPy ``array.shape`` convention.

    Returns:
        pv.ImageData: Structured grid with ``(width, height, 1)`` cells.
    """
    h, w = img_shape
    return pv.ImageData(dimensions=(w, h, 1), spacing=(1, 1, 1), origin=(0, 0, 0))


def dolfinx_mesh_to_pv_mesh(dolfinx_mesh: dolfinx.mesh.Mesh) -> pv.UnstructuredGrid:
    """Convert a DOLFINx mesh to an equivalent PyVista unstructured grid.

    The conversion uses :func:`dolfinx.plot.vtk_mesh` to extract the topology
    and geometry arrays required by PyVista.

    Args:
        dolfinx_mesh (dolfinx.mesh.Mesh): Source mesh (any topological
            dimension).

    Returns:
        pv.UnstructuredGrid: PyVista representation of the same mesh.
    """
    topology, cell_types, x = dolfinx.plot.vtk_mesh(
        dolfinx_mesh, dim=dolfinx_mesh.topology.dim
    )
    return pv.UnstructuredGrid(topology, cell_types, x)


# ---------------------------------------------------------------------------
# Stage 1 – CAD image construction
# ---------------------------------------------------------------------------


# def construct_reference_cad_image(
#     cad_mesh: pv.UnstructuredGrid, ref_img_shape: tuple[int, int]
# ) -> tuple[NDArray, NDArray]:
#     """Rasterise the CAD mesh into a synthetic grey-level image.

#     The mesh is scaled and centred to fill the image canvas as much as
#     possible while preserving its aspect ratio.  Each pixel that falls inside
#     a mesh cell receives value 1; all other pixels are 0.

#     This synthetic image is later aligned to the real DIC speckle image using
#     either :func:`register_imgs` (automatic) or :func:`register_imgs_manual`
#     (manual).

#     Algorithm
#     ---------
#     1. Convert the DOLFINx mesh to a PyVista unstructured grid.
#     2. Compute the 4×4 transform ``T_cad_ref`` that maps CAD coordinates to
#        centred pixel coordinates (see :func:`affine_transform_bounding_boxes_2D`).
#     3. Apply ``T_cad_ref`` to the PyVista mesh in place.
#     4. Create a constant DG0 (cell-wise constant) scalar field of value 1 over
#        the mesh (the "silhouette" mask).
#     5. Sample the transformed mesh onto a uniform pixel grid to produce a
#        2-D array.

#     Args:
#         cad_mesh (dolfinx.mesh.Mesh): CAD mesh in world/physical coordinates.
#         ref_img_shape (tuple[int, int]): Target image size ``(height, width)``
#             in pixels.

#     Returns:
#         tuple[NDArray, NDArray]:
#             * ``cad_ref_img`` – Grey-level image of shape ``(height, width)``
#               with values in ``{0, 1}``.
#             * ``T_cad_ref`` – 4×4 homogeneous matrix that maps CAD points to
#               the canvas pixel coordinates used to generate the image.
#     """
#     # -- Convert DOLFINx mesh to PyVista so we can apply arbitrary transforms
#     pv_mesh = dolfinx_mesh_to_pv_mesh(cad_mesh)

#     hauteur, largeur = ref_img_shape

#     # Uniform pixel grid: one cell per pixel
#     probe_grid = img_uniform_grid((hauteur, largeur))

#     # Bounding box of the CAD mesh in XY
#     bounding_box_cad = get_xy_bounding_box(cad_mesh.geometry.x)

#     # Destination bounding box = full image canvas [0,W] x [0,H]
#     bounding_box_img = np.array([[0.0, 0.0], [largeur, hauteur]])

#     # Build the centering + scaling transform
#     scaling, translation = affine_transform_bounding_boxes_2D(
#         bounding_box_cad, bounding_box_img
#     )
#     T_cad_ref = affine_transform_2D_to_4D(scaling, translation)

#     # Move the PyVista mesh to pixel coordinates
#     pv_mesh.transform(T_cad_ref, inplace=True)

#     # Paint the mesh with a uniform scalar (value = 1 everywhere inside cells)
#     DG0 = dolfinx.fem.functionspace(cad_mesh, ("DG", 0))
#     dummy_fn = dolfinx.fem.Function(DG0)
#     dummy_fn.x.array[:] = 1.0
#     pv_mesh.cell_data["mask"] = dummy_fn.x.array
#     pv_mesh.set_active_scalars("mask")

#     # Sample the mesh onto the pixel grid (interpolation by cell membership)
#     probed_grid = probe_grid.sample(pv_mesh)
#     probed_grid.set_active_scalars("mask")

#     # Reshape to (H, W) and replace NaN (pixels outside the mesh) with 0
#     cad_ref_img = np.reshape(probed_grid.active_scalars, (hauteur, largeur))
#     cad_ref_img = np.nan_to_num(cad_ref_img, nan=0.0)

#     return cad_ref_img, T_cad_ref

def construct_reference_cad_image(
    cad_mesh: pv.UnstructuredGrid, ref_img_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """Rasterise the CAD mesh into a synthetic grey-level image.

    The mesh is scaled and centred to fill the image canvas as much as
    possible while preserving its aspect ratio.  Each pixel that falls inside
    a mesh cell receives value 1; all other pixels are 0.

    This synthetic image is later aligned to the real DIC speckle image using
    either :func:`register_imgs` (automatic) or :func:`register_imgs_manual`
    (manual).

    Algorithm
    ---------
    1. Copy the PyVista mesh to preserve the original object.
    2. Compute the 4×4 transform ``T_cad_ref`` that maps CAD coordinates to
       centred pixel coordinates (see :func:`affine_transform_bounding_boxes_2D`).
    3. Apply ``T_cad_ref`` to the PyVista mesh.
    4. Set a cell-wise array of value 1 over the mesh (the "silhouette" mask).
    5. Sample the transformed mesh onto a uniform pixel grid to produce a
       2-D array.

    Args:
        cad_mesh (pv.UnstructuredGrid): CAD mesh in world/physical coordinates.
        ref_img_shape (tuple[int, int]): Target image size ``(height, width)``
            in pixels.

    Returns:
        tuple[NDArray, NDArray]:
            * ``cad_ref_img`` – Grey-level image of shape ``(height, width)``
              with values in ``{0, 1}``.
            * ``T_cad_ref`` – 4×4 homogeneous matrix that maps CAD points to
              the canvas pixel coordinates used to generate the image.
    """
    # Work on a copy to avoid mutating the original input mesh
    pv_mesh = cad_mesh.copy()

    hauteur, largeur = ref_img_shape

    # Uniform pixel grid: one cell per pixel
    probe_grid = img_uniform_grid((hauteur, largeur))

    # Bounding box of the CAD mesh in XY
    bounding_box_cad = get_xy_bounding_box(pv_mesh.points)

    # Destination bounding box = full image canvas [0,W] x [0,H]
    bounding_box_img = np.array([[0.0, 0.0], [largeur, hauteur]])

    # Build the centering + scaling transform
    scaling, translation = affine_transform_bounding_boxes_2D(
        bounding_box_cad, bounding_box_img
    )
    T_cad_ref = affine_transform_2D_to_4D(scaling, translation)

    # Move the PyVista mesh to pixel coordinates
    pv_mesh.transform(T_cad_ref, inplace=True)

    # Paint the mesh with a uniform scalar (value = 1 everywhere inside cells)
    pv_mesh.cell_data["mask"] = np.ones(pv_mesh.n_cells, dtype=np.float64)
    pv_mesh.set_active_scalars("mask")

    # Sample the mesh onto the pixel grid (interpolation by cell membership)
    probed_grid = probe_grid.sample(pv_mesh)
    probed_grid.set_active_scalars("mask")

    # Reshape to (H, W) and replace NaN (pixels outside the mesh) with 0
    cad_ref_img = np.reshape(probed_grid.active_scalars, (hauteur, largeur))
    cad_ref_img = np.nan_to_num(cad_ref_img, nan=0.0)

    return cad_ref_img, T_cad_ref


# ---------------------------------------------------------------------------
# Stage 2a – automatic DFT-based registration
# ---------------------------------------------------------------------------


def register_imgs(
    img0: NDArray,
    img1: NDArray,
    max_angle: float = 10.0,
    min_scale: float = 0.6,
    max_scale: float = 0.7,
    rescale_factor: float = 1.0,
    sigma: float = 3.0,
) -> NDArray:
    """Align the CAD silhouette image *img0* to the speckle image *img1*.

    Both images must have the same pixel dimensions.

    **Pre-processing pipeline**

    * *img0* (CAD): Otsu thresholding → binary silhouette.
    * *img1* (speckle): Gaussian blur (parameter *sigma*) to smooth speckle
      texture → Multi-Otsu thresholding → binary silhouette.
    * Both binaries are optionally down-sampled by *rescale_factor* before
      passing to the DFT correlator to speed up registration.

    **DFT registration**

    ``imreg_dft.similarity`` estimates a similarity transform
    (translation + rotation + uniform scale) between the two images in the
    frequency domain.  The result is constrained by *max_angle* (maximum
    rotation) and ``[min_scale, max_scale]`` (scale range).

    The function returns the **inverse** of the estimated transform so that it
    maps image pixels back to CAD coordinates (consistent with the convention
    in :func:`calibrate_2d`).

    A diagnostic figure is displayed showing all intermediate binary images.

    Args:
        img0 (NDArray): CAD silhouette (source), shape ``(H, W)``.
        img1 (NDArray): Real DIC speckle image (target), shape ``(H, W)``.
        max_angle (float): Maximum absolute rotation allowed (degrees).
            Defaults to ``10``.
        min_scale (float): Lower bound on the scale parameter for the
            registration. Defaults to ``0.6``.
        max_scale (float): Upper bound on the scale parameter. Defaults to
            ``0.7``.
        rescale_factor (float): Down-sampling factor applied before the DFT
            step to reduce computation time (``1.0`` = no down-sampling).
            Defaults to ``1.0``.
        sigma (float): Standard deviation of the Gaussian blur applied to
            *img1* before thresholding.  Larger values merge speckle dots
            into a smooth region. Defaults to ``3.0``.

    Returns:
        NDArray: Shape ``(4, 4)`` inverse similarity transform that maps
        image-space coordinates back to the CAD reference.

    Raises:
        AssertionError: If *img0* and *img1* have different shapes.
    """
    assert np.shape(img0) == np.shape(img1), (
        "img0 and img1 must have the same shape"
    )

    # ------------------------------------------------------------------
    # Step 1: binarise img0 (CAD silhouette) with Otsu
    # ------------------------------------------------------------------
    threshold0 = threshold_otsu(img0)
    binary0 = (img0 >= threshold0).astype(np.uint8) * 255
    binary0_ds = skimage.transform.rescale(binary0, rescale_factor, anti_aliasing=False)

    # ------------------------------------------------------------------
    # Step 2: binarise img1 (speckle) – blur first, then Multi-Otsu
    # ------------------------------------------------------------------
    img1_blurred = gaussian(img1, sigma=sigma, preserve_range=True)
    thresholds1 = threshold_multiotsu(img1_blurred)
    # Keep the brightest region (specimen surface, higher grey level)
    binary1 = (img1_blurred >= thresholds1[0]).astype(np.uint8) * 255
    binary1_ds = skimage.transform.rescale(binary1, rescale_factor, anti_aliasing=False)

    # ------------------------------------------------------------------
    # Diagnostic plot (shown interactively)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img0, cmap="gray")
    axes[0, 0].set_title("img0 – original CAD")
    axes[0, 1].axis("off")  # spacer
    axes[0, 2].imshow(binary0_ds, cmap="gray")
    axes[0, 2].set_title(f"img0 – binary (rescale={rescale_factor})")

    axes[1, 0].imshow(img1, cmap="gray")
    axes[1, 0].set_title("img1 – original speckle")
    axes[1, 1].imshow(img1_blurred, cmap="gray")
    axes[1, 1].set_title(f"img1 – Gaussian blur (σ={sigma})")
    axes[1, 2].imshow(binary1_ds, cmap="gray")
    axes[1, 2].set_title("img1 – binary (Multi-Otsu)")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 3: DFT-based similarity registration
    # imreg_dft works with (width, height) ordering → transpose inputs
    # ------------------------------------------------------------------
    constraints = {"angle": [0, max_angle], "scale": [min_scale, max_scale]}
    tform = imreg_dft.similarity(
        binary0_ds.T,
        binary1_ds.T,
        order=2,
        numiter=3,
        constraints=constraints,
    )

    # ------------------------------------------------------------------
    # Step 4: convert imreg_dft result back to full-resolution coordinates
    # imreg_dft returns translation in down-sampled pixels → rescale
    # ------------------------------------------------------------------
    trans = tform["tvec"] / rescale_factor
    scale = tform["scale"]
    angle = tform["angle"]
    theta = np.deg2rad(angle)

    # Full similarity matrix in 2-D
    rot_scale = scale * rotation_matrix_2D(theta)

    # The rotation centre is the image centre; account for it in translation
    h, w = np.shape(img0)
    img_center = np.array([w / 2.0, h / 2.0])
    full_trans = trans + (np.eye(2) - rot_scale) @ img_center

    # Embed into a 4×4 homogeneous matrix
    T4 = np.array(
        [
            [rot_scale[0, 0], rot_scale[0, 1], 0.0, full_trans[0]],
            [rot_scale[1, 0], rot_scale[1, 1], 0.0, full_trans[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Return the *inverse* so the caller can compose T_cad_img = T_inv @ T_cad_ref
    return np.linalg.inv(T4)


# ---------------------------------------------------------------------------
# Stage 2b – manual landmark-based registration
# ---------------------------------------------------------------------------

def register_imgs_manual(img0: NDArray, img1: NDArray) -> NDArray:
    """Align *img0* to *img1* via interactive landmark selection in OpenCV.

    Displays both images side-by-side at full resolution.
    The user clicks corresponding structural landmarks on
    the left (CAD image) and right (real image) panels.
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    if h0 != h1:
        raise ValueError(
            f"Les deux images doivent avoir la même hauteur pour np.hstack "
            f"(reçu {h0}px et {h1}px)."
        )

    # Convert float images to uint8 for display / OpenCV
    def _to_uint8(arr: NDArray) -> NDArray:
        if arr.dtype == np.uint8:
            return arr.copy()
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    base0 = _to_uint8(img0)
    base1 = _to_uint8(img1)

    # Lists of full-resolution point pairs
    pts0: list[list[int]] = []
    pts1: list[list[int]] = []

    window_name = "Manual Calibration - ENTER to validate, ESC to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _click_event(event: int, x: int, y: int, flags: int, param: object) -> None:
        """Mouse callback: classify click as CAD (left panel) or real (right panel)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < w0:
                # Left panel → CAD image coordinates
                pts0.append([x, y])
                print(f"  CAD point #{len(pts0):2d}: ({x}, {y})")
            else:
                # Right panel → real image coordinates
                real_x = x - w0
                pts1.append([real_x, y])
                print(f"  Real point #{len(pts1):2d}: ({real_x}, {y})")

    cv2.setMouseCallback(window_name, _click_event)

    print("\n--- MANUAL CALIBRATION INSTRUCTIONS ---")
    print("  1. Click a recognisable structural point on the LEFT  (CAD image).")
    print("  2. Click its counterpart on the RIGHT (real DIC image).")
    print("  3. Repeat for at least 3 – 4 point pairs.")
    print("  4. Press ENTER to validate, or ESC to abort.\n")

    # Interactive display loop
    while True:
        vis0 = cv2.cvtColor(base0, cv2.COLOR_GRAY2BGR) if base0.ndim == 2 else base0.copy()
        vis1 = cv2.cvtColor(base1, cv2.COLOR_GRAY2BGR) if base1.ndim == 2 else base1.copy()

        # Draw CAD landmarks in red
        for idx, pt in enumerate(pts0):
            cv2.circle(vis0, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.putText(vis0, str(idx + 1), (pt[0] + 8, pt[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw real landmarks in green
        for idx, pt in enumerate(pts1):
            cv2.circle(vis1, (pt[0], pt[1]), 5, (0, 255, 0), -1)
            cv2.putText(vis1, str(idx + 1), (pt[0] + 8, pt[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window_name, np.hstack((vis0, vis1)))

        key = cv2.waitKey(30) & 0xFF
        if key == 13:    # Enter
            break
        elif key == 27:  # Escape
            print("Calibration cancelled by user.")
            cv2.destroyAllWindows()
            return np.identity(4)

    cv2.destroyAllWindows()

    # Validate collected points
    n_pairs = min(len(pts0), len(pts1))
    if n_pairs < 3:
        print(f"[ERROR] Not enough matched pairs (need ≥ 3, got {n_pairs}).")
        return np.identity(4)

    if len(pts0) != len(pts1):
        print(
            f"[WARNING] Unbalanced point counts: using first {n_pairs} pairs."
        )

    src = np.array(pts0[:n_pairs], dtype=np.float32)
    dst = np.array(pts1[:n_pairs], dtype=np.float32)

    # Robust RANSAC-based affine estimation (2×3 matrix)
    M, _ = cv2.estimateAffine2D(src, dst)

    if M is None:
        print("[ERROR] Affine estimation failed. Returning identity.")
        return np.identity(4)

    # Embed the 2×3 affine matrix into a 4×4 homogeneous one
    T4 = np.identity(4)
    T4[:2, :2] = M[:2, :2]
    T4[:2, 3] = M[:2, 2]

    return T4
# ---------------------------------------------------------------------------
# High-level calibration entry points
# ---------------------------------------------------------------------------


def calibrate_2d(
    mesh: dolfinx.mesh.Mesh,
    ref_img: NDArray,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
) -> NDArray:
    """Compute the 4×4 CAD-to-image transform using **automatic** DFT registration.

    This is the main entry point for fully automated 2-D DIC calibration.  It
    chains :func:`construct_reference_cad_image` with :func:`register_imgs` to
    produce the composite transform:

    .. math::

        \\mathbf{T} = \\mathbf{T}_{\\text{ref} \\to \\text{img}} \\;
                      \\mathbf{T}_{\\text{CAD} \\to \\text{ref}}

    Args:
        mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates.
        ref_img (NDArray): Reference DIC image (grey-level), shape ``(H, W)``.
        min_scale (float): Lower bound on the similarity scale searched by
            ``imreg_dft``. Defaults to ``0.7``.
        max_scale (float): Upper bound on the similarity scale. Defaults to
            ``1.3``.

    Returns:
        NDArray: Shape ``(4, 4)`` transform mapping CAD points to image
        pixels.
    """
    cad_ref_img, T_cad_ref = construct_reference_cad_image(mesh, np.shape(ref_img))
    T_ref_img = register_imgs(cad_ref_img, ref_img,
                              min_scale=min_scale, max_scale=max_scale)
    return T_ref_img @ T_cad_ref


def calibrate_2d_manual(mesh: dolfinx.mesh.Mesh, ref_img: NDArray) -> NDArray:
    """Compute the 4×4 CAD-to-image transform via **manual** landmark selection.

    This is an alternative to :func:`calibrate_2d` that does not rely on
    automatic DFT registration.  It is useful when the automatic approach
    fails, e.g. for very large rotations or unusual specimen shapes.

    The pipeline is:

    1. :func:`construct_reference_cad_image` – rasterise the CAD mesh.
    2. :func:`register_imgs_manual` – interactive OpenCV point-clicking.
    3. Compose the two transforms.

    Args:
        mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates.
        ref_img (NDArray): Reference DIC image (grey-level), shape ``(H, W)``.

    Returns:
        NDArray: Shape ``(4, 4)`` transform mapping CAD points to image
        pixels.
    """
    cad_ref_img, T_cad_ref = construct_reference_cad_image(mesh, np.shape(ref_img))
    T_ref_img = register_imgs_manual(cad_ref_img, ref_img)
    return T_ref_img @ T_cad_ref


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def check_calibration_2d(
    mesh: pv.UnstructuredGrid,
    ref_img: NDArray,
    tform_cad_to_img: NDArray,
) -> None:
    """Visualise calibration quality in an interactive 3-D PyVista window.

    The DIC reference image is placed as a flat pixel grid at Z = 0.  The CAD
    mesh, after applying *tform_cad_to_img*, is overlaid at Z = 1 so it
    floats above the image.  A well-calibrated mesh should tightly follow the
    specimen boundary visible in the image.

    Args:
        mesh (pv.UnstructuredGrid): CAD mesh in world coordinates.
        ref_img (NDArray): Reference DIC image (grey-level), shape ``(H, W)``.
        tform_cad_to_img (NDArray): Shape ``(4, 4)`` calibration matrix.
    """
    h, w = np.shape(ref_img)

    # Build a flat PyVista image grid and attach grey-level values
    img_pv = pv.ImageData(dimensions=(w + 1, h + 1, 1))
    img_pv.cell_data["gray_level"] = ref_img.flatten()

    # Transform the CAD mesh and lift it slightly above the image plane
    mesh_pv = mesh.transform(tform_cad_to_img, inplace=False)
    mesh_pv = mesh_pv.translate([0.0, 0.0, 1.0], inplace=False)

    p = pv.Plotter()
    p.add_mesh(img_pv, cmap="gray", show_scalar_bar=False)
    p.add_mesh(mesh_pv, show_edges=True, color="red")
    p.show()
# ---------------------------------------------------------------------------
# Module-level smoke test (run directly: python image_calibration.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from mpi4py import MPI
    from dolfinx import io

    # Load a mesh from XDMF
    from dic_importation import read_msh_safely
    domain = read_msh_safely("carre_trou.msh")
    # Load a reference speckle image in grey-level
    ref_img = skimage.io.imread("/home/pmarechal/Documents/ESCAL/transfer_12868454_files_9cfdfa8b/face_avant/test000000.tif",
                                as_gray=True)

    # Automatic calibration
    T = calibrate_2d_manual(domain, ref_img)
    print("Calibration matrix (automatic):")
    print(T)

    # Visual verification
    check_calibration_2d(domain, ref_img, T)
