"""
image_calibration.py
=====================

2-D Digital Image Correlation (DIC) camera calibration utilities.

Purpose
-------
This module computes the 4x4 homogeneous transformation matrix `T` that
maps a point expressed in CAD / world coordinates (metres or millimetres)
to the corresponding pixel position in a real DIC camera image:

    x_img = T @ x_cad      (using homogeneous column vectors)

Pipeline overview
------------------
Calibration is performed in two independent, composable stages:

1. CAD -> synthetic reference image
   (`construct_reference_cad_image`)
   The CAD mesh is rasterised into a synthetic binary/grey-level image
   with the same pixel dimensions as the real DIC image. The mesh is
   centred and uniformly scaled to fill as much of the canvas as
   possible while preserving its aspect ratio. This produces both the
   synthetic image and the 4x4 matrix `T_cad_ref` used to generate it.

2. Synthetic reference image -> real DIC image
   Two alternative registration strategies are provided:

   * Automatic (`register_imgs`): frequency-domain (DFT) similarity
     registration (translation + rotation + uniform scale) via
     `imreg_dft`. The real speckle image is first Gaussian-blurred and
     Multi-Otsu thresholded to isolate the specimen silhouette; the CAD
     image is Otsu-thresholded.

   * Manual (`register_imgs_manual`): the user clicks matching landmark
     pairs in a side-by-side OpenCV window, and a robust (RANSAC) affine
     transform is estimated from those correspondences.

The two stages are composed into the final calibration matrix:

    T = T_ref_img @ T_cad_ref

where `T_ref_img` is the transform recovered by the registration step
that maps the synthetic reference image onto the real image.

Public API
----------
`calibrate_2d`          -- fully automatic end-to-end calibration.
`calibrate_2d_manual`   -- manual, landmark-based end-to-end calibration.
`check_calibration_2d`  -- interactive 3-D visual sanity check of the result.

Notes on mesh types
--------------------
`construct_reference_cad_image` (and therefore `calibrate_2d` /
`calibrate_2d_manual`) currently expects a `pyvista.UnstructuredGrid`
(NOT a raw `dolfinx.mesh.Mesh`). If starting from a DOLFINx mesh, first
convert it with `dolfinx_mesh_to_pv_mesh`.
"""

from __future__ import annotations

import cv2
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
# Utility - 2-D geometry helpers
# ---------------------------------------------------------------------------


def rotation_matrix_2D(theta: float) -> NDArray:
    """Build a 2x2 counter-clockwise rotation matrix.

    Args:
        theta (float): Rotation angle in radians.

    Returns:
        NDArray: Shape ``(2, 2)`` rotation matrix such that
        ``R @ v`` rotates the 2-D vector ``v`` by *theta* radians
        counter-clockwise.
    """
    # Standard 2x2 rotation matrix definition
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def get_xy_bounding_box(x: NDArray) -> NDArray:
    """Compute the axis-aligned XY bounding box of a point cloud.

    Only the first two columns (X, Y) of *x* are used; any additional
    columns (e.g. Z) are ignored. This is intentional: CAD meshes used
    here are assumed to lie in the XY plane.

    Args:
        x (NDArray): Point cloud, shape ``(N, >=2)``.

    Returns:
        NDArray: Shape ``(2, 2)`` array ``[[x_min, y_min], [x_max, y_max]]``.
    """
    # Row 0 = minimum corner, Row 1 = maximum corner
    return np.array(
        [
            [np.min(x[:, 0]), np.min(x[:, 1])],
            [np.max(x[:, 0]), np.max(x[:, 1])],
        ]
    )


def get_xyz_bounding_box(x: NDArray) -> NDArray:
    """Compute the axis-aligned XYZ bounding box of a point cloud.

    Args:
        x (NDArray): Point cloud, shape ``(N, 3)``.

    Returns:
        NDArray: Shape ``(2, 3)`` array
        ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``.
    """
    # Same idea as get_xy_bounding_box but including the Z axis
    return np.array(
        [
            [np.min(x[:, 0]), np.min(x[:, 1]), np.min(x[:, 2])],
            [np.max(x[:, 0]), np.max(x[:, 1]), np.max(x[:, 2])],
        ]
    )


# ---------------------------------------------------------------------------
# Utility - affine transform builders
# ---------------------------------------------------------------------------


def affine_transform_bounding_boxes_2D(
    bb0: NDArray, bb1: NDArray
) -> tuple[NDArray, NDArray]:
    """Compute a 2-D affine (uniform scale + translation) that fits *bb0* inside *bb1*.

    The source box *bb0* is scaled uniformly (i.e. preserving its aspect
    ratio) so that it fits entirely inside the destination box *bb1*, then
    the two boxes are centred on one another.

    The transform is returned as a *scale* vector and a *pre-scale
    translation* vector, such that the mapping applied to a point ``x`` is::

        x' = scale * (x + translation)

    i.e. translate first, then scale. This convention matches
    :func:`affine_transform_2D_to_4D`.

    Bounding-box format: ``NDArray[[x_min, y_min], [x_max, y_max]]``.

    Args:
        bb0 (NDArray): Source bounding box, shape ``(2, 2)`` (CAD space).
        bb1 (NDArray): Destination bounding box, shape ``(2, 2)`` (image
            space, in pixels).

    Returns:
        tuple[NDArray, NDArray]:
            * ``scale`` -- uniform scale factor applied to both axes,
              shape ``(2,)``.
            * ``translation`` -- pre-scale translation vector, shape ``(2,)``.
    """
    # Width/height of each bounding box
    w0 = abs(bb0[1, 0] - bb0[0, 0])
    h0 = abs(bb0[1, 1] - bb0[0, 1])
    w1 = abs(bb1[1, 0] - bb1[0, 0])
    h1 = abs(bb1[1, 1] - bb1[0, 1])

    # Choose the scale ratio that guarantees bb0 fits fully inside bb1
    # (the smaller of the two candidate ratios), preserving aspect ratio.
    w_ratio = w1 / w0
    h_ratio = h1 / h0
    resize_ratio = h_ratio if w0 * h_ratio <= w1 else w_ratio

    # Centres of both boxes
    center0 = (bb0[0] + bb0[1]) / 2.0
    center1 = (bb1[0] + bb1[1]) / 2.0

    # Solve translation t such that: resize_ratio * (center0 + t) = center1
    translation = (center1 / resize_ratio) - center0

    return np.array([resize_ratio, resize_ratio]), translation


def affine_transform_2D_to_4D(scaling: NDArray, translation: NDArray) -> NDArray:
    """Embed a 2-D *scale-then-translate* transform into a 4x4 homogeneous matrix.

    The 2-D transform is defined as::

        x' = s * (x + t)  =>  x' = s*x + s*t

    which expands into the 4x4 homogeneous matrix::

        [[s_x,  0,   0,  s_x*t_x],
         [ 0,  s_y,  0,  s_y*t_y],
         [ 0,   0,   1,    0    ],
         [ 0,   0,   0,    1    ]]

    The Z row/column is left as identity because the CAD meshes handled
    here live in the XY plane (Z = 0).

    Args:
        scaling (NDArray): Shape ``(2,)`` per-axis scale factors ``[s_x, s_y]``.
        translation (NDArray): Shape ``(2,)`` pre-scale translation ``[t_x, t_y]``.

    Returns:
        NDArray: Shape ``(4, 4)`` homogeneous transformation matrix.
    """
    assert len(scaling) == 2, "scaling must have exactly 2 components"
    assert len(translation) == 2, "translation must have exactly 2 components"

    # Diagonal scale terms + scaled translation baked into the last column
    return np.array(
        [
            [scaling[0], 0, 0, scaling[0] * translation[0]],
            [0, scaling[1], 0, scaling[1] * translation[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def transform_4D_to_3D(transform_4D: NDArray) -> NDArray:
    """Extract the equivalent 3x3 homogeneous matrix from a 4x4 one (XY plane only).

    Drops the Z row/column while preserving the in-plane rotation/scale
    block and the translation stored in column 3. Inverse operation of
    :func:`transform_3D_to_4D`.

    Args:
        transform_4D (NDArray): Shape ``(4, 4)`` homogeneous matrix.

    Returns:
        NDArray: Shape ``(3, 3)`` equivalent matrix in the XY plane.
    """
    t3 = np.zeros((3, 3), dtype=transform_4D.dtype)
    # Top-left 2x2 block: in-plane linear part (rotation/scale/shear)
    t3[:2, :2] = transform_4D[:2, :2]
    # Translation column
    t3[:2, 2] = transform_4D[:2, 3]
    # Bottom row (usually [0, 0, 1] for affine transforms)
    t3[2, :2] = transform_4D[3, :2]
    t3[2, 2] = transform_4D[3, 3]
    return t3


def transform_3D_to_4D(transform_3D: NDArray) -> NDArray:
    """Embed a 3x3 homogeneous matrix into a 4x4 one (XY plane only).

    Inverse operation of :func:`transform_4D_to_3D`. The Z row/column is
    filled with identity values so the transform leaves the Z coordinate
    untouched.

    Args:
        transform_3D (NDArray): Shape ``(3, 3)`` homogeneous matrix.

    Returns:
        NDArray: Shape ``(4, 4)`` matrix with Z row/column set to identity.
    """
    t4 = np.zeros((4, 4), dtype=transform_3D.dtype)
    # In-plane linear part
    t4[:2, :2] = transform_3D[:2, :2]
    # Translation
    t4[:2, 3] = transform_3D[:2, 2]
    # Bottom (affine) row
    t4[3, :2] = transform_3D[2, :2]
    t4[3, 3] = transform_3D[2, 2]
    # Z axis left untouched (identity)
    t4[2, 2] = 1.0
    return t4


# ---------------------------------------------------------------------------
# Utility - mesh / image conversion helpers
# ---------------------------------------------------------------------------


def img_uniform_grid(img_shape: tuple[int, int]) -> pv.ImageData:
    """Build a PyVista uniform grid whose cells match an image's pixel layout.

    Each grid cell corresponds to exactly one pixel. The grid spans from
    ``(0, 0, 0)`` to ``(width, height, 1)`` with unit spacing, matching
    the NumPy image convention where columns = X and rows = Y.

    This grid is used to *probe* (interpolate) a mesh onto pixel positions
    via :meth:`pv.DataSet.sample`.

    Args:
        img_shape (tuple[int, int]): ``(height, width)`` in pixels -- the
            standard NumPy ``array.shape`` convention.

    Returns:
        pv.ImageData: Structured grid with ``(width, height, 1)`` cells.
    """
    h, w = img_shape
    # dimensions are given in points, not cells: (w, h, 1) points => (w-1, h-1, 0) cells
    # but here w/h are used directly since a 1-pixel-thick 2D slab is intended.
    return pv.ImageData(dimensions=(w, h, 1), spacing=(1, 1, 1), origin=(0, 0, 0))


def dolfinx_mesh_to_pv_mesh(dolfinx_mesh: dolfinx.mesh.Mesh) -> pv.UnstructuredGrid:
    """Convert a DOLFINx mesh into an equivalent PyVista unstructured grid.

    Useful as a pre-processing step before calling
    :func:`construct_reference_cad_image`, which operates on PyVista
    meshes rather than raw DOLFINx mesh objects.

    Args:
        dolfinx_mesh (dolfinx.mesh.Mesh): Source mesh (any topological
            dimension).

    Returns:
        pv.UnstructuredGrid: PyVista representation of the same mesh.
    """
    # dolfinx.plot.vtk_mesh returns (topology, cell_types, points) arrays
    # in the VTK format expected by pv.UnstructuredGrid's constructor.
    topology, cell_types, x = dolfinx.plot.vtk_mesh(
        dolfinx_mesh, dim=dolfinx_mesh.topology.dim
    )
    return pv.UnstructuredGrid(topology, cell_types, x)


# ---------------------------------------------------------------------------
# Stage 1 - CAD image construction
# ---------------------------------------------------------------------------


def construct_reference_cad_image(
    cad_mesh: pv.UnstructuredGrid, ref_img_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """Rasterise a CAD mesh into a synthetic grey-level "silhouette" image.

    The mesh is scaled and centred to fill the image canvas as much as
    possible while preserving its aspect ratio. Each pixel that falls
    inside a mesh cell receives value 1; every other pixel is 0.

    This synthetic image is later aligned to the real DIC speckle image
    using either :func:`register_imgs` (automatic) or
    :func:`register_imgs_manual` (manual).

    Algorithm
    ---------
    1. Copy the input PyVista mesh so the caller's original object is
       left untouched.
    2. Compute the 4x4 transform ``T_cad_ref`` that maps CAD coordinates
       to centred pixel coordinates (see
       :func:`affine_transform_bounding_boxes_2D`).
    3. Apply ``T_cad_ref`` to the copied mesh.
    4. Assign a constant cell-wise value of 1 to every mesh cell (the
       "silhouette" mask).
    5. Sample the transformed mesh onto a uniform per-pixel grid to
       produce the final 2-D image array.

    Args:
        cad_mesh (pv.UnstructuredGrid): CAD mesh in world/physical
            coordinates.
        ref_img_shape (tuple[int, int]): Target image size
            ``(height, width)`` in pixels.

    Returns:
        tuple[NDArray, NDArray]:
            * ``cad_ref_img`` -- Grey-level image of shape
              ``(height, width)`` with values in ``{0, 1}``.
            * ``T_cad_ref`` -- 4x4 homogeneous matrix mapping CAD points
              to the canvas pixel coordinates used to generate the image.
    """
    # Work on a copy so the caller's mesh is never mutated in place
    pv_mesh = cad_mesh.copy()

    hauteur, largeur = ref_img_shape  # (height, width)

    # One grid cell per output pixel; used later to sample/interpolate
    probe_grid = img_uniform_grid((hauteur, largeur))

    # Bounding box of the mesh, in its original CAD coordinates
    bounding_box_cad = get_xy_bounding_box(pv_mesh.points)

    # Destination bounding box = the full image canvas, [0, W] x [0, H]
    bounding_box_img = np.array([[0.0, 0.0], [largeur, hauteur]])

    # Compute the scale + translation needed to centre/fit the mesh in the canvas
    scaling, translation = affine_transform_bounding_boxes_2D(
        bounding_box_cad, bounding_box_img
    )
    T_cad_ref = affine_transform_2D_to_4D(scaling, translation)

    # Move the mesh from CAD space into pixel space
    pv_mesh.transform(T_cad_ref, inplace=True)

    # Paint every cell with a constant value of 1 -- this becomes the
    # "inside the mesh" silhouette mask once sampled onto the pixel grid
    pv_mesh.cell_data["mask"] = np.ones(pv_mesh.n_cells, dtype=np.float64)
    pv_mesh.set_active_scalars("mask")

    # Interpolate/sample the mesh's mask value onto each pixel of the probe grid
    probed_grid = probe_grid.sample(pv_mesh)
    probed_grid.set_active_scalars("mask")

    # Reshape the flat sampled array back to (H, W); pixels that fell
    # outside the mesh come back as NaN, which we convert to 0
    cad_ref_img = np.reshape(probed_grid.active_scalars, (hauteur, largeur))
    cad_ref_img = np.nan_to_num(cad_ref_img, nan=0.0)

    return cad_ref_img, T_cad_ref


# ---------------------------------------------------------------------------
# Stage 2a - automatic DFT-based registration
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
    """Automatically align the CAD silhouette image *img0* onto the speckle image *img1*.

    Both images must share the same pixel dimensions.

    Pre-processing pipeline
    ------------------------
    * *img0* (CAD): Otsu thresholding -> binary silhouette.
    * *img1* (speckle): Gaussian blur (parameter *sigma*) to smooth out
      speckle texture -> Multi-Otsu thresholding -> binary silhouette.
    * Both binaries are optionally down-sampled by *rescale_factor*
      before being handed to the DFT correlator, to speed up registration.

    DFT registration
    -----------------
    ``imreg_dft.similarity`` estimates a similarity transform
    (translation + rotation + uniform scale) between the two images in
    the frequency domain. The search is constrained by *max_angle*
    (maximum rotation) and ``[min_scale, max_scale]`` (allowed scale
    range).

    The function returns the transform that maps the CAD reference image
    onto the real image (consistent with the composition convention used
    in :func:`calibrate_2d`).

    A diagnostic figure showing all intermediate binary images is
    displayed for visual sanity-checking.

    Args:
        img0 (NDArray): CAD silhouette (source), shape ``(H, W)``.
        img1 (NDArray): Real DIC speckle image (target), shape ``(H, W)``.
        max_angle (float): Maximum absolute rotation allowed (degrees).
            Defaults to ``10``.
        min_scale (float): Lower bound on the scale parameter searched by
            the registration. Defaults to ``0.6``.
        max_scale (float): Upper bound on the scale parameter. Defaults
            to ``0.7``.
        rescale_factor (float): Down-sampling factor applied before the
            DFT step to reduce computation time (``1.0`` = no
            down-sampling). Defaults to ``1.0``.
        sigma (float): Standard deviation of the Gaussian blur applied to
            *img1* before thresholding. Larger values merge speckle dots
            into a smoother region. Defaults to ``3.0``.

    Returns:
        NDArray: Shape ``(4, 4)`` transform mapping the CAD reference
        image coordinates onto the real image coordinates.

    Raises:
        AssertionError: If *img0* and *img1* do not share the same shape.
    """
    assert np.shape(img0) == np.shape(img1), (
        "img0 and img1 must have the same shape"
    )

    # ------------------------------------------------------------------
    # Step 1: binarise img0 (CAD silhouette) with a single Otsu threshold
    # ------------------------------------------------------------------
    threshold0 = threshold_otsu(img0)
    binary0 = (img0 >= threshold0).astype(np.uint8) * 255
    binary0_ds = skimage.transform.rescale(binary0, rescale_factor, anti_aliasing=False)

    # ------------------------------------------------------------------
    # Step 2: binarise img1 (real speckle image) -- blur first to remove
    # speckle noise, then use Multi-Otsu to separate specimen from background
    # ------------------------------------------------------------------
    img1_blurred = gaussian(img1, sigma=sigma, preserve_range=True)
    thresholds1 = threshold_multiotsu(img1_blurred)
    # Keep the brightest region (assumed to be the specimen surface)
    binary1 = (img1_blurred >= thresholds1[0]).astype(np.uint8) * 255
    binary1_ds = skimage.transform.rescale(binary1, rescale_factor, anti_aliasing=False)

    # ------------------------------------------------------------------
    # Diagnostic plot: show original + intermediate binary images so the
    # user can visually confirm the silhouettes look reasonable
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img0, cmap="gray")
    axes[0, 0].set_title("img0 - original CAD")
    axes[0, 1].axis("off")  # spacer, kept for grid symmetry with row 2
    axes[0, 2].imshow(binary0_ds, cmap="gray")
    axes[0, 2].set_title(f"img0 - binary (rescale={rescale_factor})")

    axes[1, 0].imshow(img1, cmap="gray")
    axes[1, 0].set_title("img1 - original speckle")
    axes[1, 1].imshow(img1_blurred, cmap="gray")
    axes[1, 1].set_title(f"img1 - Gaussian blur (sigma={sigma})")
    axes[1, 2].imshow(binary1_ds, cmap="gray")
    axes[1, 2].set_title("img1 - binary (Multi-Otsu)")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 3: DFT-based similarity registration.
    # imreg_dft expects (width, height) pixel ordering, hence the .T
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
    # Step 4: convert the imreg_dft result back to full-resolution pixel
    # coordinates (imreg_dft reports translation in down-sampled pixels)
    # ------------------------------------------------------------------
    trans = tform["tvec"] / rescale_factor
    scale = tform["scale"]
    angle = tform["angle"]
    theta = np.deg2rad(angle)

    # Combined 2x2 rotation + uniform scale matrix
    rot_scale = scale * rotation_matrix_2D(theta)

    # imreg_dft rotates/scales about the image centre, not the origin;
    # correct the translation term to account for that pivot point
    h, w = np.shape(img0)
    img_center = np.array([w / 2.0, h / 2.0])
    full_trans = trans + (np.eye(2) - rot_scale) @ img_center

    # Embed the 2-D similarity transform into a 4x4 homogeneous matrix
    T4 = np.array(
        [
            [rot_scale[0, 0], rot_scale[0, 1], 0.0, full_trans[0]],
            [rot_scale[1, 0], rot_scale[1, 1], 0.0, full_trans[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # imreg_dft's "similarity" call estimates the transform that warps
    # img0 into img1's frame, which is exactly the CAD-ref -> real-image
    # mapping we want here, so we return it directly (no inversion).
    return np.linalg.inv(T4)


# ---------------------------------------------------------------------------
# Stage 2b - manual landmark-based registration
# ---------------------------------------------------------------------------


def register_imgs_manual(img0: NDArray, img1: NDArray) -> NDArray:
    """Align *img0* onto *img1* via interactive landmark selection in OpenCV.

    Both images are displayed side-by-side in a single OpenCV window at
    full resolution. The user clicks corresponding structural landmarks:
    first on the left panel (CAD image), then on the right panel (real
    image). At least 3 matched pairs are required to estimate an affine
    transform.

    Controls:
        * Left-click on the left panel to add a CAD landmark.
        * Left-click on the right panel to add the matching real-image
          landmark.
        * ENTER to validate and compute the transform.
        * ESC to cancel (returns the identity matrix).

    Args:
        img0 (NDArray): CAD silhouette image (left panel), shape
            ``(H, W)`` or ``(H, W, 3)``.
        img1 (NDArray): Real DIC image (right panel), same height as
            *img0*.

    Returns:
        NDArray: Shape ``(4, 4)`` homogeneous affine transform mapping
        *img0* pixel coordinates to *img1* pixel coordinates. Returns the
        identity matrix if the user cancels or if fewer than 3 valid
        point pairs are collected.

    Raises:
        ValueError: If *img0* and *img1* do not have the same height
            (required to stack them side-by-side with ``np.hstack``).
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    if h0 != h1:
        raise ValueError(
            f"Les deux images doivent avoir la même hauteur pour np.hstack "
            f"(reçu {h0}px et {h1}px)."
        )

    # Convert float images (assumed in [0, 1]) to uint8 for OpenCV display
    def _to_uint8(arr: NDArray) -> NDArray:
        if arr.dtype == np.uint8:
            return arr.copy()
        return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    base0 = _to_uint8(img0)
    base1 = _to_uint8(img1)

    # Collected full-resolution landmark coordinates, one list per image
    pts0: list[list[int]] = []
    pts1: list[list[int]] = []

    window_name = "Manual Calibration - ENTER to validate, ESC to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _click_event(event: int, x: int, y: int, flags: int, param: object) -> None:
        """Mouse callback: route a click to img0 or img1 based on its x position."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < w0:
                # Click landed in the left (CAD) panel
                pts0.append([x, y])
                print(f"  CAD point #{len(pts0):2d}: ({x}, {y})")
            else:
                # Click landed in the right (real image) panel;
                # subtract w0 to express it in that panel's own coordinates
                real_x = x - w0
                pts1.append([real_x, y])
                print(f"  Real point #{len(pts1):2d}: ({real_x}, {y})")

    cv2.setMouseCallback(window_name, _click_event)

    print("\n--- MANUAL CALIBRATION INSTRUCTIONS ---")
    print("  1. Click a recognisable structural point on the LEFT  (CAD image).")
    print("  2. Click its counterpart on the RIGHT (real DIC image).")
    print("  3. Repeat for at least 3 - 4 point pairs.")
    print("  4. Press ENTER to validate, or ESC to abort.\n")

    # Interactive redraw loop: re-render both panels with their landmarks
    # each frame, until the user presses ENTER or ESC
    while True:
        # Convert to BGR (OpenCV's expected colour order) if grayscale
        vis0 = cv2.cvtColor(base0, cv2.COLOR_GRAY2BGR) if base0.ndim == 2 else base0.copy()
        vis1 = cv2.cvtColor(base1, cv2.COLOR_GRAY2BGR) if base1.ndim == 2 else base1.copy()

        # Draw CAD landmarks in red, numbered in click order
        for idx, pt in enumerate(pts0):
            cv2.circle(vis0, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.putText(vis0, str(idx + 1), (pt[0] + 8, pt[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw real-image landmarks in green, numbered in click order
        for idx, pt in enumerate(pts1):
            cv2.circle(vis1, (pt[0], pt[1]), 5, (0, 255, 0), -1)
            cv2.putText(vis1, str(idx + 1), (pt[0] + 8, pt[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show both panels stacked horizontally in one window
        cv2.imshow(window_name, np.hstack((vis0, vis1)))

        key = cv2.waitKey(30) & 0xFF
        if key == 13:    # Enter key -> validate and proceed
            break
        elif key == 27:  # Escape key -> abort calibration
            print("Calibration cancelled by user.")
            cv2.destroyAllWindows()
            return np.identity(4)

    cv2.destroyAllWindows()

    # Only use as many pairs as both lists have in common (in case of
    # an odd number of total clicks)
    n_pairs = min(len(pts0), len(pts1))
    if n_pairs < 3:
        print(f"[ERROR] Not enough matched pairs (need >= 3, got {n_pairs}).")
        return np.identity(4)

    if len(pts0) != len(pts1):
        print(
            f"[WARNING] Unbalanced point counts: using first {n_pairs} pairs."
        )

    src = np.array(pts0[:n_pairs], dtype=np.float32)
    dst = np.array(pts1[:n_pairs], dtype=np.float32)

    # Robust (RANSAC-based) affine fit: returns a 2x3 matrix, tolerant to
    # a few mis-clicked outlier pairs
    M, _ = cv2.estimateAffine2D(src, dst)

    if M is None:
        print("[ERROR] Affine estimation failed. Returning identity.")
        return np.identity(4)

    # Embed the 2x3 affine matrix into a 4x4 homogeneous matrix
    # (Z axis left as identity, consistent with the rest of the module)
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
    """Compute the CAD-to-image transform using fully automatic DFT registration.

    Main entry point for automated 2-D DIC calibration. Chains
    :func:`construct_reference_cad_image` with :func:`register_imgs` to
    produce the composite transform:

    .. math::

        \\mathbf{T} = \\mathbf{T}_{\\text{ref} \\to \\text{img}} \\;
                      \\mathbf{T}_{\\text{CAD} \\to \\text{ref}}

    Args:
        mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates. Note:
            :func:`construct_reference_cad_image` currently expects a
            ``pyvista.UnstructuredGrid`` -- convert with
            :func:`dolfinx_mesh_to_pv_mesh` first if starting from a raw
            DOLFINx mesh.
        ref_img (NDArray): Reference DIC image (grey-level), shape
            ``(H, W)``.
        min_scale (float): Lower bound on the similarity scale searched
            by ``imreg_dft``. Defaults to ``0.7``.
        max_scale (float): Upper bound on the similarity scale. Defaults
            to ``1.3``.

    Returns:
        NDArray: Shape ``(4, 4)`` transform mapping CAD points to image
        pixels.
    """
    # Stage 1: rasterise the CAD mesh into a synthetic silhouette image
    cad_ref_img, T_cad_ref = construct_reference_cad_image(mesh, np.shape(ref_img))
    # Stage 2: register that synthetic image onto the real DIC image
    T_ref_img = register_imgs(cad_ref_img, ref_img,
                              min_scale=min_scale, max_scale=max_scale)
    # Compose: CAD -> ref image -> real image
    return T_ref_img @ T_cad_ref


def calibrate_2d_manual(mesh: dolfinx.mesh.Mesh, ref_img: NDArray) -> NDArray:
    """Compute the CAD-to-image transform via manual landmark selection.

    Alternative to :func:`calibrate_2d` that does not rely on automatic
    DFT registration. Useful when the automatic approach fails, e.g. for
    very large rotations or unusual specimen shapes.

    Pipeline:
        1. :func:`construct_reference_cad_image` -- rasterise the CAD mesh.
        2. :func:`register_imgs_manual` -- interactive OpenCV point-clicking.
        3. Compose the two transforms.

    Args:
        mesh (dolfinx.mesh.Mesh): CAD mesh in world coordinates. See the
            note in :func:`calibrate_2d` regarding the expected mesh type.
        ref_img (NDArray): Reference DIC image (grey-level), shape
            ``(H, W)``.

    Returns:
        NDArray: Shape ``(4, 4)`` transform mapping CAD points to image
        pixels.
    """
    # Stage 1: rasterise the CAD mesh into a synthetic silhouette image
    cad_ref_img, T_cad_ref = construct_reference_cad_image(mesh, np.shape(ref_img))
    # Stage 2: register that synthetic image onto the real DIC image, manually
    T_ref_img = register_imgs_manual(cad_ref_img, ref_img)
    # Compose: CAD -> ref image -> real image
    return T_ref_img @ T_cad_ref


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------


def check_calibration_2d(
    mesh: pv.UnstructuredGrid,
    ref_img: NDArray,
    tform_cad_to_img: NDArray,
) -> None:
    """Visually check calibration quality in an interactive 3-D PyVista window.

    The DIC reference image is displayed as a flat pixel grid at Z = 0.
    The CAD mesh, after applying *tform_cad_to_img*, is overlaid at
    Z = 1 so it floats visibly above the image. A well-calibrated mesh
    should tightly follow the specimen boundary visible in the
    underlying image.

    Args:
        mesh (pv.UnstructuredGrid): CAD mesh in world coordinates.
        ref_img (NDArray): Reference DIC image (grey-level), shape
            ``(H, W)``.
        tform_cad_to_img (NDArray): Shape ``(4, 4)`` calibration matrix,
            typically the output of :func:`calibrate_2d` or
            :func:`calibrate_2d_manual`.

    Returns:
        None: Opens an interactive PyVista plotting window as a side
        effect.
    """
    h, w = np.shape(ref_img)

    # Build a flat image grid (one cell per pixel) and attach the
    # reference image's grey levels as cell data for display
    img_pv = pv.ImageData(dimensions=(w + 1, h + 1, 1))
    img_pv.cell_data["gray_level"] = ref_img.flatten()

    # Apply the calibration transform to the mesh, then lift it along Z
    # so it renders visibly above (rather than coincident with) the image
    mesh_pv = mesh.transform(tform_cad_to_img, inplace=False)
    mesh_pv = mesh_pv.translate([0.0, 0.0, 1.0], inplace=False)

    # Render both the image and the transformed mesh in the same 3-D scene
    p = pv.Plotter()
    p.add_mesh(img_pv, cmap="gray", show_scalar_bar=False)
    p.add_mesh(mesh_pv, show_edges=True, color="red")
    p.show()