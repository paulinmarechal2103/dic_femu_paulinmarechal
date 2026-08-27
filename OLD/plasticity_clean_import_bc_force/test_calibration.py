"""calibration_demo.py - Run 2-D DIC calibration on a real mesh and image.

Usage
-----
Run from the ``plasticity_clean/`` directory (or adjust the paths below):

    python calibration_demo.py

What this script does
---------------------
1. Load the ``astar_6mm`` mesh (XDMF format) from the ``plasticity/`` folder.
2. Load the first DIC reference image (``VK03-1-16-0001_0.tif``).
3. Call :func:`image_calibration.calibrate_2d_manual` which opens an OpenCV
   window - click at least 3 matching points between the CAD silhouette
   (left) and the real speckle image (right), then press ENTER.
4. Print the resulting 4×4 homogeneous transformation matrix.
5. Open a 3-D PyVista window to visually verify the calibration:
   the mesh outline should overlap the specimen in the image.
6. Save the matrix to ``calibration_matrix.npy`` so it can be reused by
   ``dic_importation.py`` without repeating the calibration step.

Switching to automatic calibration
-----------------------------------
Replace the ``calibrate_2d_manual`` call with ``calibrate_2d`` if you prefer
the DFT-based automatic registration.  Tune ``min_scale`` / ``max_scale``
to bracket the expected pixel-to-mm ratio of your setup.

    tform = calibrate_2d(mesh, ref_img, min_scale=0.5, max_scale=1.5)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyvista as pv
import skimage.io
from mpi4py import MPI
from dolfinx import io as dxio

# ---------------------------------------------------------------------------
# Paths - plain string variables using os.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MESH_MSH  = os.path.join(PROJECT_ROOT, "plasticity/A305_COARSE.msh")
REF_IMAGE  = os.path.join("/home/pmarechal/Documents/ESCAL/transfer_12868454_files_9cfdfa8b/face_avant/test000000.tif")
OUTPUT_NPY = os.path.join(SCRIPT_DIR, "calibration_matrix.npy")

# ---------------------------------------------------------------------------
# Import calibration functions
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)

from image_calibration import (
    calibrate_2d,
    calibrate_2d_manual,
    check_calibration_2d,
    dolfinx_mesh_to_pv_mesh,
)

from dic_importation import (
    read_msh_safely,
)

# ---------------------------------------------------------------------------
# 1. Load mesh and convert to PyVista
# ---------------------------------------------------------------------------
print(f"[1/4] Loading mesh: {MESH_MSH}")
if not os.path.exists(MESH_MSH):
    sys.exit(f"ERROR: mesh file not found at {MESH_MSH}")

mesh_pv = read_msh_safely(MESH_MSH)

print(f"      Mesh loaded and converted - {mesh_pv.n_cells} cells")

# ---------------------------------------------------------------------------
# 2. Load reference image
# ---------------------------------------------------------------------------
print(f"[2/4] Loading reference image: {REF_IMAGE}")
if not os.path.exists(REF_IMAGE):
    sys.exit(f"ERROR: reference image not found at {REF_IMAGE}")

ref_img = skimage.io.imread(REF_IMAGE, as_gray=True)

# Normalise to [0, 1] if stored as uint8 or uint16
if ref_img.dtype != np.float64:
    ref_img = ref_img.astype(np.float64)
    ref_img /= ref_img.max()

print(f"      Image loaded - shape {ref_img.shape}, dtype {ref_img.dtype}")

# ---------------------------------------------------------------------------
# 3. Run calibration
# ---------------------------------------------------------------------------
USE_MANUAL = True # Set to False to use automatic DFT approach

print(
    "\n[3/4] Running calibration "
    f"({'MANUAL - click matching points' if USE_MANUAL else 'AUTOMATIC - DFT'})…\n"
)

if USE_MANUAL:
    tform = calibrate_2d_manual(mesh_pv, ref_img)
else:
    tform = calibrate_2d(mesh_pv, ref_img, min_scale=0.5, max_scale=1.5)

# ---------------------------------------------------------------------------
# 4. Print and save the result
# ---------------------------------------------------------------------------
print("\n[4/4] Calibration complete.")
print("\nTransformation matrix T (CAD → image pixels):")
print(tform)

np.save(OUTPUT_NPY, tform)
print(f"\nMatrix saved to: {OUTPUT_NPY}")

# ---------------------------------------------------------------------------
# 5. Visual verification - PyVista 3-D window
# ---------------------------------------------------------------------------
print("\nOpening verification window …")
check_calibration_2d(mesh_pv, ref_img, tform)