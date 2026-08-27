"""
run_calibration.py
==================

Execution script for 2-D Digital Image Correlation (DIC) camera calibration.

Purpose
-------
This script loads a 3-D volumetric mesh (.msh) and a 2-D DIC reference image (.tif),
computes the 4x4 homogeneous transformation matrix `T` that maps CAD coordinates
to camera pixel positions, exports the resulting matrix to a NumPy array (.npy),
and launches an interactive 3-D visual verification window.

Pipeline overview
------------------
1. Mesh loading & conversion
   Reads the volumetric Gmsh mesh via `read_msh_safely` into a PyVista grid.

2. Image loading & normalisation
   Loads the reference speckle image (`skimage.io.imread`), casts to `float64`,
   and normalises pixel intensities to [0, 1].

3. Calibration execution
   Executes either manual landmark selection (`calibrate_2d_manual`) or automatic
   DFT-based registration (`calibrate_2d`) based on the `USE_MANUAL` toggle.

4. Matrix export & verification
   Saves matrix `T` to disk (`calibration_matrix.npy`) and launches `check_calibration_2d`
   for visual inspection of the alignment.

Outputs
-------
calibration_matrix.npy : 4x4 homogeneous transformation matrix (CAD → image pixels).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pyvista as pv
import skimage.io

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MESH_MSH = os.path.join(PROJECT_ROOT, "plasticity/A305_COARSE.msh")
REF_IMAGE = os.path.join(
    "/home/pmarechal/Documents/ESCAL/transfer_12868454_files_9cfdfa8b/face_avant/test000000.tif"
)
OUTPUT_NPY = os.path.join(SCRIPT_DIR, "calibration_matrix.npy")

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)

from dic_projection.dic_importation import read_msh_safely
from dic_projection.image_calibration import (
    calibrate_2d,
    calibrate_2d_manual,
    check_calibration_2d,
)

# ---------------------------------------------------------------------------
# 1. Load CAD/FEM mesh
# ---------------------------------------------------------------------------
print(f"[1/4] Loading Gmsh mesh: {MESH_MSH}")
if not os.path.exists(MESH_MSH):
    sys.exit(f"ERROR: Mesh file not found at {MESH_MSH}")

mesh_pv = read_msh_safely(MESH_MSH)
print(f"      Mesh successfully converted to PyVista: {mesh_pv.n_cells} cells")

# ---------------------------------------------------------------------------
# 2. Load and normalise reference image
# ---------------------------------------------------------------------------
print(f"[2/4] Loading DIC reference image: {REF_IMAGE}")
if not os.path.exists(REF_IMAGE):
    sys.exit(f"ERROR: Reference image not found at {REF_IMAGE}")

ref_img = skimage.io.imread(REF_IMAGE, as_gray=True)

# Normalise intensity values to [0, 1] if required
if ref_img.dtype != np.float64:
    ref_img = ref_img.astype(np.float64)
    ref_img /= ref_img.max()

print(f"      Image loaded successfully: shape {ref_img.shape}, dtype {ref_img.dtype}")

# ---------------------------------------------------------------------------
# 3. Compute 2-D DIC calibration matrix
# ---------------------------------------------------------------------------
USE_MANUAL = True  # Toggle: True for interactive landmark pairing, False for automatic DFT

mode_label = "MANUAL (interactive landmark selection)" if USE_MANUAL else "AUTOMATIC (DFT registration)"
print(f"\n[3/4] Running calibration pipeline: {mode_label}...\n")

if USE_MANUAL:
    tform = calibrate_2d_manual(mesh_pv, ref_img)
else:
    tform = calibrate_2d(mesh_pv, ref_img, min_scale=0.5, max_scale=1.5)

# ---------------------------------------------------------------------------
# 4. Export transformation matrix
# ---------------------------------------------------------------------------
print("\n[4/4] Calibration complete.")
print("\nTransformation matrix T (CAD -> image pixel coordinates):")
print(tform)

np.save(OUTPUT_NPY, tform)
print(f"\nMatrix successfully saved to: {OUTPUT_NPY}")

# ---------------------------------------------------------------------------
# 5. Interactive 3-D visual inspection
# ---------------------------------------------------------------------------
print("\nOpening 3-D verification window...")
check_calibration_2d(mesh_pv, ref_img, tform)