"""
main_test_isotropic.py
======================

Main execution script for FEMU parameter identification.

Purpose
-------
This script coordinates the end-to-end identification of material parameters from 
experimental DIC kinematic fields and machine force acquisitions using Finite 
Element Model Updating (FEMU).

Pipeline overview
------------------
1. Force data preprocessing (optional)
   Reads raw mechanical test force acquisition text files, extracts axial force measurements
   corresponding to target image timesteps, converts values from kN to N, and saves them to
   a NumPy matrix (.npy).

2. Stage 1: DIC CSV series projection onto CAD mesh (optional)
   Projects raw 2-D DIC displacement CSV files onto a 3-D volumetric CAD mesh using the
   inverse camera calibration matrix, generating a series of VTU timestep files and a
   PVD XML manifest.

3. Stage 2: FEMU optimization workflow
   Loads the projected VTU time-series and experimental force data, sets up the 
   FEM plasticity solver (`J2IsotropicHardening`), and calls the optimization toolbox
   (`femu_res_toolbox`) to identify material parameters within bounded constraints.

Outputs
-------
forces_echantillonnees.npy : Exported force vector matrix (N).
dic_series_projected.pvd   : Projected DIC displacement dataset manifest and VTU files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from dic_projection.dic_importation import process_csv_series_to_cad_mesh
from femu_toolbox import femu_res_toolbox
from plasticity_solver_import_bc.plasticity_simu import run_simulation_bc_vtu_fast
from simu_tools import (
    ElasticModel,
    J2IsotropicHardening,
    build_function_spaces,
    get_vtu_files_from_pvd,
    load_domain_from_vtu,
)

# ---------------------------------------------------------------------------
# Pipeline Execution Flags
# ---------------------------------------------------------------------------
force_export = 0  # Set to 1 to extract and export experimental forces from TXT, 0 to skip
project_csv = 0   # Set to 1 to run DIC CSV projection onto CAD, 0 to skip
femu = 1          # Set to 1 to run FEMU parameter identification, 0 to skip


# ---------------------------------------------------------------------------
# 1. Helper: Experimental Force Data Extraction
# ---------------------------------------------------------------------------


def exporter_force_npy(
    fichier_txt: str,
    img_debut: int,
    img_fin: int,
    ech: int,
    fichier_sortie: str = "matrice_force.npy",
) -> np.ndarray:
    """
    Extract axial force values from raw acquisition text files and export to NumPy array.

    Parses text lines, filters first occurrences of specified image indices, converts
    axial force from kilonewtons (kN) to Newtons (N), and exports the resulting column vector.

    Parameters
    ----------
    fichier_txt : str
        Path to the raw machine force acquisition text file.
    img_debut : int
        First image index to extract.
    img_fin : int
        Last image index to extract.
    ech : int
        Subsampling stride / step between target image indices.
    fichier_sortie : str, optional
        Output path for the .npy force matrix (default is "matrice_force.npy").

    Returns
    -------
    matrice_force : np.ndarray
        Column vector array of shape (N, 1) containing extracted force values in Newtons.
    """
    forces = []
    images_traitees = set()

    # Define target image numbers
    images_cibles = set(range(img_debut, img_fin + 1, ech))

    with open(fichier_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and header metadata block
            if not line or any(
                line.startswith(prefix)
                for prefix in ["Chemin", "Essai", "Exécutions", "Date", "Axial", "kN"]
            ):
                continue

            # Standardize decimal separator
            elements = line.replace(",", ".").split()

            if len(elements) >= 5:
                try:
                    force_kN = float(elements[0])
                    nb_image = int(float(elements[4]))

                    # Keep only the first occurrence of each target image index
                    if nb_image in images_cibles and nb_image not in images_traitees:
                        force_N = force_kN * 1000.0  # Convert kN -> N
                        forces.append(force_N)
                        images_traitees.add(nb_image)

                except ValueError:
                    continue

    # Convert to 2-D column vector (N rows, 1 column)
    matrice_force = np.array(forces).reshape(-1, 1)

    # Save to disk (.npy)
    np.save(fichier_sortie, matrice_force)

    print(f"Export complete: {len(forces)} values saved to '{fichier_sortie}'.")
    return matrice_force


if force_export == 1:
    matrice = exporter_force_npy(
        fichier_txt=r"/home/pmarechal/Documents/ESCAL/transfer_12868454_files_9cfdfa8b/machine/Acqui_image - (Niveau delta)1.txt",
        img_debut=9,
        img_fin=5971,
        ech=108,
        fichier_sortie="forces_echantillonnees.npy",
    )

    plt.figure()
    plt.plot(matrice)
    plt.xlabel("Sampled Index")
    plt.ylabel("Force (N)")
    plt.title("Sampled Experimental Force Curve")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# 2. Load 4x4 Calibration Matrix
# ---------------------------------------------------------------------------
# Matrix generated via 2-D calibration module (CAD -> image pixel transformation)

#YOU MUST GENERATE THIS NPY FILE WITH THE test_calibration.py FILE OR AN ANALOGOUS PROGRAM

T = np.load("calibration_matrix_A305.npy")


# ---------------------------------------------------------------------------
# 3. Stage 1 – Project DIC CSV Series onto CAD Mesh
# ---------------------------------------------------------------------------
if project_csv == 1:
    print("[Stage 1] Launching DIC CSV displacement projection onto CAD mesh...")
    process_csv_series_to_cad_mesh(
        folder_path="/home/pmarechal/Documents/A305/A305_rectangle",
        file_prefix="test00",
        mesh_cad_path="/home/pmarechal/Documents/projet_dic/plasticity/A305_COARSE.msh",
        tform_img_to_cad_4D=np.linalg.inv(T),  # Inverse calibration matrix (image -> CAD)
        output_pvd_path="MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        alpha=20.0,
        ech=108,
        start_idx=9,
        end_idx=5791,
    )


# ---------------------------------------------------------------------------
# 4. Stage 2 – FEMU Parameter Identification (J2 Isotropic Material)
# ---------------------------------------------------------------------------
if femu == 1:
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected_A305.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection/forces_sample_A305.npy"

    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    def my_solver(run_cfg: Dict[str, Any]) -> Tuple[np.ndarray, pv.MultiBlock]:
        """
        FEM solver example wrapper for J2 isotropic hardening parameter identification.

        Parameters
        ----------
        run_cfg : Dict[str, Any]
            Configuration dictionary containing current material parameter values.

        Returns
        -------
        f_sim : np.ndarray
            Scaled force vector output from the simulation (N).
        sim_multiblock : pv.MultiBlock
            PyVista MultiBlock containing simulated displacement fields across timesteps.
        """
        model = J2IsotropicHardening(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            Q_var=run_cfg["Q_var"],
            k=run_cfg["k_hardening"],
        )
        f_sim, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)

        return 10 * np.array(f_sim), sim_multiblock

    # Initial guesses and fixed values for model parameters
    Optim_params = {
        "E": 210_000.0,          # Young's modulus (MPa)
        "nu": 0.30,               # Poisson's ratio (fixed)
        "sigma_Y": 200.0,         # Initial yield stress (MPa)
        "Q_var": 50.0,            # Voce isotropic hardening modulus (MPa)
        "k_hardening": 500.0,     # Voce hardening exponent
    }

    # Optimization bounds (parameters not listed here remain fixed)
    Bounds = {
        "Q_var": (50.0, 1000.0),
        "sigma_Y": (10.0, 1000.0),
        "k_hardening": (10.0, 1_500.0),
    }

    config = {
        "pvd_file_path": PVD_FILE,
        "num_steps": 53,
        "T": 3.0,
        "weight_u": 0.0,
        "weight_f": 5.0,
    }

    # Run FEMU identification optimization loop
    result = femu_res_toolbox(
        PVD_file=PVD_FILE,
        FORCE_file=FORCE_FILE,
        Optim_params=Optim_params,
        Bounds=Bounds,
        solver=my_solver,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Identification complete.")
    for name, val in zip(result.param_names, result.x):
        print(f"  {name:>15s} = {val:.4f}")
    if result.fixed_params:
        print("Fixed parameters used:")
        for name, val in result.fixed_params.items():
            print(f"  {name:>15s} = {val}")
    print(f"  Final cost (sum r^2) = {float(np.sum(result.fun ** 2)):.6e}")
    print("=" * 60)