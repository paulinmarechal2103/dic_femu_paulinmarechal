import math
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import os
from abc import ABC, abstractmethod

import meshio
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import pyvista as pv
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import binary_closing, disk

"""
import dlf_dic.dic.p1_interpolation_jit
from dlf_dic.dic.calibration import kinematic_sensitivity
"""


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




def read_geof(file_path):
    zset_to_vtk = {
        'c2d3': pv.CellType.TRIANGLE,
        'c2d4': pv.CellType.QUAD,
        'c3d4': pv.CellType.TETRA,
        'c3d8': pv.CellType.HEXAHEDRON,
        'c3d10': pv.CellType.QUADRATIC_TETRA,
        'c3d20': pv.CellType.QUADRATIC_HEXAHEDRON,
    }

    nodes = []
    cells = []
    cell_types = []
    nsets = {} # Dictionnaire pour stocker les groupes temporairement

    with open(file_path, 'r') as f:
        lines = f.readlines()

    it = iter(lines)
    nb_nodes = 0

    for line in it:
        clean = line.strip()
        if not clean: continue

        if clean.startswith('**node'):
            h = next(it).split()
            nb_nodes, dim = int(h[0]), int(h[1])
            for _ in range(nb_nodes):
                p = next(it).split()
                coords = [float(x) for x in p[1:]]
                if dim == 2: coords.append(0.0)
                nodes.append(coords)

        elif clean.startswith('**element'):
            nb_el = int(next(it).strip())
            for _ in range(nb_el):
                p = next(it).split()
                indices = [int(n) - 1 for n in p[2:]]
                cells.append(len(indices))
                cells.extend(indices)
                cell_types.append(zset_to_vtk.get(p[1], pv.CellType.POLYGON))

        elif clean.startswith('**nset'):
            name = clean.split()[-1]
            # On lit la ligne suivante qui contient les IDs des noeuds
            ids = [int(n) - 1 for n in next(it).split()]
            mask = np.zeros(len(nodes) if nodes else nb_nodes, dtype=bool)
            mask[ids] = True
            nsets[name] = mask

    grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types, dtype=np.uint8), np.array(nodes))
    
    # Ajout des nsets dans l'objet
    for name, mask in nsets.items():
        grid.point_data[name] = mask

    return grid



#grid = read_geof("carre.geof")
#print(f"Nombre de cellules : {grid.n_cells}")
#grid.save("carre.vtk")




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



def construct_reference_cad_image_from_pv_mesh(
    pv_mesh: pv.UnstructuredGrid, 
    ref_img_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """
    Construit une image de référence à partir d'un mesh PyVista (fenicsx).
    """
    # 1. Préparation de la grille de sondage (image)
    # Attention : ref_img_shape est souvent (largeur, hauteur)
    w, h = ref_img_shape
    # Création d'une grille uniforme PyVista pour représenter les pixels
    # origin=(0,0,0), spacing=(1,1,1), dimensions=(w, h, 1)
    probe_grid = pv.ImageData(dimensions=(w, h, 1))

    # 2. Calcul de la transformation pour caler le mesh dans l'image
    # On récupère les coordonnées (X, Y) du mesh geof
    coords_xy = pv_mesh.points[:, :2]
    min_xy = coords_xy.min(axis=0)
    max_xy = coords_xy.max(axis=0)
    
    # Bounding box actuelle [[xmin, ymin], [xmax, ymax]]
    bounding_box_xy = np.array([min_xy, max_xy])
    
    # Bounding box cible (les pixels de l'image)
    # On laisse une petite marge si nécessaire, ici on prend toute l'image
    bounding_box_img = np.array([[0.0, 0.0], [float(w), float(h)]])

    # Utilisation de tes fonctions utilitaires (scaling / translation)
    scaling, translation = affine_transform_bounding_boxes_2D(
        bounding_box_xy, bounding_box_img
    )
    transformation_4D = affine_transform_2D_to_4D(scaling, translation)

    # 3. Application de la transformation sur le mesh
    # On travaille sur une copie pour ne pas modifier le mesh original
    pv_mesh_transformed = pv_mesh.copy()
    pv_mesh_transformed.transform(transformation_4D, inplace=True)

    # 4. "Peinture" du maillage
    # Au lieu d'une fonction DG0 dolfinx, on crée un tableau de 1.0 
    # pour toutes les cellules du maillage Z-set
    pv_mesh_transformed.cell_data["dummy"] = np.ones(pv_mesh_transformed.n_cells)
    pv_mesh_transformed.set_active_scalars("dummy")

    # 5. Échantillonnage (Sampling)
    # On projette le maillage sur la grille de pixels
    # On utilise 'categorical' car on veut du 0 ou 1 (masque)
    probed_grid = probe_grid.sample(pv_mesh_transformed, categorical_2_point=True)
    
    # 6. Extraction de l'image
    # Les données samplées sont dans point_data
    # On reshape pour retrouver le format image (W, H)
    raw_scalars = probed_grid.point_data["dummy"]
    # Remplacement des NaN (zones hors mesh) par 0
    raw_scalars[np.isnan(raw_scalars)] = 0.0
    
    cad_ref_img = raw_scalars.reshape((h, w)).T # Transpose selon ton axe DIC
    
    return cad_ref_img, transformation_4D

