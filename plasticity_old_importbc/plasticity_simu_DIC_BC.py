# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
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
import h5py
from plasticity_simu import *
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import dolfinx
import pyvista as pv
import basix.ufl
from time import sleep

# ---------------------------------------------------------------------------
# Default simulation configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name    = "res",
    # Elastic constants (used when no model is supplied)
    E           = 200_000.0,
    nu          = 0.3,
    # J2 isotropic hardening parameters (used when no model is supplied)
    sigma_Y     = 100.0,
    Q_var       = 50.0,
    k_hardening = 1000.0,
)

def load_domain_from_vtu(vtu_path, comm=MPI.COMM_WORLD):
    """
    Lit un fichier .vtu avec PyVista et reconstruit un dolfinx.mesh.Mesh.
    Fonctionne pour les maillages 2D et 3D homogènes.
    """
    import pyvista as pv
    import dolfinx
    import ufl
    
    pv_mesh = pv.read(vtu_path)
    points = pv_mesh.points  # Shape: (N, 3)
    
    cells_dict = pv_mesh.cells_dict
    if not cells_dict:
        raise ValueError("Le fichier VTU ne contient pas de maillage ou de cellules valides.")
        
    if len(cells_dict) > 1:
        raise ValueError("Le maillage contient des types de cellules mixtes, non supportés en natif.")
    
    vtk_type, cells = list(cells_dict.items())[0]
    
    if vtk_type == 5:    # VTK_TRIANGLE
        cell_type = dolfinx.mesh.CellType.triangle
    elif vtk_type == 9:  # VTK_QUAD
        cell_type = dolfinx.mesh.CellType.quadrilateral
    elif vtk_type == 10: # VTK_TETRA
        cell_type = dolfinx.mesh.CellType.tetrahedron
    elif vtk_type == 12: # VTK_HEXAHEDRON
        cell_type = dolfinx.mesh.CellType.hexahedron
    else:
        raise NotImplementedError(f"Le type VTK {vtk_type} n'est pas encore mappé.")
    
    gdim = points.shape[1]
    coordinate_element = basix.ufl.element("Lagrange", cell_type.name, 1, shape=(gdim,))
    domain = dolfinx.mesh.create_mesh(comm, cells, ufl.Mesh(coordinate_element), points)
    return domain

def get_vtu_files_from_pvd(pvd_file_path):
    """
    Parse un fichier .pvd pour extraire la liste ordonnée des fichiers .vtu associés.
    """
    tree = ET.parse(pvd_file_path)
    root = tree.getroot()
    vtu_files = []
    
    base_dir = os.path.dirname(pvd_file_path)
    for dataset in root.iter('DataSet'):
        file_rel_path = dataset.get('file')
        full_path = os.path.join(base_dir, file_rel_path)
        vtu_files.append(full_path)
        
    return vtu_files

def estimate_boundary_tol(domain, safety_factor=1.5):
    """
    Estime une tolérance de capture des DOFs de bord basée sur la taille
    locale des éléments au bord du maillage.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    gdim = domain.geometry.dim

    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)

    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    f_to_v = domain.topology.connectivity(fdim, 0)
    x = domain.geometry.x[:, :gdim]

    h_facets = np.zeros(len(boundary_facets))
    for i, facet in enumerate(boundary_facets):
        verts = f_to_v.links(facet)
        coords = x[verts]
        if len(coords) > 1:
            diffs = coords[:, None, :] - coords[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            h_facets[i] = dists.max()
        else:
            h_facets[i] = 0.0

    h_facets = h_facets[h_facets > 0]
    tol = safety_factor * np.median(h_facets)
    return tol

 

import meshio

def dirichlet_bcs_from_vtu(domain, space, vtu_file_path, field_name="displacement_projected", tol_rugosite=5.0):
    """
    Construit les conditions aux limites de Dirichlet à partir d'un fichier VTU
    en projetant les données via un KDTree sur les bords du maillage.
    
    Compatible avec l'appel :
    dirichlet_bcs_from_vtu(domain, V, vtu_files[0], vtu_function_name, tol)
    """
    tdim = domain.topology.dim  # Dimension topologique
    fdim = tdim - 1              # Dimension des facettes
    gdim = domain.geometry.dim  # Dimension géométrique
    
    # Nombre de composantes vectorielles (ex: 2 pour du 2D, 3 pour du 3D)
    vdim = space.dofmap.bs      

    # 1. Identification des facettes et DOFs limites (filtre sur l'axe Y, indice 1)
    y_coords = domain.geometry.x[:, 1]
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    facet_centers = mesh.compute_midpoints(domain, fdim, boundary_facets)

    left_facets = boundary_facets[facet_centers[:, 1] <= (y_min + tol_rugosite)]
    right_facets = boundary_facets[facet_centers[:, 1] >= (y_max - tol_rugosite)]

    left_dofs = fem.locate_dofs_topological(space, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(space, fdim, right_facets)

    boundary_dofs = np.unique(np.concatenate([left_dofs, right_dofs]))

    # 2. Lecture des données du fichier VTU
    try:
        m = meshio.read(vtu_file_path)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture du fichier VTU '{vtu_file_path}' : {e}")

    if field_name not in m.point_data:
        raise KeyError(f"Le champ '{field_name}' n'existe pas dans {vtu_file_path}. Champs disponibles : {list(m.point_data.keys())}")

    points_parent = m.points[:, :gdim]
    vtu_data = m.point_data[field_name]

    # Sécurité sur la forme des données vectorielles
    if vtu_data.ndim == 1 and vdim == 1:
        vtu_data = vtu_data[:, np.newaxis]
    elif vtu_data.shape[1] < vdim:
        raise ValueError(f"Le champ VTU contient {vtu_data.shape[1]} composante(s), mais l'espace nécessite {vdim} composantes.")

    # 3. Construction du KD-Tree sur le maillage d'origine VTU
    tree = KDTree(points_parent)

    all_dof_coords = space.tabulate_dof_coordinates()
    boundary_coords = all_dof_coords[boundary_dofs, :gdim]

    distances, mapping_indices = tree.query(boundary_coords)

    if np.max(distances) > 1e-5:
        print(f"Attention : Écart spatial max de {np.max(distances):.6e} entre le VTU et les DOFs du bord.")

    # 4. Remplissage ciblé du champ u_boundary
    u_boundary = fem.Function(space)
    u_boundary.x.array[:] = 0.0

    for comp in range(vdim):
        dof_indices_flat = boundary_dofs * vdim + comp
        u_boundary.x.array[dof_indices_flat] = vtu_data[mapping_indices, comp]

    u_boundary.x.scatter_forward()

    # 5. Application des conditions de Dirichlet
    bc_left = fem.dirichletbc(u_boundary, left_dofs)
    bc_right = fem.dirichletbc(u_boundary, right_dofs)

    return [bc_left, bc_right]





def run_simulation_bc_vtu_fast(domain, V, W, WT, config=None, coord=1, model: PlasticityModel = None):
    import pyvista as pv
    import dolfinx.plot
    import numpy as np
    
    cfg       = {**DEFAULT_CONFIG, **(config or {})}
    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = (cfg["T"] - t) / num_steps
    pvd_file_path = cfg["pvd_file_path"]
    vtu_function_name = cfg.get("vtu_function_name", "displacement_projected")

    vtu_files = get_vtu_files_from_pvd(pvd_file_path)
    
    if len(vtu_files) < num_steps + 1:
        raise ValueError(f"Pas assez de fichiers VTU ({len(vtu_files)}) pour le nombre de pas de temps demandé ({num_steps + 1}).")

    tol = estimate_boundary_tol(domain,1.1)
    print(f"Boundary tolerance: {tol}")

    if model is None:
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------- boundary conditions --------
    bcs = dirichlet_bcs_from_vtu(domain, V, vtu_files[0], vtu_function_name, tol)

    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds                  = build_right_facet_tag(domain, coord)

    # ----------------- initialisation PyVista MultiBlock -----------------
    cells, types, x = dolfinx.plot.vtk_mesh(V)
    displ_multiblock = pv.MultiBlock()

    # -------------------------------------------- time loop --------------
    force_vec  = []

    opts = PETSc.Options()
    opts["ksp_monitor"]  = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    for step in range(num_steps + 1):
        if step > 0:
            t += dt
            bcs = dirichlet_bcs_from_vtu(domain, V, vtu_files[step], vtu_function_name, tol)
            problem.bcs = bcs
            
        solver.solve(uh)

        # ---- post-processing for this step ----
        eps                  = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        # ---- création du bloc PyVista pour ce pas de temps ----
        step_grid = pv.UnstructuredGrid(cells, types, x)
        
        bs = V.dofmap.index_map_bs
        d_values = uh.x.array.copy().reshape((-1, bs))
        
        if bs < 3:
            d_values_3d = np.zeros((d_values.shape[0], 3))
            d_values_3d[:, :bs] = d_values
        else:
            d_values_3d = d_values
            
        step_grid.point_data["displacement"] = d_values_3d
        displ_multiblock.append(step_grid)
        displ_multiblock.set_block_name(step, f"step_{step:02d}")

        # ---- calcul de la force de réaction ----
        stress    = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
        force     = fem.assemble_scalar(fem.form(stress[1, 1] * ds(1)))
        force_vec.append(force)

        model.commit(state, uh)
        
    print("p_max =", np.max(state.p_old.x.array), " | p_mean =", np.mean(state.p_old.x.array))

    return force_vec, displ_multiblock


def animer_deformee(multiblock, factor=10.0, component=None, fps=20):
    all_scalars = []
    for block in multiblock:
        disp = block.point_data["displacement"]
        if component is None:
            val = np.linalg.norm(disp, axis=1)
        else:
            val = disp[:, component]
        all_scalars.append(val)

    clim = [min(np.min(v) for v in all_scalars), max(np.max(v) for v in all_scalars)]

    current_mesh = multiblock[0].warp_by_vector("displacement", factor=factor)
    current_mesh["disp_plot"] = all_scalars[0]

    plotter = pv.Plotter()
    title_str = "Deplacement Magnitude (mm)" if component is None else f"Deplacement U_{component}"
    
    plotter.add_mesh(
        current_mesh,
        scalars="disp_plot",
        clim=clim,
        cmap="jet",
        scalar_bar_args={"title": title_str}
    )
    plotter.add_axes()
    plotter.show(interactive_update=True)

    print("Lancement de l'animation... (Faites Ctrl+C dans le terminal pour figer l'image)")
    step = 0
    delai = 1.0 / fps
    
    try:
        while True:
            deformed_step = multiblock[step].warp_by_vector("displacement", factor=factor)
            current_mesh.points = deformed_step.points
            current_mesh["disp_plot"] = all_scalars[step]
            plotter.add_text(f"Pas de temps : {step:02d}", name="step_text", position="upper_left", font_size=12)
            plotter.update()
            sleep(delai)
            
            step += 1
            if step >= len(multiblock):
                step = 0
                
    except KeyboardInterrupt:
        print("\nAnimation interrompue.")

    print("Fenêtre finale active pour manipulation statique.")
    plotter.show()

if __name__ == "__main__":
    config = dict(
        t_start     = 0.0,
        T           = 3.0,
        num_steps   = 50,
        load_amp    = 0.01,
        length      = 10.0,
        mesh_file   = "Flat_specimen_refined.msh",
        output_dir  = "results_plasticity",
        file_name    = "import_bcs",
        E           = 200_000.0,
        nu          = 0.3,
        sigma_Y     = 100.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,
        pvd_file_path     = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        vtu_function_name = "displacement_projected"
    )
    from time import time
    start_time = time()
    
    vtu_file = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected_0000.vtu"
    
    domain = load_domain_from_vtu(vtu_file)
    print(f"Domaine chargé avec succès !")
    print(f"Nombre de cellules : {domain.topology.index_map(domain.topology.dim).size_global}")
    V, W, WT = build_function_spaces(domain)

    forces, multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, config=config)
    end_time = time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print("pas de soucis la team")