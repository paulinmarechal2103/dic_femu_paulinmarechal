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

def dirichlet_bcs_from_vtu(domain, space, vtu_file_path, field_name="displacement_projected", loading_axis=1):
    """
    Construit les CL de Dirichlet sur les surfaces hautes et basses via l'orientation des normales.
    Correction : Les composantes X et Y sont interpolées à partir du VTU 2D, tandis que le déplacement
    hors-plan Z est laissé LIBRE. Un point unique est bloqué en Z pour éliminer le mode rigide.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    gdim = domain.geometry.dim

    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0) # Facette -> Sommets
    
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    f_to_v = domain.topology.connectivity(fdim, 0)
    geom_x = domain.geometry.x[:, :gdim]

    top_facets_list = []
    bottom_facets_list = []

    for facet in boundary_facets:
        verts = f_to_v.links(facet)
        nodes = geom_x[verts]
        
        if tdim == 3 and len(nodes) >= 3:
            v1 = nodes[1] - nodes[0]
            v2 = nodes[2] - nodes[0]
            n = np.cross(v1, v2)
        elif tdim == 2 and len(nodes) >= 2:
            v = nodes[1] - nodes[0]
            n = np.array([-v[1], v[0]])
        else:
            continue
            
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n /= n_norm
        else:
            continue
        
        if np.abs(n[loading_axis]) > 1e-6:
            if n[loading_axis] > 0:
                top_facets_list.append(facet)
            else:
                bottom_facets_list.append(facet)

    top_facets = np.array(top_facets_list, dtype=np.int32)
    bottom_facets = np.array(bottom_facets_list, dtype=np.int32)

    # --- 2. Lecture et initialisation de l'interpolation depuis le VTU ---
    import meshio
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    
    try:
        m = meshio.read(vtu_file_path)
    except Exception as e:
        raise RuntimeError(f"Erreur lecture fichier {vtu_file_path}: {e}")

    points_parent = m.points[:, :gdim]
    vtu_data = m.point_data[field_name]

    interpolator = LinearNDInterpolator(points_parent, vtu_data)
    tree = cKDTree(points_parent)

    # --- 3. Séparation par composante (X et Y) ---
    Vx, _ = space.sub(0).collapse()
    Vy, _ = space.sub(1).collapse()

    top_dofs_x = fem.locate_dofs_topological((space.sub(0), Vx), fdim, top_facets)
    top_dofs_y = fem.locate_dofs_topological((space.sub(1), Vy), fdim, top_facets)
    
    bottom_dofs_x = fem.locate_dofs_topological((space.sub(0), Vx), fdim, bottom_facets)
    bottom_dofs_y = fem.locate_dofs_topological((space.sub(1), Vy), fdim, bottom_facets)

    ux_bc = fem.Function(Vx)
    uy_bc = fem.Function(Vy)

    # Interpolation aux coordonnées des degrés de liberté de l'espace effondré
    coords_dofs = Vx.tabulate_dof_coordinates()[:, :gdim]
    ux_values = interpolator(coords_dofs)[:, 0]
    uy_values = interpolator(coords_dofs)[:, 1]

    # Gestion des NaNs éventuels en périphérie (via plus proches voisins)
    if np.isnan(ux_values).any():
        nan_mask = np.isnan(ux_values)
        _, backup_indices = tree.query(coords_dofs[nan_mask])
        ux_values[nan_mask] = vtu_data[backup_indices, 0]

    if np.isnan(uy_values).any():
        nan_mask = np.isnan(uy_values)
        _, backup_indices = tree.query(coords_dofs[nan_mask])
        uy_values[nan_mask] = vtu_data[backup_indices, 1]

    ux_bc.x.array[:] = ux_values
    uy_bc.x.array[:] = uy_values
    ux_bc.x.scatter_forward()
    uy_bc.x.scatter_forward()

    # Génération des CLs uniquement sur X (sub(0)) et Y (sub(1))
    bc_top_x = fem.dirichletbc(ux_bc, top_dofs_x, space.sub(0))
    bc_top_y = fem.dirichletbc(uy_bc, top_dofs_y, space.sub(1))
    
    bc_bottom_x = fem.dirichletbc(ux_bc, bottom_dofs_x, space.sub(0))
    bc_bottom_y = fem.dirichletbc(uy_bc, bottom_dofs_y, space.sub(1))

    bcs = [bc_top_x, bc_top_y, bc_bottom_x, bc_bottom_y]

    # --- 4. Blocage du mode rigide de translation suivant Z ---
    if gdim == 3:
        Vz, _ = space.sub(2).collapse()
        if len(bottom_facets) > 0:
            pin_facet = np.array([bottom_facets[0]], dtype=np.int32)
        else:
            pin_facet = np.array([boundary_facets[0]], dtype=np.int32)
            
        dof_pin_z = fem.locate_dofs_topological((space.sub(2), Vz), fdim, pin_facet)
        
        # CORRECTION ICI : Utilisation d'une Function dédiée sur l'espace effondré Vz
        uz_bc = fem.Function(Vz)
        uz_bc.x.array[:] = 0.0
        
        bc_z_pin = fem.dirichletbc(uz_bc, dof_pin_z, space.sub(2))
        bcs.append(bc_z_pin)

    return bcs


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

    tol = 1
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