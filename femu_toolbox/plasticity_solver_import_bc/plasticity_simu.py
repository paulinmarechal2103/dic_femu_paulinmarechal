"""
plasticity_simu.py
------------------
FEM simulation runner that applies Dirichlet boundary conditions derived
from DIC data (imported as a PVD/VTU series), and produces a PyVista
MultiBlock of displacement fields alongside a reaction-force vector.

Public API
~~~~~~~~~~
load_domain_from_vtu(vtu_path, comm)             -> dolfinx.mesh.Mesh
get_vtu_files_from_pvd(pvd_file_path)           -> list[str]
dirichlet_bcs(domain, space, up, down, tol)      -> list[DirichletBC]
run_simulation_bc_vtu_fast(domain, V, W, WT, ...) -> (force_vec, MultiBlock)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
import dolfinx
import dolfinx.plot
from dolfinx.mesh import locate_entities, meshtags
from dolfinx import fem, io, log, mesh
import numpy as np
import pyvista as pv
import meshio
from scipy.spatial import KDTree
import ufl
from dolfinx import fem, log
from mpi4py import MPI
from petsc4py import PETSc

from simu_tools import (
    ElasticModel,
    J2IsotropicHardening,
    PlasticityModel,
    build_function_spaces,
    build_solver,
    build_right_facet_tag,
    animer_deformee,
    DEFAULT_CONFIG,
    get_vtu_files_from_pvd,
    load_domain_from_vtu,
)


# ---------------------------------------------------------------------------
# Dirichlet boundary conditions (upper / lower boundaries along Y axis)
# ---------------------------------------------------------------------------

def dirichlet_bcs_from_vtu(
    domain: dolfinx.mesh.Mesh,
    space: fem.FunctionSpace,
    vtu_file_path: str,
    field_name: str = "displacement_projected",
    tol_rugosite: float = 1e-3,
) -> List[fem.DirichletBC]:
    """
    Construct DOLFINx Dirichlet boundary conditions from projected DIC VTU fields.

    Identifies exterior boundary facets (lateral Ymin/Ymax faces and target Z face),
    projects displacement vectors from the VTU file onto boundary DOFs via KDTree
    nearest-neighbor interpolation, and enforces directional constraints per subspace.

    /!\ This function assumes a 3D domain and vector function space. It computes a boundary 
    condition that fixes Ux and Uy on the Ymin/Ymax faces, and Uz = 0 on one of the Z faces.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        3-D DOLFINx computational mesh domain.
    space : fem.FunctionSpace
        Vector function space for displacement fields.
    vtu_file_path : str
        Path to the input VTU file containing projected displacement data.
    field_name : str, optional
        Name of the displacement field in the VTU point data (default is "displacement_projected").
    tol_rugosite : float, optional
        Geometrical tolerance for selecting boundary facet midpoints (default is 1e-3).

    Returns
    -------
    bcs : List[fem.DirichletBC]
        List of DOLFINx Dirichlet boundary condition objects targeting specific subspaces.

    Raises
    ------
    ValueError
        If the domain/space dimension is less than 3D or if the domain crosses z=0.
    KeyError
        If `field_name` is missing from the VTU file point data.
    """
    tdim = domain.topology.dim  # Dimension topologique (doit être 3D)
    fdim = tdim - 1              # Dimension des facettes
    gdim = domain.geometry.dim  # Dimension géométrique
    vdim = space.dofmap.bs      # Nombre de composantes vectorielles

    if vdim < 3 or gdim < 3:
        raise ValueError("Cette logique de blocage selon Z nécessite un domaine et un espace vectoriel 3D.")

    # 1. Identification des facettes
    y_coords = domain.geometry.x[:, 1]
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    z_coords = domain.geometry.x[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    facet_centers = mesh.compute_midpoints(domain, fdim, boundary_facets)

    # Bords en Y
    left_facets = boundary_facets[facet_centers[:, 1] <= (y_min + tol_rugosite)]
    right_facets = boundary_facets[facet_centers[:, 1] >= (y_max - tol_rugosite)]

    # Bord en Z (sélection selon le signe de Zmin / Zmax)
    if z_min + tol_rugosite > 0:
        z_facets = boundary_facets[facet_centers[:, 2] >= (z_max - tol_rugosite)]
    elif z_max - tol_rugosite < 0:
        z_facets = boundary_facets[facet_centers[:, 2] <= (z_min + tol_rugosite)]
    else:
        raise ValueError(f"Domaine traversant z=0 (z_min={z_min:.2f}, z_max={z_max:.2f}). Définis quelle face Z choisir.")

    # Union de toutes les facettes impactées pour la recherche KDTree
    all_target_facets = np.unique(np.concatenate([left_facets, right_facets, z_facets]))

    # 2. Identification des DOFs
    # Pour Ymin et Ymax : uniquement les composantes X (0) et Y (1)
    left_dofs_x = fem.locate_dofs_topological(space.sub(0), fdim, left_facets)
    left_dofs_y = fem.locate_dofs_topological(space.sub(1), fdim, left_facets)

    right_dofs_x = fem.locate_dofs_topological(space.sub(0), fdim, right_facets)
    right_dofs_y = fem.locate_dofs_topological(space.sub(1), fdim, right_facets)

    # Pour la face Z choisie : uniquement la composante Z (2)
    z_dofs_z = fem.locate_dofs_topological(space.sub(2), fdim, z_facets)

    # 3. Lecture du VTU et projection via KDTree
    try:
        m = meshio.read(vtu_file_path)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture du fichier VTU '{vtu_file_path}' : {e}")

    if field_name not in m.point_data:
        raise KeyError(f"Le champ '{field_name}' n'existe pas dans {vtu_file_path}.")

    points_parent = m.points[:, :gdim]
    vtu_data = m.point_data[field_name]

    # DOFs globaux sur la frontière pour projeter les données du VTU
    boundary_dofs_global = fem.locate_dofs_topological(space, fdim, all_target_facets)

    tree = KDTree(points_parent)
    all_dof_coords = space.tabulate_dof_coordinates()
    boundary_coords = all_dof_coords[boundary_dofs_global, :gdim]

    distances, mapping_indices = tree.query(boundary_coords)
    if np.max(distances) > 1e-5:
        print(f"Attention : Écart spatial max de {np.max(distances):.6e} entre le VTU et les DOFs du bord.")

    # 4. Remplissage du champ u_boundary
    u_boundary = fem.Function(space)
    u_boundary.x.array[:] = 0.0

    for comp in range(vdim):
        dof_indices_flat = boundary_dofs_global * vdim + comp
        u_boundary.x.array[dof_indices_flat] = vtu_data[mapping_indices, comp]

    u_boundary.x.scatter_forward()

    # 5. Application ciblée des conditions de Dirichlet par sous-espace
    bcs = [
        # Sur Ymin : Ux et Uy fixés (Uz libre)
        fem.dirichletbc(u_boundary.sub(0), left_dofs_x),
        fem.dirichletbc(u_boundary.sub(1), left_dofs_y),
        # Sur Ymax : Ux et Uy fixés (Uz libre)
        fem.dirichletbc(u_boundary.sub(0), right_dofs_x),
        fem.dirichletbc(u_boundary.sub(1), right_dofs_y),
        # Sur la face Z ciblée : Uz fixé
        fem.dirichletbc(u_boundary.sub(2), z_dofs_z)
    ]

    return bcs


# ---------------------------------------------------------------------------
# Main simulation runner with DIC-derived BCs
# ---------------------------------------------------------------------------



def run_simulation_bc_vtu_fast(
    domain: dolfinx.mesh.Mesh,
    V: fem.FunctionSpace,
    W: fem.FunctionSpace,
    WT: fem.FunctionSpace,
    config: Optional[Dict[str, Any]] = None,
    coord: int = 1,
    model: Optional[PlasticityModel] = None,
) -> Tuple[List[float], pv.MultiBlock]:
    """
    Execute time-dependent elastoplasticity simulation driven by DIC VTU boundary conditions.

    Iterates through timesteps defined in the PVD file sequence, updates Dirichlet boundary
    conditions from VTU displacement fields, solves non-linear constitutive state updates,
    integrates Y-axis reaction forces, and stores displacement fields in PyVista MultiBlock format.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        3-D computational FEM domain.
    V : fem.FunctionSpace
        Vector displacement function space.
    W : fem.FunctionSpace
        Quadrature tensor space for internal state variables.
    WT : fem.FunctionSpace
        Quadrature scalar space for plastic strain.
    config : Dict[str, Any], optional
        Simulation configuration dictionary containing material constants and file paths.
    coord : int, optional
        Boundary tag coordinate identifier for force assembly (default is 1).
    model : PlasticityModel, optional
        Constitutive material model instance. Defaults to J2 isotropic hardening if None.

    Returns
    -------
    force_vec : List[float]
        Time-series of integrated reaction forces along the boundary.
    displ_multiblock : pv.MultiBlock
        PyVista MultiBlock container containing 3-D displacement grids for each timestep.
    """
    cfg       = {**DEFAULT_CONFIG, **(config or {})}
    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = (cfg["T"] - t) / num_steps
    pvd_file_path = cfg["pvd_file_path"]
    vtu_function_name = cfg.get("vtu_function_name", "displacement_projected")

    vtu_files = get_vtu_files_from_pvd(pvd_file_path)
    
    if len(vtu_files) < num_steps + 1:
        raise ValueError(f"Pas assez de fichiers VTU ({len(vtu_files)}) pour le nombre de pas de temps demandé ({num_steps + 1}).")

    tol = 1e-6#estimate_boundary_tol(domain,1.1)
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


if __name__ == "__main__":
    config = dict(
        t_start     = 0.0,
        T           = 3.0,
        num_steps   = 50,
        E           = 200_000.0,
        nu          = 0.3,
        sigma_Y     = 400.0,
        Q_var       = 200.0,
        k_hardening = 50.0,
        pvd_file_path     = "/home/pmarechal/Documents/projet_dic/femu_toolbox/MAINTEST/pyvista_exports/csv_projection_tensile_iso/dic_series_projected.pvd",
        vtu_function_name = "displacement_projected"
    )
    from time import time
    import matplotlib.pyplot as plt
    start_time = time()
    
    vtu_file = "/home/pmarechal/Documents/projet_dic/femu_toolbox/MAINTEST/pyvista_exports/csv_projection_tensile_iso/dic_series_projected_0009.vtu"
    
    domain = load_domain_from_vtu(vtu_file)
    print(f"Domaine chargé avec succès !")
    print(f"Nombre de cellules : {domain.topology.index_map(domain.topology.dim).size_global}")
    V, W, WT = build_function_spaces(domain)

    forces, multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, config=config)
    end_time = time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print("pas de soucis la team")
    animer_deformee(multiblock, factor=1.0, component=None, fps=20)
    plt.figure()
    plt.plot(forces)
    plt.xlabel("Time step")
    plt.ylabel("Reaction force")
    plt.title("Reaction force vs. time step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

