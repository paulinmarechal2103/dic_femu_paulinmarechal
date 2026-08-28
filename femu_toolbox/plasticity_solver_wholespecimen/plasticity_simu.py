"""
plasticity_simu.py
------------------
author : Marechal Paulin (marechal.paulin@gmail.com)

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

import dolfinx
import dolfinx.plot
import numpy as np
import pyvista as pv
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
)


# ---------------------------------------------------------------------------
# Dirichlet boundary conditions (upper / lower boundaries along Y axis)
# ---------------------------------------------------------------------------
def dirichlet_bcs(domain, space, disp_value_up, disp_value_down, tol=1e-6):
    """
    Construct Dirichlet boundary conditions on top and bottom boundaries.

    Boundaries are detected dynamically by finding minimum and maximum Y coordinates
    of the domain geometry.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational mesh.
    space : fem.FunctionSpace
        Vector displacement space V.
    disp_value_up : array-like [ux, uy, uz]
        Prescribed displacement vector for upper boundary.
    disp_value_down : array-like [ux, uy, uz]
        Prescribed displacement vector for lower boundary.
    tol : float
        Geometric search tolerance (default: 1e-6).

    Returns
    -------
    list of fem.DirichletBC
        List containing [bc_down, bc_up].
    """
    fdim = domain.topology.dim - 1
    bs   = space.dofmap.index_map_bs   # Block size: 2 in 2D, 3 in 3D

    vec_up   = np.asarray(disp_value_up,   dtype=PETSc.ScalarType)[:bs]
    vec_down = np.asarray(disp_value_down, dtype=PETSc.ScalarType)[:bs]

    # Detect geometry boundaries along Y axis (coordinate index 1)
    y_coords        = domain.geometry.x[:, 1]
    y_min, y_max    = np.min(y_coords), np.max(y_coords)

    # Locate topological facets on bottom and top faces
    down_facets = dolfinx.mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] <= (y_min + tol)
    )
    up_facets = dolfinx.mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] >= (y_max - tol)
    )

    c_down = fem.Constant(domain, vec_down)
    c_up   = fem.Constant(domain, vec_up)

    bc_down = fem.dirichletbc(
        c_down, fem.locate_dofs_topological(space, fdim, down_facets), space
    )
    bc_up = fem.dirichletbc(
        c_up, fem.locate_dofs_topological(space, fdim, up_facets), space
    )
    return [bc_down, bc_up]


# ---------------------------------------------------------------------------
# Main simulation runner with DIC-derived BCs
# ---------------------------------------------------------------------------
def run_simulation_bc_vtu_fast(domain, V, W, WT, config=None, coord=1, model=None):
    """
    Run quasi-static elasto-plastic FE simulation with linearly ramped DIC BCs.

    At each step:
      1. Ramps boundary displacement vectors.
      2. Solves global non-linear mechanical equilibrium via PETSc Newton solver.
      3. Assembles scalar reaction force on top boundary integral ∫ σ_yy ds.
      4. Captures nodal displacement vector field in PyVista MultiBlock.
      5. Commits plastic history variables in-place.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    V, W, WT : fem.FunctionSpace
        Vector displacement (V), scalar history (W), tensor plastic strain (WT).
    config : dict, optional
        Simulation parameters overriding `DEFAULT_CONFIG`.
    coord : int
        Axis index for reaction force boundary integration (default: 1 for Y axis).
    model : PlasticityModel, optional
        Constitutive model instance (defaults to `J2IsotropicHardening` if None).

    Returns
    -------
    force_vec : list of float
        Simulated reaction force at each time step.
    displ_multiblock : pyvista.MultiBlock
        PyVista datasets containing displacement fields per step.
    """
    cfg = {
        "t_start":        0.0,
        "T":              3.0,
        "num_steps":      50,
        "bc_tol":         1e-6,
        "disp_value_up":  [0.0,  0.01, 0.0],
        "disp_value_down":[0.0, -0.01, 0.0],
        **(config or {}),
    }

    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = cfg["T"] / num_steps
    bc_tol    = cfg.get("bc_tol", 1e-6)

    base_up   = np.array(cfg["disp_value_up"],   dtype=PETSc.ScalarType)
    base_down = np.array(cfg["disp_value_down"],  dtype=PETSc.ScalarType)

    # Instantiate default J2 isotropic model if no custom model is supplied
    if model is None:
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # Initialize Dirichlet boundary conditions and Newton solver at initial time t
    bcs                 = dirichlet_bcs(domain, V, base_up * t, base_down * t, tol=bc_tol)
    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds                  = build_right_facet_tag(domain, coord)

    # Prepare VTK mesh metadata for PyVista extraction
    cells, types, x  = dolfinx.plot.vtk_mesh(V)
    displ_multiblock = pv.MultiBlock()

    force_vec = []

    # Configure PETSc options to run quietly
    opts                = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["snes_monitor"]= None
    log.set_log_level(log.LogLevel.ERROR)

    # Time-stepping simulation loop
    for step in range(num_steps + 1):
        current_disp_up   = base_up   * t
        current_disp_down = base_down * t

        # Update Dirichlet boundary values for current timestep
        problem.bcs = dirichlet_bcs(
            domain, V, current_disp_up, current_disp_down, tol=bc_tol
        )
        
        # Solve global non-linear equilibrium
        solver.solve(uh)

        # Post-process state: strain tensor & plastic strain increment
        eps                  = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        # Store nodal displacement field into PyVista UnstructuredGrid
        step_grid = pv.UnstructuredGrid(cells, types, x)
        bs        = V.dofmap.index_map_bs
        d_values  = uh.x.array.copy().reshape((-1, bs))
        
        # Expand 2D displacements to 3D for PyVista visualization compatibility
        if bs < 3:
            d_values_3d         = np.zeros((d_values.shape[0], 3))
            d_values_3d[:, :bs] = d_values
        else:
            d_values_3d = d_values
            
        step_grid.point_data["displacement"] = d_values_3d
        displ_multiblock.append(step_grid)
        displ_multiblock.set_block_name(step, f"step_{step:02d}")

        # Assemble scalar reaction force (integral of σ_yy over upper boundary facet ds(1))
        stress    = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
        force = fem.assemble_scalar(fem.form(stress[1, 1] * ds(1)))
        force_vec.append(force)

        # Commit plastic state variables in-place to step t+dt
        model.commit(state, uh)
        t += dt
        # 1. Vérifier la largeur réelle du maillage (doit afficher 15.0 pour mm)


    print("p_max =", np.max(state.p_old.x.array),
          " | p_mean =", np.mean(state.p_old.x.array))

    return force_vec, displ_multiblock