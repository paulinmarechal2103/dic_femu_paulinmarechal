"""
_                                              /WWWWWWWWWW
/WWWWWWWWW-           |W##\                       /W| -WWWW@\
    |WW| -WWWW@_      |W| \#\                     |W|      -@@\
    |WW|      @#\     |W|   \#\                   |W|        /@
    |WW|   /WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW@WWWWW-
    |WW|     _-@@#/   |#|     \#\        |R|      |W#WW\
/@WWWWWWWWWWWW@+      |W|       \#\      |R|      |W|  @W\        /+\
    |WW|              |W|         +/     |R|      |W|    WW\     /WW|
    |WW|              |W|                |R|      |W|      @W\  /W/|W|
    |WW|              \_'                |R|      `+         \#`#/ |W|
    |WW|                                 |R|                /@@/ \@\|W|
    |WW|                                 |R|              /@@/     WWW|
    |WW|                                 |R|            /@@/       |WW@\
    \+                                   |R|           /@@/         |W| \@\
     \                                    +/                        |A|   \@

Paulin MARECHAL
paulin.marechal@minesparis-psl.eu

------------------------------------
plasticity_simulation_bc_h5.py
------------------------------------
Elasto-plastic FEM simulation using FEniCSx / dolfinx, driven by Dirichlet
boundary conditions read from an external HDF5 displacement field (e.g. a
DIC / digital-image-correlation measurement or a prior simulation export)
instead of an analytically prescribed loading.
 
Architecture
~~~~~~~~~~~~
dirichlet_bcs_from_h5_interpolate  – BCs via scattered-data (LinearNDInterpolator)
                                      mapping of an H5 displacement field onto
                                      the current mesh; robust to non-matching
                                      meshes, but expensive (full-domain fit).
dirichlet_bcs_from_h5              – BCs via KD-Tree nearest-neighbour mapping
                                      restricted to the boundary DOFs only;
                                      the fast path used inside run_simulation.
                                      Must use matching meshes.
run_simulation_bc_h5_write         – full time-stepping loop with per-step
                                      XDMF/H5 field export (displacement,
                                      strain, plastic strain, von Mises
                                      stress, stress components, pressure,
                                      plus debug consistency checks).
run_simulation_bc_h5_fast          – lean time-stepping loop with no file
                                      I/O, intended for FEMU / optimisation
                                      inner loops where only the displacement 
                                      matter.
 
Relies on plasticity_simu.py functions.
 
Usage
~~~~~
Populate a config dict (see DEFAULT_CONFIG) with, at minimum, "h5_bc_path"
(path to the HDF5 file holding the boundary displacement field) and
"h5_function_path" (HDF5 dataset group storing that field), then call
run_simulation_bc_h5_fast(...) or run_simulation_bc_h5_write(...) with the
domain and function spaces built from your mesh.
"""



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

# ---------------------------------------------------------------------------
# Default simulation configuration
# ---------------------------------------------------------------------------
# All keys can be overridden by passing a partial dict to run_simulation_*.
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


def dirichlet_bcs_from_h5_interpolate(domain, space, h5_file_path, t_index, dataset_path="/Function/displacement_projected"):
    """
    Build Dirichlet boundary conditions on the rough top/bottom surfaces by
    interpolating a displacement field stored in an HDF5 file (defined on the
    original/parent mesh) onto the current (possibly non-matching) domain.

    This is the "scattered-data interpolation" variant: it builds one
    ``scipy.interpolate.LinearNDInterpolator`` per spatial component from the
    parent mesh node coordinates and displacement values, then evaluates it
    at the DOF coordinates of the target function space. This is more
    expensive but works even if the target DOFs do not exactly coincide with
    parent mesh nodes.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The current computational domain on which boundary conditions are
        to be imposed.
    space : dolfinx.fem.FunctionSpace
        The (vector-valued) displacement function space associated with
        ``domain``.
    h5_file_path : str
        Path to the HDF5 file containing the parent mesh geometry and the
        time-series of the source displacement field (e.g. an XDMF/H5
        export).
    t_index : int
        Index of the time step / dataset to read from
        ``dataset_path/{t_index}`` inside the HDF5 file.
    dataset_path : str, optional
        Base HDF5 group path under which the per-time-step displacement
        datasets are stored. Defaults to
        ``"/Function/displacement_projected"``.

    Returns
    -------
    list[dolfinx.fem.DirichletBC]
        A two-element list ``[bc_left, bc_right]`` containing the Dirichlet
        boundary condition objects for the low-Y ("left"/bottom) and
        high-Y ("right"/top) boundary surfaces, both driven by the
        interpolated displacement field.

    Notes
    -----
    - Boundary facets are identified geometrically: any exterior facet whose
      midpoint lies within ``tol_rugosite`` of the global Y-extent (min or
      max) of the domain is classified as belonging to the bottom or top
      surface, respectively. The tolerance must be larger than the local
      surface roughness amplitude.
    - The HDF5 file must expose the parent mesh node coordinates under
      ``"Mesh/Grid/geometry"`` (typical of a classic XDMF export); if this
      key is missing, a ``KeyError`` is raised since interpolation is not
      possible without the source coordinates.
    - Any DOF falling outside the convex hull of the parent point cloud
      yields ``NaN`` from ``LinearNDInterpolator``; such values are replaced
      with ``0.0`` as a safety fallback before the BC is applied.
    """
    fdim = domain.topology.dim - 1
    gdim = domain.geometry.dim

    # 1. Robust identification of the top/bottom surface facets.
    # Since the surface is very rough, facets are targeted based on the
    # extreme Y-coordinates of their barycenters rather than a fixed normal.
    x_coords = domain.geometry.x[:, 1]
    y_min, y_max = np.min(x_coords), np.max(x_coords)

    # Tolerance adapted to the local surface roughness (must exceed the
    # roughness amplitude).
    tol_rugosite = 5.0

    domain.topology.create_connectivity(fdim, gdim)
    # Topological identification of the boundary facets.
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Compute the midpoint of each facet to filter by location.
    facet_centers = mesh.compute_midpoints(domain, fdim, boundary_facets)

    left_facets = boundary_facets[facet_centers[:, 1] <= (y_min + tol_rugosite)]
    right_facets = boundary_facets[facet_centers[:, 1] >= (y_max - tol_rugosite)]

    left_dofs = fem.locate_dofs_topological(space, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(space, fdim, right_facets)

    # 2. Read the original (parent) mesh coordinates and displacements from
    # the HDF5 file.
    with h5py.File(h5_file_path, "r") as f:
        # NOTE: it is essential to also read the parent mesh point
        # coordinates. Adjust the paths below to match the exact structure
        # of your HDF5 file (e.g. an XDMF export).
        try:
            points_parent = f["Mesh/Grid/geometry"][:]  # typical path
        except KeyError:
            # If absent, the parent domain must be reconstructed by other
            # means before calling this function.
            raise KeyError("Le fichier H5 doit contenir les coordonnées des nœuds d'origine pour l'interpolation.")

        full_path = f"{dataset_path}/{t_index}"
        h5_data = f[full_path][:]  # Shape: (N_nodes_parent, gdim)

    # 3. Temporary reconstruction of the parent field to serve as the
    # interpolation source. Rather than reconstructing the whole (possibly
    # large) parent mesh, a non-matching-mesh (scattered-data) interpolation
    # is used here via scipy.

    u_boundary = fem.Function(space)

    # A more robust option would be to use the DOLFINx interpolation
    # utilities if the parent mesh object is directly available. Here, a
    # scipy-based (Griddata-style) approach is used because the meshes no
    # longer match topologically:

    # Build one interpolator per component (X, Y, and Z if 3-D).
    interpolators = [LinearNDInterpolator(points_parent[:, :gdim], h5_data[:, comp]) for comp in range(gdim)]

    # Retrieve the coordinates of the DOFs of the current displacement
    # space (V).
    dof_coords = space.tabulate_dof_coordinates()

    # Fill the u_boundary array component by component.
    bs = space.dofmap.bs
    for comp in range(gdim):
        # Evaluate the interpolator at the DOF coordinates of the target
        # (sub-)mesh.
        u_boundary.x.array[comp::bs] = interpolators[comp](dof_coords[:, :gdim])

    # Replace any out-of-hull NaNs with 0.0 as a safety fallback.
    np.nan_to_num(u_boundary.x.array, copy=False, nan=0.0)
    u_boundary.x.scatter_forward()

    # 4. Apply the Dirichlet boundary conditions.
    bc_left = fem.dirichletbc(u_boundary, left_dofs)
    bc_right = fem.dirichletbc(u_boundary, right_dofs)

    return [bc_left, bc_right]


def dirichlet_bcs_from_h5(domain, space, f, t_index, dataset_path="/Function/displacement_projected"):
    """
    Build Dirichlet boundary conditions on the rough top/bottom surfaces by
    directly mapping displacements from an already-open HDF5 file handle
    onto the boundary DOFs of the current domain, using a KD-Tree
    nearest-neighbour search restricted to the boundary DOFs.

    Compared to :func:`dirichlet_bcs_from_h5_interpolate`, this variant is
    the fast/lean path suitable for use inside a time-stepping loop: instead
    of building a global scattered-data interpolator over the whole domain,
    it only performs a nearest-neighbour lookup for the (much smaller) set
    of boundary DOFs, and it expects the HDF5 file to already be open
    (avoiding repeated file-open overhead across time steps).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The current computational domain on which boundary conditions are
        to be imposed.
    space : dolfinx.fem.FunctionSpace
        The (vector-valued) displacement function space associated with
        ``domain``.
    f : h5py.File
        An already-open HDF5 file handle containing the parent mesh
        geometry and the time-series of the source displacement field.
    t_index : int
        Index of the time step / dataset to read from
        ``dataset_path/{t_index}`` inside the HDF5 file.
    dataset_path : str, optional
        Base HDF5 group path under which the per-time-step displacement
        datasets are stored. Defaults to
        ``"/Function/displacement_projected"``.

    Returns
    -------
    list[dolfinx.fem.DirichletBC]
        A two-element list ``[bc_left, bc_right]`` containing the Dirichlet
        boundary condition objects for the low-Y ("left"/bottom) and
        high-Y ("right"/top) boundary surfaces, both driven by the mapped
        displacement field.

    Notes
    -----
    - Boundary facets are identified geometrically in the same way as in
      :func:`dirichlet_bcs_from_h5_interpolate`: exterior facets whose
      midpoint lies within ``tol_rugosite`` of the global Y-extent are
      classified as bottom or top surface facets.
    - A single KD-Tree is built over the full parent point cloud
      (``Mesh/Grid/geometry``) and queried only for the boundary DOF
      coordinates, which is significantly cheaper than a full-domain
      interpolation.
    - If the maximum nearest-neighbour distance across the boundary DOFs
      exceeds ``1e-5``, a warning is printed, since this indicates that the
      current boundary DOFs do not closely coincide with parent mesh nodes
      (the mapping is then only an approximation).
    - All DOFs are first zero-initialized; only the identified boundary DOFs
      are subsequently overwritten with values gathered from the HDF5
      dataset via the KD-Tree mapping.
    """
    fdim = domain.topology.dim - 1
    gdim = domain.geometry.dim

    # 1. Identification of the facets and DOFs of the top/bottom surfaces.
    x_coords = domain.geometry.x[:, 1]
    y_min, y_max = np.min(x_coords), np.max(x_coords)
    tol_rugosite = 5.0

    domain.topology.create_connectivity(fdim, gdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    facet_centers = mesh.compute_midpoints(domain, fdim, boundary_facets)

    left_facets = boundary_facets[facet_centers[:, 1] <= (y_min + tol_rugosite)]
    right_facets = boundary_facets[facet_centers[:, 1] >= (y_max - tol_rugosite)]

    left_dofs = fem.locate_dofs_topological(space, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(space, fdim, right_facets)

    # Union of all boundary DOFs of interest, used for the geometric lookup.
    boundary_dofs = np.unique(np.concatenate([left_dofs, right_dofs]))

    # 2. Read the data from the HDF5 file.
    try:
        points_parent = f["Mesh/Grid/geometry"][:]
    except KeyError:
        raise KeyError("Le fichier H5 doit contenir les coordonnées des nœuds d'origine.")

    full_path = f"{dataset_path}/{t_index}"
    h5_data = f[full_path][:]  # Shape: (N_nodes_parent, gdim)

    # 3. Build the KD-Tree over the full parent HDF5 mesh.
    # (We search the whole parent point cloud for the points matching our
    # current boundary DOFs.)
    tree = KDTree(points_parent[:, :gdim])

    # Retrieve the coordinates of ALL DOFs, then extract those on our
    # boundaries of interest.
    all_dof_coords = space.tabulate_dof_coordinates()
    boundary_coords = all_dof_coords[boundary_dofs, :gdim]

    # KD-Tree query restricted to the boundary DOF coordinates.
    distances, mapping_indices = tree.query(boundary_coords)

    if np.max(distances) > 1e-5:
        print(f"Attention: Écart max constaté de {np.max(distances)} sur les frontières.")

    # 4. Targeted filling of the u_boundary field.
    u_boundary = fem.Function(space)

    # By default the whole field is 0.0 (interior DOFs remain at 0).
    u_boundary.x.array[:] = 0.0

    bs = space.dofmap.bs
    # Apply displacements only to the boundary DOFs.
    for comp in range(gdim):
        # Real indices into the flat DOLFINx array.
        dof_indices_flat = boundary_dofs * bs + comp

        # Direct assignment of the corresponding HDF5 values.
        u_boundary.x.array[dof_indices_flat] = h5_data[mapping_indices, comp]

    u_boundary.x.scatter_forward()

    # 5. Apply the Dirichlet boundary conditions.
    bc_left = fem.dirichletbc(u_boundary, left_dofs)
    bc_right = fem.dirichletbc(u_boundary, right_dofs)

    return [bc_left, bc_right]


def run_simulation_bc_h5_write(domain, V, W, WT, config=None, coord=1,
                         model: PlasticityModel = None, write_output: bool = True):
    """
    Run the elasto-plastic simulation and (optionally) write fields to XDMF/H5.

    Combines the lightweight time-stepping of run_simulation_V3 with the
    field-projection and Paraview-export capabilities of an earlier version.

    Output fields (when write_output=True)
    ------------------------------------------
    - displacement
    - total strain tensor ε
    - plastic strain tensor εᵖ
    - cumulative plastic strain p
    - von Mises stress
    - cumulative plastic increment error  (debug: Δεᵖ[coord,coord] – Δp)
    - cumulated plastic error             (debug: εᵖ[coord,coord] – p)
    - stress components (σ_xx, σ_yy, σ_xy in 2-D; + σ_zz, σ_xz, σ_yz in 3-D)
    - hydrostatic pressure  p_h = –tr(σ)/3

    Parameters
    ----------
    domain       : dolfinx.mesh.Mesh
    V, W, WT     : fem.FunctionSpace – displacement, scalar DG-0, tensor DG-0
    config       : dict, optional – keys from DEFAULT_CONFIG to override
    coord        : int – loading direction (0 = X, 1 = Y)
    model        : PlasticityModel, optional
    write_output : bool
        If False, all XDMF/H5 I/O and field projections are skipped for
        maximum throughput (suitable for FEMU / optimisation inner loops).

    Returns
    -------
    force_vec : list of float
    displ_val : list of np.ndarray
    """
    cfg       = {**DEFAULT_CONFIG, **(config or {})}
    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = (cfg["T"] - t) / num_steps
    h5_bc_path = cfg["h5_bc_path"]
    h5_function_path = cfg["h5_function_path"]

    # ---------------------------------------- output file ----------------
    fic = None
    if write_output:
        import os
        # Remove any previous results directory to avoid stale data
        if os.path.exists(cfg['output_dir']):
            os.system(f"rm -rf {cfg['output_dir']}")
        os.makedirs(cfg['output_dir'], exist_ok=True)
        fic = io.XDMFFile(domain.comm, f"{cfg['output_dir']}/{cfg['file_name']}.xdmf", "w")
        fic.write_mesh(domain)

    # ----------------------------------------- build plasticity model ----
    if model is None:
        # NOTE: domain.topology.dim replaced with domain.geometry.dim.
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.geometry.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------- boundary conditions --------
    gdim = domain.geometry.dim
    with h5py.File(h5_bc_path, "r") as f:
        bcs = dirichlet_bcs_from_h5(domain, V,f, 0, h5_function_path)

        uh, problem, solver = build_solver(domain, V, model, state, bcs)
        ds                  = build_right_facet_tag(domain, coord)

        # -------------------------------------------- time loop --------------
        force_vec  = []
        displ_val  = []
        t_paraview = 0  # integer frame index for the XDMF time axis

        opts = PETSc.Options()
        opts["ksp_monitor"]  = None
        opts["snes_monitor"] = None
        log.set_log_level(log.LogLevel.ERROR)

        for step in range(num_steps + 1):
            print(step)
            t      += dt
            bcs = dirichlet_bcs_from_h5(domain, V, f, step, h5_function_path)

            problem.bcs = bcs
            solver.solve(uh)

            # ---- mandatory post-processing (always executed) ----
            eps                  = model.elastic.epsilon(uh)
            delta_p, delta_eps_p = model.update(state, eps)

            current_displ = uh.x.array.copy()
            displ_val.append(current_displ)

            stress = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
            force  = fem.assemble_scalar(fem.form(stress[1, 1] * ds(1)))
            force_vec.append(force)

            # ---- optional XDMF / Paraview output (skipped when write_output=False) ----
            if write_output and fic is not None:

                # Displacement field
                uh.name = "displacement"
                fic.write_function(uh, t_paraview)

                # Total strain tensor ε
                eps_proj = fem.Function(WT)
                eps_proj.interpolate(fem.Expression(eps, WT.element.interpolation_points))
                eps_proj.name = "Epsilon"
                fic.write_function(eps_proj, t_paraview)

                # Plastic strain tensor εᵖ = εᵖ_old + Δεᵖ
                eps_p_proj = fem.Function(WT)
                eps_p_proj.interpolate(
                    fem.Expression(delta_eps_p + state.eps_p_old, WT.element.interpolation_points)
                )
                eps_p_proj.name = "Epsilon_p"
                fic.write_function(eps_p_proj, t_paraview)

                # Cumulative plastic strain  p + Δp
                p_proj = fem.Function(W)
                p_proj.interpolate(
                    fem.Expression(delta_p + state.p_old, W.element.interpolation_points)
                )
                p_proj.name = "Cumulative plastic strain"
                fic.write_function(p_proj, t_paraview)

                # von Mises equivalent stress
                vm_proj = fem.Function(W)
                vm_proj.interpolate(
                    fem.Expression(model.elastic.von_mises(stress), W.element.interpolation_points)
                )
                vm_proj.name = "Von Mises stress"
                fic.write_function(vm_proj, t_paraview)

                # Debug: difference between diagonal plastic strain increment and Δp
                #   Should be ≈ 0 in uniaxial tension; non-zero in multiaxial states
                p_incr_proj = fem.Function(W)
                p_incr_proj.interpolate(
                    fem.Expression(delta_eps_p[coord, coord] - delta_p, W.element.interpolation_points)
                )
                p_incr_proj.name = "Cumulative plastic increment error"
                fic.write_function(p_incr_proj, t_paraview)

                # Debug: cumulated version of the above error
                p_tot_proj = fem.Function(W)
                p_tot_proj.interpolate(
                    fem.Expression(
                        (state.eps_p_old[coord, coord] + delta_eps_p[coord, coord]) - (state.p_old + delta_p),
                        W.element.interpolation_points,
                    )
                )
                p_tot_proj.name = "Cumulated plastic error"
                fic.write_function(p_tot_proj, t_paraview)

                # Individual stress components (dimension-aware)
                if gdim == 2:
                    components = {
                        "sigma_xx": (0, 0), "sigma_yy": (1, 1), "sigma_xy": (0, 1)
                    }
                else:
                    components = {
                        "sigma_xx": (0, 0), "sigma_yy": (1, 1), "sigma_zz": (2, 2),
                        "sigma_xy": (0, 1), "sigma_xz": (0, 2), "sigma_yz": (1, 2)
                    }

                for name, (i, j) in components.items():
                    s_comp      = fem.Function(W)
                    s_comp.name = name
                    s_comp.interpolate(fem.Expression(stress[i, j], W.element.interpolation_points))
                    fic.write_function(s_comp, t_paraview)

                # Hydrostatic pressure  p_h = –tr(σ)/3
                pression      = fem.Function(W)
                pression.name = "Pressure"
                pression.interpolate(fem.Expression(-1.0/3.0 * ufl.tr(stress), W.element.interpolation_points))
                fic.write_function(pression, t_paraview)

                t_paraview += 1  # advance the XDMF time counter

            # Advance history variables to tₙ₊₁
            model.commit(state, uh)

    if fic is not None:
        fic.close()

    return force_vec, displ_val


def run_simulation_bc_h5_fast(domain, V, W, WT, config=None, coord=1, model: PlasticityModel = None):
    """
    Run the elasto-plastic simulation (lean version – no file output).

    This is the core time-stepping loop.  It is designed to be called
    repeatedly inside optimisation / FEMU loops where I/O overhead must be
    minimised.  All field projections and XDMF writes are skipped.

    Algorithm per step
    ------------------
    1. Increment time and update the Dirichlet boundary conditions.
    2. Solve the nonlinear equilibrium equations (Newton iterations).
    3. Evaluate the stress and integrate the reaction force on the tagged boundary.
    4. Call model.commit to advance the internal variables to tₙ₊₁.

    Parameters
    ----------
    domain  : dolfinx.mesh.Mesh
    V, W, WT : fem.FunctionSpace – displacement, scalar DG-0, tensor DG-0
    config  : dict, optional
        Keys from DEFAULT_CONFIG to override.  Missing keys fall back to defaults.
    coord   : int
        Loading direction (0 = X, 1 = Y).  Controls which axis the displacement
        BCs are applied along and where the reaction force is integrated.
    model   : PlasticityModel, optional
        Plasticity model instance.  If None, a J2IsotropicHardening
        model is built from the config values.

    Returns
    -------
    force_vec : list of float
        Reaction force (integrated stress component) at the loaded boundary,
        one entry per time step.
    displ_val : list of np.ndarray
        Copy of the displacement DOF vector at each time step.
    """
    cfg       = {**DEFAULT_CONFIG, **(config or {})}
    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = (cfg["T"] - t) / num_steps
    h5_bc_path = cfg["h5_bc_path"]
    h5_function_path = cfg["h5_function_path"]

    fic = None  # no output file in this variant

    # ----------------------------------------- build plasticity model ----
    if model is None:
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------- boundary conditions --------
    gdim = domain.geometry.dim
    with h5py.File(h5_bc_path, "r") as f:
        bcs = dirichlet_bcs_from_h5(domain, V,f, 0, h5_function_path)

        uh, problem, solver = build_solver(domain, V, model, state, bcs)
        ds                  = build_right_facet_tag(domain, coord)

        # -------------------------------------------- time loop --------------
        force_vec  = []
        displ_val  = []

        opts = PETSc.Options()
        opts["ksp_monitor"]  = None
        opts["snes_monitor"] = None
        log.set_log_level(log.LogLevel.ERROR)

        for step in range(num_steps + 1):
            print(step)
            t      += dt
            bcs = dirichlet_bcs_from_h5(domain, V, f, step, h5_function_path)

            problem.bcs = bcs
            solver.solve(uh)

            # ---- post-processing for this step ----
            eps                  = model.elastic.epsilon(uh)
            delta_p, delta_eps_p = model.update(state, eps)

            current_displ = uh.x.array.copy()
            displ_val.append(current_displ)

            # Integrate the normal stress component over the tagged boundary
            stress    = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
            force     = fem.assemble_scalar(fem.form(stress[1, 1] * ds(1)))
            force_vec.append(force)

            # Advance history variables to tₙ₊₁
            model.commit(state, uh)

    if fic is not None:
        fic.close()

    return force_vec, displ_val


if __name__ == "__main__":
    # ---- simulation configuration ----------------------------------------
    config = dict(
        t_start     = 0.0,
        T           = 3.0,
        num_steps   = 50,
        load_amp    = 0.01,       # amplitude of the applied displacement
        length      = 10.0,       # half-length of the specimen
        mesh_file   = "Flat_specimen_refined.msh",
        output_dir  = "results_plasticity",
        file_name    = "import_bcs",
        # Elastic constants
        E           = 200_000.0,
        nu          = 0.3,
        # J2 isotropic hardening parameters
        sigma_Y     = 100.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,
        h5_bc_path = "results/ref_astar.h5",
        h5_function_path = "/Function/displacement"
    )
    from time import time
    start_time = time()
    # ---- load mesh and build function spaces -----------------------------
    with io.XDMFFile(MPI.COMM_WORLD, "results/projection_cad_temporelle_mask_volume_interet.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
    V, W, WT = build_function_spaces(domain)

    # ---- run simulation with relaxation phase ----------------------------
    forces, _ = run_simulation_bc_h5_fast(domain, V, W, WT, config=config)
    end_time = time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print("pas de soucis la team")

    # ---- plot reaction force vs. time step -------------------------------
    plt.figure()
    plt.plot(forces)
    plt.xlabel("Time step")
    plt.ylabel("Reaction force")
    plt.title("Reaction force vs. time step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     with h5py.File("results/resampling_time_series_linear.h5", "r") as f:
#         def print_struct(name, obj):
#             if isinstance(obj, h5py.Dataset):
#                 print(f"Dataset: {name} | Shape: {obj.shape}")
#         f.visititems(print_struct)