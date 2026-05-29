from plasticity_simu import *
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
    k_hardening = 10.0,
)




def create_2D_mesh(msh, cell_type):
    """
    Extract a meshio.Mesh of the given *cell_type* from a raw meshio object.

    Parameters
    ----------
    msh       : meshio.Mesh
    cell_type : str  – 'tetra', 'triangle', 'line', …
    prune_z   : bool – drop the z-coordinate (for 2-D meshes)
    """
    cells  = msh.get_cells_type(cell_type)
    points = msh.points[:, :2]
    return meshio.Mesh(points=points, cells={cell_type: cells})


def load_and_write_2D_mesh(mesh_file):
    """
    Read a Gmsh .msh file, write XDMF sub-meshes, return the dolfinx domain.

    Only rank 0 writes; all ranks read.

    Parameters
    ----------
    mesh_file : str – path to the .msh file

    Returns
    -------
    domain : dolfinx.mesh.Mesh
    """
    if MPI.COMM_WORLD.rank == 0:
        msh           = meshio.read(mesh_file)
        triangle_mesh = create_2D_mesh(msh, "triangle")
        line_mesh     = create_2D_mesh(msh, "line")
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf",   line_mesh)

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    return domain



# ===========================================================================
# L2-projection helper
# ===========================================================================
def project(v, target_func, bcs=None):
    """
    L2-project a UFL expression *v* onto *target_func*.

    Parameters
    ----------
    v           : UFL expression
    target_func : fem.Function – modified in-place
    bcs         : list of DirichletBC (optional)
    """
    if bcs is None:
        bcs = []
    domain = target_func.function_space.mesh
    V      = target_func.function_space
    dx     = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    w, Pv  = ufl.TestFunction(V), ufl.TrialFunction(V)
    a      = fem.form(ufl.inner(Pv, w) * dx)
    L      = fem.form(ufl.inner(v,  w) * dx)

    A = fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.solve(b, target_func.vector)
    return target_func


# ===========================================================================
# Boundary conditions
# ===========================================================================


def dirichlet_bcs_tensile(domain, space, disp_value):
    """
    Symmetric tensile BCs: automatically detects left and right boundaries
    based on the mesh bounding box.
    """
    fdim = domain.topology.dim - 1

    # 1. Détection automatique des bornes géométriques en X
    # On extrait les coordonnées de tous les nœuds du maillage
    x_coords = domain.geometry.x[:, 0]
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)

    # 2. Définition des fonctions de localisation avec une tolérance numérique
    # (indispensable pour éviter les ratés dus aux arrondis machine)
    tol = 1e-6
    left_facets  = mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[0] <= (x_min + tol))
    right_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[0] >= (x_max - tol))

    # 3. Application des conditions aux limites
    bc_left  = fem.dirichletbc(fem.Constant(domain, -disp_value),
                               fem.locate_dofs_topological(space, fdim, left_facets), space)
    bc_right = fem.dirichletbc(fem.Constant(domain,  disp_value),
                               fem.locate_dofs_topological(space, fdim, right_facets), space)
    return [bc_left, bc_right]



#with io.XDMFFile(domain.comm, xdmf_file_path, "r") as xdmf_ref:
import h5py
def dirichlet_bcs(domain, space, h5_file_path, field_name, step):
    fdim = domain.topology.dim - 1
    ref_func = fem.Function(space)
    
    # Lecture brute via h5py
    with h5py.File(h5_file_path, "r") as h5_file:
        dataset_path = f"/Function/{field_name}/{step}" 
        
        # 1. Lecture des données brutes en 3D (N, 3)
        raw_data = h5_file[dataset_path][:]
        
        # 2. Extraction des colonnes X et Y uniquement (N, 2)
        data_2d = raw_data[:, :2]
        
        # 3. Aplatissement en 1D pour correspondre à la structure de ref_func.x.array
        ref_func.x.array[:] = data_2d.flatten()
    
    # Détection des bords
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological(space, fdim, facets)
    
    return [fem.dirichletbc(ref_func, dofs)]


# ===========================================================================
# Variational form and Newton solver
# ===========================================================================
def build_solver(domain, V, model: PlasticityModel, state: PlasticState, bcs,
                 quadrature_degree: int = 1):
    """
    Build  ∫ σ(u) : sym∇v dx = 0  and the Newton solver.

    Parameters
    ----------
    domain            : dolfinx.mesh.Mesh
    V                 : vector CG-1 FunctionSpace
    model             : PlasticityModel
    state             : PlasticState – referenced by the variational form
    bcs               : list of DirichletBC
    quadrature_degree : int

    Returns
    -------
    uh      : fem.Function – displacement solution
    problem : NewtonSolverNonlinearProblem
    solver  : NewtonSolver
    """
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    v  = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    uh = fem.Function(V)

    F = ufl.inner(model.cauchy_stress(state, uh), ufl.sym(ufl.grad(v))) * dx
    J = ufl.derivative(F, uh, du)

    problem               = NewtonSolverNonlinearProblem(F, uh, bcs=bcs, J=J)
    solver                = NewtonSolver(domain.comm, problem)
    solver.atol           = 1e-8
    solver.rtol           = 1e-8
    solver.max_it         = 50
    solver.convergence_criterion = "incremental"
    return uh, problem, solver




# ===========================================================================
# Main simulation loop
# ===========================================================================
def run_simulation_2D(xdmf_bc_ref_path, config=None, model: PlasticityModel = None, write_output: bool = True):
    """
    Run the elasto-plastic simulation.

    Parameters
    ----------
    config : dict            – keys from DEFAULT_CONFIG to override
    model  : PlasticityModel – plasticity model to use.
             Defaults to J2IsotropicHardening built from config values.
    write_output : bool      – if False, skips all XDMF I/O and field projections
                               for maximum speed during FEMU/optimization loops.

    Returns
    -------
    force_vec : list of float – reaction force at the right boundary per step
    displ_val : list of np.ndarray – displacement vector per step
    """
    cfg       = {**DEFAULT_CONFIG, **(config or {})}
    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = (cfg["T"] - t) / num_steps
    load_amp  = cfg["load_amp"]

    # ------------------------------------------------------------------ mesh
    domain = load_and_write_2D_mesh(cfg["mesh_file"])

    # ---------------------------------------------------------- output file
    fic = None
    if write_output:
        os.system(f"rm -rf {cfg['output_dir']}")
        fic = io.XDMFFile(domain.comm, f"{cfg['output_dir']}/{cfg['file_name']}.xdmf", "w")
        fic.write_mesh(domain)

    # -------------------------------------------------------- function spaces
    V, W, WT = build_function_spaces(domain)

    # ------------------------------------------------------- plasticity model
    if model is None:
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------------------------- BCs + solver
    disp_value          = np.array((load_amp, 0.1 * load_amp), dtype=PETSc.ScalarType)
    bcs                 = dirichlet_bcs(domain, V, xdmf_bc_ref_path, "displacement", 0)
    uh, problem, solver = build_solver(domain, V, model, state, bcs)

    # ------------------------------------------------------------ time loop
    force_vec  = []
    displ_val  = []
    t_paraview = 0

    # Silence PETSc/SNES logs for cleaner FEMU output
    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    for step in range(num_steps + 1):
        t          += dt
        print(disp_value * t)
        bcs         = dirichlet_bcs(domain, V, xdmf_bc_ref_path, "displacement", step)
        problem.bcs = bcs

        solver.solve(uh)

        # --- Calculs physiques essentiels (toujours exécutés) ---
        eps                  = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        # Stockage du déplacement
        current_displ = uh.x.array.copy()
        displ_val.append(current_displ)

        # Contrainte & Force de réaction
        stress = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))

        # --------------------------------------------------------------
        # Projections & E/S Paraview (uniquement si write_output=True)
        # --------------------------------------------------------------
        if write_output and fic is not None:
            uh.name = "displacement"
            fic.write_function(uh, t_paraview)

            # Total strain
            eps_proj = fem.Function(WT)
            eps_proj.interpolate(fem.Expression(eps, WT.element.interpolation_points))
            eps_proj.name = "Epsilon"
            fic.write_function(eps_proj, t_paraview)

            # Plastic strain
            eps_p_proj = fem.Function(WT)
            eps_p_proj.interpolate(
                fem.Expression(delta_eps_p + state.eps_p_old, WT.element.interpolation_points)
            )
            eps_p_proj.name = "Epsilon_p"
            fic.write_function(eps_p_proj, t_paraview)

            # Cumulative plastic strain
            p_proj = fem.Function(W)
            p_proj.interpolate(
                fem.Expression(delta_p + state.p_old, W.element.interpolation_points)
            )
            p_proj.name = "Cumulative plastic strain"
            fic.write_function(p_proj, t_paraview)

            # Von Mises stress
            vm_proj = fem.Function(W)
            vm_proj.interpolate(
                fem.Expression(model.elastic.von_mises(stress), W.element.interpolation_points)
            )
            vm_proj.name = "Von Mises stress"
            fic.write_function(vm_proj, t_paraview)

            # Debug: cumulative plastic strain increment error
            p_incr_proj = fem.Function(W)
            p_incr_proj.interpolate(
                fem.Expression(delta_eps_p[0, 0] - delta_p, W.element.interpolation_points)
            )
            p_incr_proj.name = "Cumulative plastic increment error"
            fic.write_function(p_incr_proj, t_paraview)

            # Debug: cumulated plastic error
            p_tot_proj = fem.Function(W)
            p_tot_proj.interpolate(
                fem.Expression(
                    (state.eps_p_old[0, 0] + delta_eps_p[0, 0]) - (state.p_old + delta_p),
                    W.element.interpolation_points,
                )
            )
            p_tot_proj.name = "Cumulated plastic error"
            fic.write_function(p_tot_proj, t_paraview)

            # Stress components
            components = {
                "sigma_xx": (0, 0), 
                "sigma_yy": (1, 1),
                "sigma_xy": (0, 1),
            }
            for name, (i, j) in components.items():
                s_comp = fem.Function(W)
                s_comp.name = name
                s_comp.interpolate(fem.Expression(stress[i, j], W.element.interpolation_points))
                fic.write_function(s_comp, t_paraview)

            # Hydrostatic pressure
            pression = fem.Function(W)
            pression.name = "Pressure"
            pression.interpolate(fem.Expression(-1.0/3.0 * ufl.tr(stress), W.element.interpolation_points))
            fic.write_function(pression, t_paraview)

            t_paraview += 1

        # Advance internal variables to tₙ₊₁
        model.commit(state, uh)

    if fic is not None:
        fic.close()

    return force_vec, displ_val



# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    config = dict(
        mesh_file = "carre.msh",
        output_dir = "results_plasticity",
        file_name = "res",
        num_steps = 5,
    )
    #load_and_write_2D_mesh(config["mesh_file"])
    run_simulation_2D("test_import_bc.h5",config)