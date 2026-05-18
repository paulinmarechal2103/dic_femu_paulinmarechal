"""
plasticity_simulation.py
------------------------
Elasto-plastic FEM simulation using FEniCSx / dolfinx.
All public functions can be imported independently.

Usage:
    python plasticity_simulation.py

Import example:
    from plasticity_simulation import (
        create_mesh, build_function_spaces, elastobuild_material_model_plasticity,
        InternalVariables, run_simulation
    )
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
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


# ---------------------------------------------------------------------------
# Simulation parameters  (override before calling run_simulation if needed)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    t_start    = 0.0,
    T          = 3.0,
    num_steps  = 50,
    load_amp   = 0.01,   # amplitude of displacement loading
    length     = 10.0,
    mesh_file  = "Flat_specimen_refined.msh",
    output_dir = "results_plasticity",
    # Material
    E          = 200_000.0,
    nu         = 0.3,
    sigma_Y    = 100.0,
    Q_var      = 50.0,
    k_hardening= 1_000.0,
)


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------
def create_mesh(msh, cell_type, prune_z=True):
    """
    Extract a meshio.Mesh of the given *cell_type* from a raw meshio object.

    Parameters
    ----------
    msh       : meshio.Mesh  – input mesh (e.g. read from a .msh file)
    cell_type : str          – 'tetra', 'triangle', 'line', …
    prune_z   : bool         – drop the z-coordinate (for 2-D meshes)

    Returns
    -------
    meshio.Mesh
    """
    cells  = msh.get_cells_type(cell_type)
    points = msh.points[:, :2] if prune_z else msh.points
    return meshio.Mesh(points=points, cells={cell_type: cells})


def load_and_write_mesh(mesh_file):
    """
    Read a Gmsh .msh file, split it into volume and boundary sub-meshes,
    write them as XDMF, and return the dolfinx domain.

    Only executed on MPI rank 0; all ranks read the resulting XDMF files.

    Parameters
    ----------
    mesh_file : str  – path to the .msh file

    Returns
    -------
    domain : dolfinx.mesh.Mesh
    """
    if MPI.COMM_WORLD.rank == 0:
        msh            = meshio.read(mesh_file)
        triangle_mesh  = create_mesh(msh, "tetra",  prune_z=False)
        line_mesh      = create_mesh(msh, "line",   prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf",   line_mesh)

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    domain.topology.create_connectivity(
        domain.topology.dim, domain.topology.dim - 1
    )
    return domain


# ---------------------------------------------------------------------------
# Function spaces
# ---------------------------------------------------------------------------
def build_function_spaces(domain):
    """
    Build the three function spaces used in the simulation.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh

    Returns
    -------
    V  : CG-1 vector space (displacements)
    W  : DG-0 scalar space  (scalar internal variables)
    WT : DG-0 tensor space  (tensor internal variables)
    """
    tdim = domain.topology.dim
    V  = fem.functionspace(domain, ("CG", 1, (tdim,)))
    W  = fem.functionspace(domain, ("DG", 0))
    WT = fem.functionspace(domain, ("DG", 0, (tdim, tdim)))
    return V, W, WT


# ---------------------------------------------------------------------------
# L2-projection helper
# ---------------------------------------------------------------------------
def project(v, target_func, bcs=None):
    """
    L2-project a UFL expression *v* onto *target_func*.

    Parameters
    ----------
    v           : UFL expression
    target_func : fem.Function  – receives the projected values
    bcs         : list of DirichletBC (optional)

    Returns
    -------
    target_func (modified in-place)
    """
    if bcs is None:
        bcs = []
    domain   = target_func.function_space.mesh
    V        = target_func.function_space
    metadata = {"quadrature_degree": 2}
    dx       = ufl.Measure("dx", domain=domain, metadata=metadata)

    w  = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a  = fem.form(ufl.inner(Pv, w) * dx)
    L  = fem.form(ufl.inner(v,  w) * dx)

    A = fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
    return target_func


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
def build_right_facet_tag(domain, length):
    """
    Create a MeshTags object that marks the right boundary facets with tag 1.
    Used for integrating the reaction force.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    length : float  – half-length of the specimen (boundary at x = +length)

    Returns
    -------
    ds             : ufl.Measure  – surface measure restricted to right boundary
    """
    right_boundary = lambda x: x[0] >= (length - 1e-8)
    facets         = locate_entities(domain, domain.topology.dim - 1, right_boundary)
    facet_markers  = np.full_like(facets, 1)
    facet_tag      = meshtags(domain, domain.topology.dim - 1, facets, facet_markers)
    ds             = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    return ds


def dirichlet_bcs(domain, space, disp_value, length):
    """
    Apply symmetric displacement BCs: left boundary pulled left, right pulled right.

    Parameters
    ----------
    domain     : dolfinx.mesh.Mesh
    space      : fem.FunctionSpace  – vector space V
    disp_value : array-like         – displacement vector (scaled by time later)
    length     : float

    Returns
    -------
    list of fem.DirichletBC
    """
    facet_dim = domain.topology.dim - 1

    left_facets  = mesh.locate_entities_boundary(
        domain, facet_dim, lambda x: x[0] <= (-length + 1e-8)
    )
    right_facets = mesh.locate_entities_boundary(
        domain, facet_dim, lambda x: x[0] >= (+length - 1e-8)
    )

    u_left  = fem.Constant(domain, -disp_value)
    u_right = fem.Constant(domain,  disp_value)

    bc_left  = fem.dirichletbc(u_left,  fem.locate_dofs_topological(space, facet_dim, left_facets),  space)
    bc_right = fem.dirichletbc(u_right, fem.locate_dofs_topological(space, facet_dim, right_facets), space)
    return [bc_left, bc_right]


# ---------------------------------------------------------------------------
# Material model
# ---------------------------------------------------------------------------
def build_material_model_isoelastoplastic(domain, W, E, nu, sigma_Y, Q_var, k_hardening):
    """
    Build UFL expressions for the elasto-plastic material model.

    Parameters
    ----------
    domain      : dolfinx.mesh.Mesh
    W           : scalar DG-0 FunctionSpace
    E, nu       : Young's modulus and Poisson ratio
    sigma_Y     : initial yield stress
    Q_var       : isotropic hardening saturation
    k_hardening : isotropic hardening rate

    Returns
    -------
    dict with keys:
        mu_lame, lambda_lame, sigma_Y, Q_var, k_hardening,
        quant_param (fem.Function),
        epsilon, sigma_hooke, sigma_d, J, yield_func, flow_normal
    """
    mu_lame     = E / (2.0 * (1.0 + nu))
    lambda_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    tdim        = domain.topology.dim

    # Spatially-varying quantisation parameter (unused in default run)
    quant_param = fem.Function(W)
    quant_param.interpolate(
        lambda x: 1.5e-1
        * (1.0e-10 * np.sin((x[0] + x[1]) / 0.25) + 1.0)
        * sigma_Y / mu_lame
    )
    quant_param.name = "quant_param"

    def epsilon(v):
        """Symmetric gradient (strain tensor)."""
        return 0.5 * (ufl.grad(v) + ufl.grad(v).T)

    def sigma_hooke(eps):
        """Linear-elastic Cauchy stress."""
        return (
            lambda_lame * ufl.tr(eps) * ufl.Identity(tdim)
            + 2.0 * mu_lame * eps
        )

    def sigma_d(sigma):
        """Deviatoric part of a stress tensor."""
        return sigma - (1.0 / 3.0) * ufl.tr(sigma) * ufl.Identity(tdim)

    def J(sigma):
        """von Mises equivalent stress (J2 invariant)."""
        return ufl.sqrt((3.0 / 2.0) * ufl.inner(sigma_d(sigma), sigma_d(sigma)))

    def yield_func(sigma, p):
        """Yield function f(σ, p) = J(σ) - σ_Y - R(p)."""
        R_p = Q_var * (1.0 - ufl.exp(-k_hardening * p))
        return J(sigma) - sigma_Y - R_p

    def flow_normal(sigma):
        """Unit outward normal to the yield surface."""
        return (3.0 / (2.0 * J(sigma))) * sigma_d(sigma)

    return dict(
        mu_lame     = mu_lame,
        lambda_lame = lambda_lame,
        sigma_Y     = sigma_Y,
        Q_var       = Q_var,
        k_hardening = k_hardening,
        quant_param = quant_param,
        epsilon     = epsilon,
        sigma_hooke = sigma_hooke,
        sigma_d     = sigma_d,
        J           = J,
        yield_func  = yield_func,
        flow_normal = flow_normal,
    )


# ---------------------------------------------------------------------------
# Internal variables
# ---------------------------------------------------------------------------
class InternalVariables:
    """
    Container for the plastic internal variables at the previous time step.

    Attributes
    ----------
    p_old     : fem.Function (scalar DG-0)  – cumulative plastic strain
    eps_p_old : fem.Function (tensor DG-0)  – plastic strain tensor
    """
    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)
        self.eps_p_old = fem.Function(WT)


# ---------------------------------------------------------------------------
# Plasticity return-mapping
# ---------------------------------------------------------------------------
def update_internal_variables(mat, iv, eps):
    """
    Closest-point return-mapping (one Newton step, linear hardening).

    Parameters
    ----------
    mat : dict      – material model dict from elastobuild_material_model_plasticity()
    iv  : InternalVariables
    eps : UFL expr  – total strain ε(u)

    Returns
    -------
    delta_p     : UFL expr  – increment of cumulative plastic strain
    delta_eps_p : UFL expr  – increment of plastic strain tensor
    """
    sigma_star = mat["sigma_hooke"](eps - iv.eps_p_old)
    f_val      = mat["yield_func"](sigma_star, iv.p_old)
    n          = mat["flow_normal"](sigma_star)

    R_prime  = mat["Q_var"] * (mat["k_hardening"] * ufl.exp(-mat["k_hardening"] * iv.p_old))
    f_prime  = -R_prime - 3.0 * mat["mu_lame"]
    delta_p0 = -(1.0 / f_prime) * mat["yield_func"](sigma_star, iv.p_old)

    # Active only if f > 0 (elastic predictor outside yield surface)
    delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
    delta_p     = ufl.sqrt(2.0 / 3.0 * ufl.inner(delta_eps_p, delta_eps_p))
    return delta_p, delta_eps_p


def cauchy_stress(mat, iv, u):
    """
    Cauchy stress accounting for plastic strain.

    Parameters
    ----------
    mat : dict
    iv  : InternalVariables
    u   : fem.Function  – displacement

    Returns
    -------
    UFL expression for σ
    """
    eps = mat["epsilon"](u)
    _, delta_eps_p = update_internal_variables(mat, iv, eps)
    return mat["sigma_hooke"](eps - (iv.eps_p_old + delta_eps_p))


# ---------------------------------------------------------------------------
# Variational form and Newton solver
# ---------------------------------------------------------------------------
def build_solver(domain, V, mat, iv, bcs, quadrature_degree=1):
    """
    Build the nonlinear residual, Jacobian, and Newton solver.

    Parameters
    ----------
    domain            : dolfinx.mesh.Mesh
    V                 : vector CG-1 FunctionSpace
    mat               : material dict
    iv                : InternalVariables
    bcs               : list of DirichletBC
    quadrature_degree : int (default 1)

    Returns
    -------
    uh      : fem.Function  – solution (displacement)
    problem : NonlinearProblem
    solver  : NewtonSolver
    """
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    v  = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    uh = fem.Function(V)

    F  = ufl.inner(cauchy_stress(mat, iv, uh), ufl.sym(ufl.grad(v))) * dx
    J  = ufl.derivative(F, uh, du)

    problem        = NewtonSolverNonlinearProblem(F, uh, bcs=bcs, J=J)
    solver         = NewtonSolver(domain.comm, problem)
    solver.atol    = 1e-8
    solver.rtol    = 1e-8
    solver.max_it  = 50
    solver.convergence_criterion = "incremental"
    return uh, problem, solver


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_simulation(config=None):
    """
    Run the full elasto-plastic simulation.

    Parameters
    ----------
    config : dict  – override any key from DEFAULT_CONFIG

    Returns
    -------
    force_vec : list of float  – reaction force at the right boundary per step
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    t          = cfg["t_start"]
    T          = cfg["T"]
    num_steps  = cfg["num_steps"]
    dt         = (T - t) / num_steps
    load_amp   = cfg["load_amp"]
    length     = cfg["length"]
    output_dir = cfg["output_dir"]

    # --- Mesh -----------------------------------------------------------
    domain = load_and_write_mesh(cfg["mesh_file"])
    print("Mesh topology dim:", domain.topology.dim)

    # --- Output file ----------------------------------------------------
    os.system(f"rm -rf {output_dir}")
    fic = io.XDMFFile(domain.comm, f"{output_dir}/res.xdmf", "w")
    fic.write_mesh(domain)

    # --- Function spaces ------------------------------------------------
    V, W, WT = build_function_spaces(domain)

    # --- Material model -------------------------------------------------
    mat = build_material_model_isoelastoplastic(
        domain, W,
        E          = cfg["E"],
        nu         = cfg["nu"],
        sigma_Y    = cfg["sigma_Y"],
        Q_var      = cfg["Q_var"],
        k_hardening= cfg["k_hardening"],
    )
    fic.write_function(mat["quant_param"])

    # --- Internal variables ---------------------------------------------
    iv = InternalVariables(W, WT)

    # --- Boundary conditions (initial) ----------------------------------
    disp_value = np.array((load_amp, 0.1 * load_amp, 0), dtype=PETSc.ScalarType)
    bcs        = dirichlet_bcs(domain, V, disp_value, length)

    # --- Solver ---------------------------------------------------------
    uh, problem, solver = build_solver(domain, V, mat, iv, bcs)

    # --- Right-boundary surface measure (reaction force) ----------------
    ds = build_right_facet_tag(domain, length)

    # --- Time loop ------------------------------------------------------
    force_vec  = []
    t_paraview = 0

    for step in range(num_steps + 1):
        print(f"--- Step {step} ---")
        t += dt
        bcs         = dirichlet_bcs(domain, V, disp_value * t, length)
        problem.bcs = bcs
        print("  disp_value * t =", disp_value * t)

        log.set_log_level(log.LogLevel.INFO)
        solver.solve(uh)
        log.set_log_level(log.LogLevel.WARNING)

        eps                = mat["epsilon"](uh)
        delta_p, delta_eps_p = update_internal_variables(mat, iv, eps)

        # Displacement
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
            fem.Expression(delta_eps_p + iv.eps_p_old, WT.element.interpolation_points)
        )
        eps_p_proj.name = "Epsilon_p"
        fic.write_function(eps_p_proj, t_paraview)

        # Cumulative plastic strain
        p_proj = fem.Function(W)
        p_proj.interpolate(
            fem.Expression(delta_p + iv.p_old, W.element.interpolation_points)
        )
        p_proj.name = "Cumulative plastic strain"
        fic.write_function(p_proj, t_paraview)

        # Von Mises stress
        stress       = mat["sigma_hooke"](eps - (delta_eps_p + iv.eps_p_old))
        von_mises    = mat["J"](stress)
        vm_proj      = fem.Function(W)
        vm_proj.interpolate(fem.Expression(von_mises, W.element.interpolation_points))
        vm_proj.name = "Von Mises stress"
        fic.write_function(vm_proj, t_paraview)

        # Reaction force
        force = fem.assemble_scalar(fem.form(stress[0, 0] * ds(1)))
        force_vec.append(force)
        print("  Reaction force:", force)

        # Cumulative plastic strain increment error (debug)
        p_incr      = delta_eps_p[0, 0] - delta_p
        p_incr_proj = fem.Function(W)
        p_incr_proj.interpolate(fem.Expression(p_incr, W.element.interpolation_points))
        p_incr_proj.name = "Cumulative plastic increment error"
        fic.write_function(p_incr_proj, t_paraview)

        # Cumulated plastic error (debug)
        p_tot      = (iv.eps_p_old[0, 0] + delta_eps_p[0, 0]) - (iv.p_old + delta_p)
        p_tot_proj = fem.Function(W)
        p_tot_proj.interpolate(fem.Expression(p_tot, W.element.interpolation_points))
        p_tot_proj.name = "Cumulated plastic error"
        fic.write_function(p_tot_proj, t_paraview)

        # Update internal variables for next step
        iv.p_old.interpolate(
            fem.Expression(iv.p_old + delta_p, W.element.interpolation_points)
        )
        delta_eps_p_proj = fem.Function(WT)
        delta_eps_p_proj.interpolate(
            fem.Expression(delta_eps_p, WT.element.interpolation_points)
        )
        iv.eps_p_old.x.array[:] += delta_eps_p_proj.x.array[:]

        t_paraview += 1
        print("  t_paraview =", t_paraview)

    fic.close()
    print("Simulation complete.")
    print("Reaction forces:", force_vec)
    return force_vec


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    force_vec = run_simulation()

    plt.figure()
    plt.plot(force_vec)
    plt.xlabel("Time step")
    plt.ylabel("Reaction force")
    plt.title("Reaction force vs. time step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()