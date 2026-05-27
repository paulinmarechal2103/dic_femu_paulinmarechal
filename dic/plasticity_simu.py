"""
plasticity_simulation.py
------------------------
Elasto-plastic FEM simulation using FEniCSx / dolfinx.

Architecture
~~~~~~~~~~~~
ElasticModel          – linear-elastic constants + UFL primitives
PlasticState (ABC)    – internal-variable storage (subclass per model)
PlasticityModel (ABC) – interface all plasticity models must satisfy
J2IsotropicHardening  – concrete model (current behaviour, unchanged)

Adding a new model
~~~~~~~~~~~~~~~~~~
1. Subclass PlasticState to store whatever internal variables you need.
2. Subclass PlasticityModel and implement:
       create_state(domain, W, WT)  ->  your PlasticState subclass
       update(state, eps)           ->  (delta_p, delta_eps_p)  [UFL]
       commit(state, uh)            ->  None  (update state in-place)
   Optionally override cauchy_stress(state, u) if the default is not suitable.
3. Pass an instance to run_simulation(model=YourModel(...)).

Usage
~~~~~
    python plasticity_simulation.py

Import example::

    from plasticity_simulation import (
        ElasticModel, J2IsotropicHardening,
        build_function_spaces, run_simulation, DEFAULT_CONFIG,
    )
    elastic = ElasticModel(E=200_000, nu=0.3, tdim=3)
    model   = J2IsotropicHardening(elastic, sigma_Y=100, Q_var=50, k=1000)
    forces  = run_simulation(model=model)
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


# ---------------------------------------------------------------------------
# Default simulation configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 2,
    load_amp    = 0.01,       # amplitude of the applied displacement
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name    = "res",
    # Elastic constants (used when no model is supplied)
    E           = 200_000.0,
    nu          = 0.3,
    # J2 isotropic hardening parameters (used when no model is supplied)
    sigma_Y     = 100.0,
    Q_var       = 50.0,
    k_hardening = 1_000.0,
)

# ===========================================================================
# Elastic model
# ===========================================================================
class ElasticModel:
    """
    Linear-elastic constitutive model.

    Stores Lamé constants and exposes UFL expression builders.
    Shared by all plasticity models that need a linear-elastic predictor.

    Parameters
    ----------
    E    : float – Young's modulus
    nu   : float – Poisson ratio
    tdim : int   – spatial dimension of the mesh
    """

    def __init__(self, E: float, nu: float, tdim: int):
        self.E    = E
        self.nu   = nu
        self.tdim = tdim
        self.mu   = E / (2.0 * (1.0 + nu))
        self.lam  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(self, v):
        """Symmetric gradient – linearised strain tensor ε(v)."""
        return 0.5 * (ufl.grad(v) + ufl.grad(v).T)

    def sigma(self, eps):
        """Cauchy stress for a given strain (Hooke's law)."""
        return (
            self.lam * ufl.tr(eps) * ufl.Identity(self.tdim)
            + 2.0 * self.mu * eps
        )

    def sigma_d(self, s):
        """Deviatoric part of a stress (or any symmetric tensor)."""
        return s - (1.0 / 3.0) * ufl.tr(s) * ufl.Identity(self.tdim)

    def von_mises(self, s):
        """von Mises equivalent stress  J = sqrt(3/2 · s_d : s_d)."""
        return ufl.sqrt(1.5 * ufl.inner(self.sigma_d(s), self.sigma_d(s)))


# ===========================================================================
# Plasticity model interface
# ===========================================================================
class PlasticState(ABC):
    """
    Abstract container for internal (history) variables.

    Concrete subclasses must store their internal variables as
    ``fem.Function`` objects so that UFL expressions built from them
    remain valid across time steps (values updated in-place by commit).

    All concrete states are expected to expose at minimum:
        eps_p_old : fem.Function (tensor DG-0) – plastic strain tensor
        p_old     : fem.Function (scalar DG-0) – cumulative plastic strain
    If your model uses different variables, override cauchy_stress() as well.
    """


class PlasticityModel(ABC):
    """
    Abstract base class for plasticity models.

    To implement a new model, subclass this and implement the three
    abstract methods below.  ``cauchy_stress`` has a sensible default
    for standard elastoplastic models (σ = C : (ε – εᵖ)).

    Parameters
    ----------
    elastic : ElasticModel
    """

    def __init__(self, elastic: ElasticModel):
        self.elastic = elastic

    @abstractmethod
    def create_state(self, domain, W, WT) -> PlasticState:
        """
        Allocate and return internal-variable storage for this model.

        Called once before the time loop.

        Parameters
        ----------
        domain : dolfinx.mesh.Mesh
        W      : scalar DG-0 FunctionSpace
        WT     : tensor DG-0 FunctionSpace
        """

    @abstractmethod
    def update(self, state: PlasticState, eps) -> tuple:
        """
        Compute plastic increments from the current total strain.

        Parameters
        ----------
        state : PlasticState  – internal variables at tₙ
        eps   : UFL expression – total strain ε(u) at current Newton iterate

        Returns
        -------
        delta_p     : UFL expression – increment of cumulative plastic strain
        delta_eps_p : UFL expression – increment of plastic strain tensor
        """

    @abstractmethod
    def commit(self, state: PlasticState, uh) -> None:
        """
        Advance internal variables from tₙ to tₙ₊₁.

        Called once per time step, after Newton convergence.
        Must update ``state`` **in-place** (never replace fem.Function objects;
        UFL forms hold references to them).

        Parameters
        ----------
        state : PlasticState
        uh    : fem.Function – converged displacement field
        """

    def cauchy_stress(self, state: PlasticState, u) -> object:
        """
        Cauchy stress  σ = C : (ε(u) – εᵖ_old – Δεᵖ).

        Suitable for standard elastoplastic models.  Override for models
        that require a different stress computation (e.g. viscoplasticity,
        damage, kinematic hardening with back-stress).
        """
        eps = self.elastic.epsilon(u)
        _, delta_eps_p = self.update(state, eps)
        return self.elastic.sigma(eps - (state.eps_p_old + delta_eps_p))


# ===========================================================================
# J2 plasticity with Voce isotropic hardening
# ===========================================================================
class _J2State(PlasticState):
    """Internal variables for J2 isotropic hardening."""

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # cumulative plastic strain p
        self.eps_p_old = fem.Function(WT)  # plastic strain tensor εᵖ
        self._W        = W
        self._WT       = WT


class J2IsotropicHardening(PlasticityModel):
    """
    J2 plasticity with Voce isotropic hardening.

    Yield function:   f(σ, p) = J(σ) – σ_Y – Q·(1 – exp(–k·p))
    Flow rule:        Δεᵖ = Δp · n,   n = (3/2) σ_d / J(σ)
    Return mapping:   one linearised Newton step

    Parameters
    ----------
    elastic  : ElasticModel
    sigma_Y  : float – initial yield stress
    Q_var    : float – saturation stress (isotropic hardening)
    k        : float – hardening rate
    """

    def __init__(self, elastic: ElasticModel, sigma_Y: float, Q_var: float, k: float):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.Q_var   = Q_var
        self.k       = k

    # -- internal helpers (UFL) --------------------------------------------
    def _yield_func(self, sigma, p):
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self.elastic.von_mises(sigma) - self.sigma_Y - R

    def _flow_normal(self, sigma):
        return (3.0 / (2.0 * self.elastic.von_mises(sigma))) * self.elastic.sigma_d(sigma)

    # -- PlasticityModel interface -----------------------------------------
    def create_state(self, domain, W, WT) -> _J2State:
        return _J2State(W, WT)

    def update(self, state: _J2State, eps) -> tuple:
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._yield_func(sigma_tr, state.p_old)
        n        = self._flow_normal(sigma_tr)

        R_prime  = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)
        f_prime  = -R_prime - 3.0 * self.elastic.mu
        delta_p0 = -(1.0 / f_prime) * self._yield_func(sigma_tr, state.p_old)

        # Active only when the elastic predictor is outside the yield surface
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        delta_p     = ufl.sqrt(2.0 / 3.0 * ufl.inner(delta_eps_p, delta_eps_p))
        return delta_p, delta_eps_p

    def commit(self, state: _J2State, uh) -> None:
        eps                  = self.elastic.epsilon(uh)
        delta_p, delta_eps_p = self.update(state, eps)

        state.p_old.interpolate(
            fem.Expression(state.p_old + delta_p, state._W.element.interpolation_points)
        )
        delta_eps_p_proj = fem.Function(state._WT)
        delta_eps_p_proj.interpolate(
            fem.Expression(delta_eps_p, state._WT.element.interpolation_points)
        )
        state.eps_p_old.x.array[:] += delta_eps_p_proj.x.array[:]


# ===========================================================================
# Mesh utilities
# ===========================================================================
def create_mesh(msh, cell_type, prune_z=True):
    """
    Extract a meshio.Mesh of the given *cell_type* from a raw meshio object.
    """
    cells  = msh.get_cells_type(cell_type)
    points = msh.points[:, :2] if prune_z else msh.points
    return meshio.Mesh(points=points, cells={cell_type: cells})


def load_and_write_mesh(mesh_file):
    """
    Read a Gmsh .msh file, write XDMF sub-meshes, return the dolfinx domain.
    """
    if MPI.COMM_WORLD.rank == 0:
        msh           = meshio.read(mesh_file)
        triangle_mesh = create_mesh(msh, "tetra", prune_z=False)
        line_mesh     = create_mesh(msh, "line",  prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf",   line_mesh)

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    return domain


# ===========================================================================
# Function spaces
# ===========================================================================
def build_function_spaces(domain):
    """
    Build the three FEM function spaces.
    """
    tdim = domain.topology.dim
    V    = fem.functionspace(domain, ("CG", 1, (tdim,)))
    W    = fem.functionspace(domain, ("DG", 0))
    WT   = fem.functionspace(domain, ("DG", 0, (tdim, tdim)))
    return V, W, WT


# ===========================================================================
# L2-projection helper
# ===========================================================================
def project(v, target_func, bcs=None):
    """
    L2-project a UFL expression *v* onto *target_func*.
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
# Boundary conditions (Automated boundaries detection)
# ===========================================================================
def build_right_facet_tag(domain):
    """
    Automatically detect the maximum X boundary coordinate and mark its facets with tag 1.

    Returns
    -------
    ds : ufl.Measure restricted to the right boundary
    """
    x_max     = domain.geometry.x[:, 0].max()
    facets    = locate_entities(domain, domain.topology.dim - 1,
                                lambda x: x[0] >= (x_max - 1e-4))
    facet_tag = meshtags(domain, domain.topology.dim - 1,
                         facets, np.full_like(facets, 1))
    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


def dirichlet_bcs(domain, space, disp_value):
    """
    Symmetric tensile BCs applied automatically to the actual min/max X boundaries.

    Parameters
    ----------
    domain     : dolfinx.mesh.Mesh
    space      : vector FunctionSpace V
    disp_value : array-like – displacement amplitude (already time-scaled)
    """
    fdim         = domain.topology.dim - 1
    x_coords     = domain.geometry.x[:, 0]
    x_min, x_max = x_coords.min(), x_coords.max()

    left_facets  = mesh.locate_entities_boundary(domain, fdim, lambda x: x[0] <= (x_min + 1e-4))
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: x[0] >= (x_max - 1e-4))

    bc_left  = fem.dirichletbc(fem.Constant(domain, -disp_value),
                               fem.locate_dofs_topological(space, fdim, left_facets), space)
    bc_right = fem.dirichletbc(fem.Constant(domain,  disp_value),
                               fem.locate_dofs_topological(space, fdim, right_facets), space)
    return [bc_left, bc_right]


# ===========================================================================
# Variational form and Newton solver
# ===========================================================================
def build_solver(domain, V, model: PlasticityModel, state: PlasticState, bcs,
                 quadrature_degree: int = 1):
    """
    Build  ∫ σ(u) : sym∇v dx = 0  and the Newton solver.
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
def run_simulation(config=None, model: PlasticityModel = None):
    """
    Run the elasto-plastic simulation.

    Parameters
    ----------
    config : dict            – keys from DEFAULT_CONFIG to override
    model  : PlasticityModel – plasticity model to use.

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
    domain = load_and_write_mesh(cfg["mesh_file"])

    # ---------------------------------------------------------- output file
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
    disp_value          = np.array((load_amp, 0.1 * load_amp, 0), dtype=PETSc.ScalarType)
    bcs                 = dirichlet_bcs(domain, V, disp_value)
    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds                  = build_right_facet_tag(domain)

    # ------------------------------------------------------------ time loop
    force_vec  = []
    t_paraview = 0
    displ_val  = []

    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    for step in range(num_steps + 1):
        t          += dt
        bcs         = dirichlet_bcs(domain, V, disp_value * t)
        problem.bcs = bcs
       
        solver.solve(uh)

        eps                  = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        # Displacement
        uh.name = "displacement"
        fic.write_function(uh, t_paraview)
        current_displ = uh.x.array.copy()
        displ_val.append(current_displ)

        # Total strain
        eps_proj = fem.Function(WT)
        eps_proj.interpolate(fem.Expression(eps, WT.element.interpolation_points))
        eps_proj.name = "Epsilon"
        fic.write_function(eps_proj, t_paraview)

        # Plastic strain  εᵖ = εᵖ_old + Δεᵖ
        eps_p_proj = fem.Function(WT)
        eps_p_proj.interpolate(
            fem.Expression(delta_eps_p + state.eps_p_old, WT.element.interpolation_points)
        )
        eps_p_proj.name = "Epsilon_p"
        fic.write_function(eps_p_proj, t_paraview)

        # Cumulative plastic strain  p = p_old + Δp
        p_proj = fem.Function(W)
        p_proj.interpolate(
            fem.Expression(delta_p + state.p_old, W.element.interpolation_points)
        )
        p_proj.name = "Cumulative plastic strain"
        fic.write_function(p_proj, t_paraview)

        # Von Mises stress
        stress  = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
        vm_proj = fem.Function(W)
        vm_proj.interpolate(
            fem.Expression(model.elastic.von_mises(stress), W.element.interpolation_points)
        )
        vm_proj.name = "Von Mises stress"
        fic.write_function(vm_proj, t_paraview)

        # Reaction force
        force = fem.assemble_scalar(fem.form(stress[0, 0] * ds(1)))
        force_vec.append(force)

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

        # Stress components tensor
        stress = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))

        components = {
            "sigma_xx": (0, 0),
            "sigma_yy": (1, 1),
            "sigma_zz": (2, 2),
            "sigma_xy": (0, 1),
            "sigma_xz": (0, 2),
            "sigma_yz": (1, 2)
        }

        for name, indices in components.items():
            i, j = indices
            s_comp = fem.Function(W)
            s_comp.name = name
            s_comp.interpolate(fem.Expression(stress[i, j], W.element.interpolation_points))
            fic.write_function(s_comp, t_paraview)

        # Hydrostatic pressure
        pression = fem.Function(W)
        pression.name = "Pressure"
        pression.interpolate(fem.Expression(-1.0/3.0 * ufl.tr(stress), W.element.interpolation_points))
        fic.write_function(pression, t_paraview)

        # Advance internal variables to tₙ₊₁
        model.commit(state, uh)
        t_paraview += 1

    fic.close()
    return force_vec, displ_val


# ===========================================================================
# Main simulation loop -------- V2 -------
# ===========================================================================


def run_simulation_V2(config=None, model: PlasticityModel = None, write_output: bool = True):
    """
    Run the elasto-plastic simulation.

    Parameters
    ----------
    config : dict            – keys from DEFAULT_CONFIG to override
    model  : PlasticityModel – plasticity model to use.
    write_output : bool      – if False, skips all XDMF I/O and field projections

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
    domain = load_and_write_mesh(cfg["mesh_file"])

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
    disp_value          = np.array((load_amp, 0.1 * load_amp, 0), dtype=PETSc.ScalarType)
    bcs                 = dirichlet_bcs(domain, V, disp_value)
    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds                  = build_right_facet_tag(domain)

    # ------------------------------------------------------------ time loop
    force_vec  = []
    displ_val  = []
    t_paraview = 0

    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    for step in range(num_steps + 1):
        t          += dt
        bcs         = dirichlet_bcs(domain, V, disp_value * t)
        problem.bcs = bcs

        solver.solve(uh)

        eps                  = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        current_displ = uh.x.array.copy()
        displ_val.append(current_displ)

        stress = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
        force  = fem.assemble_scalar(fem.form(stress[0, 0] * ds(1)))
        force_vec.append(force)

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

            # Debug logs
            p_incr_proj = fem.Function(W)
            p_incr_proj.interpolate(
                fem.Expression(delta_eps_p[0, 0] - delta_p, W.element.interpolation_points)
            )
            p_incr_proj.name = "Cumulative plastic increment error"
            fic.write_function(p_incr_proj, t_paraview)

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
                "sigma_xx": (0, 0), "sigma_yy": (1, 1), "sigma_zz": (2, 2),
                "sigma_xy": (0, 1), "sigma_xz": (0, 2), "sigma_yz": (1, 2)
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


if __name__ == "__main__":
    forces, deplacements = run_simulation_V2()

    # plt.figure()
    # plt.plot(forces)
    # plt.xlabel("Time step")
    # plt.ylabel("Reaction force")
    # plt.title("Reaction force vs. time step")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
