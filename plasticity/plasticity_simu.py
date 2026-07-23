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

------------------------
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

# config_test = dict(
#     t_start     = 0.0,
#     T           = 5.0,
#     num_steps   = 50,
#     load_amp    = 0.01,       # amplitude of the applied displacement
#     length      = 10.0,       # half-length of the specimen
#     mesh_file   = "Flat_specimen_refined.msh",
#     output_dir  = "results_plasticity",
#     # Elastic constants (used when no model is supplied)
#     E           = 200_000.0,
#     nu          = 0.3,
#     # J2 isotropic hardening parameters (used when no model is supplied)
#     sigma_Y     = 1000.0,
#     Q_var       = 50.0,
#     k_hardening = 1_000.0,
# )


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
    E    : float – Young's modulus  [MPa or consistent units]
    nu   : float – Poisson ratio    [dimensionless, in (0, 0.5)]
    tdim : int   – spatial dimension of the mesh (2 or 3)

    Computed attributes
    -------------------
    mu  : float – shear modulus        μ = E / (2(1+ν))
    lam : float – first Lamé constant  λ = Eν / ((1+ν)(1-2ν))
    """

    def __init__(self, E: float, nu: float, tdim: int):
        self.E    = E
        self.nu   = nu
        self.tdim = tdim
        # Derived Lamé parameters used throughout the constitutive relations
        self.mu   = E / (2.0 * (1.0 + nu))
        self.lam  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(self, v):
        """
        Symmetric gradient – linearised strain tensor ε(v).

        ε(v) = ½ (∇v + ∇vᵀ)

        Parameters
        ----------
        v : UFL function – displacement field

        Returns
        -------
        UFL expression of shape (tdim, tdim)
        """
        return 0.5 * (ufl.grad(v) + ufl.grad(v).T)

    def sigma(self, eps):
        """
        Cauchy stress for a given strain tensor (Hooke's law).

        σ = λ tr(ε) I + 2μ ε

        Parameters
        ----------
        eps : UFL expression – strain tensor (or elastic strain ε – εᵖ)

        Returns
        -------
        UFL expression of shape (tdim, tdim)
        """
        return (
            self.lam * ufl.tr(eps) * ufl.Identity(self.tdim)
            + 2.0 * self.mu * eps
        )

    def sigma_d(self, s):
        """
        Deviatoric part of a stress (or any symmetric tensor).

        s_d = s – (1/3) tr(s) I

        Parameters
        ----------
        s : UFL expression – symmetric second-order tensor

        Returns
        -------
        UFL expression of shape (tdim, tdim)
        """
        return s - (1.0 / 3.0) * ufl.tr(s) * ufl.Identity(self.tdim)

    def von_mises(self, s):
        """
        von Mises equivalent stress.

        J(s) = √( 3/2 · s_d : s_d )

        This is the scalar measure used as the J2 yield criterion radius.

        Parameters
        ----------
        s : UFL expression – Cauchy stress tensor

        Returns
        -------
        UFL scalar expression
        """
        return ufl.sqrt(1.5 * ufl.inner(self.sigma_d(s), self.sigma_d(s)))


# ===========================================================================
# Plasticity model interface
# ===========================================================================
class PlasticState(ABC):
    """
    Abstract container for internal (history) variables.

    Concrete subclasses must store their internal variables as
    fem.Function objects so that UFL expressions built from them
    remain valid across time steps (values updated in-place by commit).

    All concrete states are expected to expose at minimum:
        eps_p_old : fem.Function (tensor DG-0) – plastic strain tensor
        p_old     : fem.Function (scalar DG-0) – cumulative plastic strain

    If your model uses different variables, override cauchy_stress() as well.

    Notes
    -----
    DG-0 (piecewise-constant, discontinuous Galerkin) spaces are chosen
    because internal variables are cell-wise quantities in the standard
    closest-point return-mapping algorithm.
    """


class PlasticityModel(ABC):
    """
    Abstract base class for plasticity models.

    To implement a new model, subclass this and implement the three
    abstract methods below.  cauchy_stress has a sensible default
    for standard elastoplastic models (σ = C : (ε – εᵖ)).

    Parameters
    ----------
    elastic : ElasticModel
        Shared elastic predictor used by all concrete models.

    Extension guide
    ---------------
    1. Create a PlasticState subclass with your history variables.
    2. Implement create_state, update, and commit.
    3. Override cauchy_stress if your stress formula differs from the
       standard  σ = C : (ε – εᵖ_old – Δεᵖ)  form.
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

        Returns
        -------
        PlasticState subclass instance
        """

    @abstractmethod
    def update(self, state: PlasticState, eps) -> tuple:
        """
        Compute plastic increments from the current total strain.

        This is the *return-mapping* step, expressed in UFL so it can be
        differentiated automatically for the Newton tangent.

        Parameters
        ----------
        state : PlasticState  – internal variables at tₙ
        eps   : UFL expression – total strain ε(u) at current Newton iterate

        Returns
        -------
        delta_p     : UFL expression – increment of cumulative plastic strain Δp
        delta_eps_p : UFL expression – increment of plastic strain tensor Δεᵖ
        """

    @abstractmethod
    def commit(self, state: PlasticState, uh) -> None:
        """
        Advance internal variables from tₙ to tₙ₊₁.

        Called once per time step, after Newton convergence.

        .. important::
            Must update state **in-place** (never replace fem.Function
            objects; UFL forms hold references to them).

        Parameters
        ----------
        state : PlasticState
        uh    : fem.Function – converged displacement field at tₙ₊₁
        """

    def cauchy_stress(self, state: PlasticState, u) -> object:
        """
        Cauchy stress  σ = C : (ε(u) – εᵖ_old – Δεᵖ).

        This default is suitable for standard elastoplastic models.
        Override for models that require a different stress computation
        (e.g. viscoplasticity, damage, or kinematic hardening with
        back-stress).

        Parameters
        ----------
        state : PlasticState – holds εᵖ_old and p_old at tₙ
        u     : UFL / fem.Function – current displacement iterate

        Returns
        -------
        UFL expression for the Cauchy stress tensor
        """
        eps = self.elastic.epsilon(u)
        # Compute plastic increment via the return-mapping update
        _, delta_eps_p = self.update(state, eps)
        # Elastic strain = total strain minus accumulated plastic strain
        return self.elastic.sigma(eps - (state.eps_p_old + delta_eps_p))


# ===========================================================================
# J2 plasticity with Voce isotropic hardening
# ===========================================================================
class _J2State(PlasticState):
    """
    Internal variables for the J2 isotropic hardening model.

    Attributes
    ----------
    p_old     : fem.Function (scalar DG-0)
        Cumulative equivalent plastic strain p at tₙ.
    eps_p_old : fem.Function (tensor DG-0)
        Plastic strain tensor εᵖ at tₙ.
    _W        : scalar DG-0 FunctionSpace (kept for interpolation)
    _WT       : tensor DG-0 FunctionSpace (kept for interpolation)
    """

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # cumulative plastic strain p
        self.eps_p_old = fem.Function(WT)  # plastic strain tensor εᵖ
        # Store spaces for use inside commit()
        self._W        = W
        self._WT       = WT


class J2IsotropicHardening(PlasticityModel):
    """
    J2 plasticity with Voce isotropic hardening.

    Constitutive equations
    ----------------------
    Yield function   :  f(σ, p) = J(σ) – σ_Y – Q·(1 – exp(–k·p))
    Flow rule        :  Δεᵖ = Δp · n,   n = (3/2) σ_d / J(σ)
    Hardening law    :  R(p) = Q·(1 – exp(–k·p))   (Voce saturation)
    Return mapping   :  one linearised (secant) Newton step

    The linearised correction  Δp = –f / (∂f/∂p – 3μ)  is applied
    element-wise as a UFL conditional expression (active only outside
    the elastic domain).

    Parameters
    ----------
    elastic  : ElasticModel
    sigma_Y  : float – initial yield stress  [same units as E]
    Q_var    : float – saturation hardening stress
    k        : float – hardening rate (Voce exponent)
    """

    def __init__(self, elastic: ElasticModel, sigma_Y: float, Q_var: float, k: float):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.Q_var   = Q_var
        self.k       = k

    # -- internal UFL helpers ----------------------------------------------

    def _yield_func(self, sigma, p):
        """
        Scalar yield function  f = J(σ) – σ_Y – R(p).

        R(p) = Q·(1 – exp(–k·p))  is the Voce isotropic hardening radius.
        """
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self.elastic.von_mises(sigma) - self.sigma_Y - R

    def _flow_normal(self, sigma):
        """
        Unit outward normal to the yield surface in stress space.

        n = (3/2) · σ_d / J(σ)

        This is the Prandtl–Reuss flow direction (associated flow rule).
        """
        return (3.0 / (2.0 * self.elastic.von_mises(sigma))) * self.elastic.sigma_d(sigma)

    # -- PlasticityModel interface -----------------------------------------

    def create_state(self, domain, W, WT) -> _J2State:
        """Allocate J2 history variables (all initialised to zero)."""
        return _J2State(W, WT)

    def update(self, state: _J2State, eps) -> tuple:
        """
        One-step linearised return mapping for J2 isotropic hardening.

        Algorithm
        ---------
        1. Elastic predictor:  σ_tr = C : (ε – εᵖ_old)
        2. Check yield:        f_tr = f(σ_tr, p_old)
        3. If f_tr ≥ 0 (plastic):
               Δp  = –f_tr / (∂f/∂Δp)|_{σ_tr}
                   = –f_tr / (–R'(p_old) – 3μ)
               Δεᵖ = Δp · n(σ_tr)
           else Δp = 0, Δεᵖ = 0.

        The result is expressed as UFL conditionals so that automatic
        differentiation (for the Newton tangent) works correctly.

        Parameters
        ----------
        state : _J2State – tₙ history variables
        eps   : UFL expression – total strain at current iterate

        Returns
        -------
        delta_p     : UFL scalar – Δp
        delta_eps_p : UFL tensor – Δεᵖ
        """
        # Elastic trial stress
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._yield_func(sigma_tr, state.p_old)
        n        = self._flow_normal(sigma_tr)

        # Derivative of the isotropic hardening modulus  dR/dp = Q·k·exp(–k·p)
        R_prime  = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)
        # Linearised denominator: ∂f/∂Δp = –R' – 3μ  (consistency condition)
        f_prime  = -R_prime - 3.0 * self.elastic.mu
        # Plastic multiplier from one Newton step  Δp = –f / f'
        delta_p0 = -(1.0 / f_prime) * self._yield_func(sigma_tr, state.p_old)

        # Activate the plastic correction only when the predictor violates yield
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        # Equivalent plastic strain increment  Δp = √(2/3 · Δεᵖ : Δεᵖ)
        delta_p     = ufl.sqrt(2.0 / 3.0 * ufl.inner(delta_eps_p, delta_eps_p))
        return delta_p, delta_eps_p

    def commit(self, state: _J2State, uh) -> None:
        """
        Update J2 history variables from tₙ to tₙ₊₁.

        Steps
        -----
        1. Interpolate the incremental cumulative plastic strain Δp and
           add it to p_old  → p_{n+1}.
        2. Interpolate the incremental plastic strain tensor Δεᵖ and
           accumulate it into eps_p_old  → εᵖ_{n+1}.

        .. note::
            eps_p_old.x.array[:] += ... performs the tensor update
            directly on the PETSc vector data to avoid creating a
            temporary fem.Function (cheaper and avoids UFL aliasing).
        """
        eps                  = self.elastic.epsilon(uh)
        delta_p, delta_eps_p = self.update(state, eps)

        # --- update cumulative plastic strain p ---
        state.p_old.interpolate(
            fem.Expression(state.p_old + delta_p, state._W.element.interpolation_points)
        )

        # --- update plastic strain tensor εᵖ ---
        # Project Δεᵖ into the DG-0 tensor space first, then add in-place
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

    Parameters
    ----------
    msh       : meshio.Mesh – full mesh read from file
    cell_type : str  – 'tetra', 'triangle', 'line', …
    prune_z   : bool – drop the z-coordinate (for 2-D meshes embedded in 3-D)

    Returns
    -------
    meshio.Mesh containing only cells of the requested type
    """
    cells  = msh.get_cells_type(cell_type)
    # Strip z for 2-D: dolfinx expects (x, y) only in 2-D problems
    points = msh.points[:, :2] if prune_z else msh.points
    return meshio.Mesh(points=points, cells={cell_type: cells})


def create_2D_mesh(msh, cell_type):
    """
    Extract a 2-D meshio.Mesh (always strips the z-coordinate).

    Convenience wrapper around create_mesh for the 2-D case.

    Parameters
    ----------
    msh       : meshio.Mesh
    cell_type : str  – 'triangle', 'line', …

    Returns
    -------
    meshio.Mesh with 2-D coordinates
    """
    cells  = msh.get_cells_type(cell_type)
    points = msh.points[:, :2]  # drop z column unconditionally
    return meshio.Mesh(points=points, cells={cell_type: cells})

# def load_and_write_mesh(mesh_file):
#     """
#     Read a Gmsh .msh file, write XDMF sub-meshes, return the dolfinx domain.
#
#     Only rank 0 writes; all ranks read.
#
#     Parameters
#     ----------
#     mesh_file : str – path to the .msh file
#
#     Returns
#     -------
#     domain : dolfinx.mesh.Mesh
#     """
#     if MPI.COMM_WORLD.rank == 0:
#         msh           = meshio.read(mesh_file)
#         triangle_mesh = create_mesh(msh, "tetra", prune_z=False)
#         line_mesh     = create_mesh(msh, "line",  prune_z=True)
#         meshio.write("mesh.xdmf", triangle_mesh)
#         meshio.write("mt.xdmf",   line_mesh)
#
#     with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
#         domain = xdmf.read_mesh(name="Grid")
#
#     domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
#     return domain


def load_and_write_mesh(mesh_file):
    """
    Read a Gmsh .msh file, detect if it is 2D or 3D,
    write appropriate XDMF sub-meshes, and return the dolfinx domain.

    Dimension detection
    -------------------
    - If 'tetra' cells are present  → 3-D problem; domain = tetra, boundary = triangle.
    - If only 'triangle' cells      → 2-D problem; domain = triangle, boundary = line.

    MPI strategy
    ------------
    Only rank 0 performs I/O (meshio read + XDMF write).
    A MPI_Barrier ensures all ranks wait before reading the shared XDMF.

    Parameters
    ----------
    mesh_file : str – path to the Gmsh .msh file

    Returns
    -------
    domain : dolfinx.mesh.Mesh
        The loaded and topology-connected mesh, distributed across all MPI ranks.

    Raises
    ------
    ValueError
        If the mesh contains neither 'tetra' nor 'triangle' cells.
    """
    if MPI.COMM_WORLD.rank == 0:
        msh = meshio.read(mesh_file)

        # Auto-detect spatial dimension from cell types present in the file
        has_tetra    = any(cell.type == "tetra"    for cell in msh.cells)
        has_triangle = any(cell.type == "triangle" for cell in msh.cells)

        if has_tetra:
            print(f"[Mesh Loader] Détection d'un maillage 3D ({mesh_file})")
            # 3-D: volumetric cells = tetrahedra; surface boundary = triangles
            domain_mesh   = create_mesh(msh, "tetra",    prune_z=False)
            boundary_mesh = create_mesh(msh, "line",     prune_z=False)
        elif has_triangle:
            print(f"[Mesh Loader] Détection d'un maillage 2D ({mesh_file})")
            # 2-D: surface cells = triangles; edge boundary = lines
            # prune_z=True is critical to give dolfinx a proper 2-D coordinate array
            domain_mesh   = create_2D_mesh(msh, "triangle")
            boundary_mesh = create_2D_mesh(msh, "line")
        else:
            raise ValueError("Le maillage ne contient ni 'tetra' ni 'triangle'. Format non supporté.")

        # Write XDMF files that all MPI ranks will subsequently read
        meshio.write("mesh.xdmf", domain_mesh)
        meshio.write("mt.xdmf",   boundary_mesh)

    # Ensure rank 0 has finished writing before other ranks try to read
    MPI.COMM_WORLD.Barrier()

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # Build facet–cell connectivity needed for boundary-condition marking
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    return domain


# ===========================================================================
# Function spaces
# ===========================================================================
def build_function_spaces(domain):
    """
    Build the three FEM function spaces required by the solver.

    Spaces
    ------
    V  : CG-1 vector   – nodal displacements u  (continuous, degree 1)
    W  : DG-0 scalar   – scalar internal variables (p, von Mises, …)
    WT : DG-0 tensor   – tensor internal variables (εᵖ, ε, σ, …)

    DG-0 (piecewise-constant) spaces are standard for return-mapping
    internal variables because the closest-point algorithm operates
    element-by-element.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh

    Returns
    -------
    V, W, WT : tuple of fem.FunctionSpace
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
    L2-project a UFL expression *v* onto the function space of *target_func*.

    Solves the variational problem:
        find Pv ∈ V  s.t.  ∫ Pv·w dx = ∫ v·w dx  ∀ w ∈ V

    Parameters
    ----------
    v           : UFL expression – quantity to project
    target_func : fem.Function   – result stored here (modified in-place)
    bcs         : list of DirichletBC – optional essential boundary conditions

    Returns
    -------
    target_func : fem.Function (same object, now containing the projection)

    Notes
    -----
    Uses a PETSc KSP solver (default: GMRES + ILU) on the assembled mass
    matrix.  For DG-0 spaces the mass matrix is block-diagonal so this is
    very cheap.
    """
    if bcs is None:
        bcs = []
    domain = target_func.function_space.mesh
    V      = target_func.function_space
    dx     = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    w, Pv  = ufl.TestFunction(V), ufl.TrialFunction(V)

    # Assemble mass matrix and load vector
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
def build_right_facet_tag(domain, coord=1):
    """
    Mark the boundary facets at the positive extreme of *coord* with tag 1.

    Typically used for the right (or top) boundary where the reaction force
    is integrated.  The tagged measure ds(1) restricts integration to
    those facets.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    coord  : int – coordinate axis to search along (0 = X, 1 = Y, 2 = Z)

    Returns
    -------
    ds : ufl.Measure restricted to facets tagged as 1
    """
    x_coords  = domain.geometry.x[:, coord]
    x_max     = np.max(x_coords)
    # Locate facets within a floating-point tolerance of the max coordinate
    facets    = locate_entities(domain, domain.topology.dim - 1,
                                lambda x: x[coord] >= (x_max - 1e-8))
    facet_tag = meshtags(domain, domain.topology.dim - 1,
                         facets, np.full_like(facets, 1))
    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


# def dirichlet_bcs(domain, space, disp_value, length):
#     """
#     Symmetric tensile BCs: left at –disp_value, right at +disp_value.
#     ...
#     """
#     fdim         = domain.topology.dim - 1
#     left_facets  = mesh.locate_entities_boundary(
#         domain, fdim, lambda x: x[0] <= (-length + 1e-8))
#     right_facets = mesh.locate_entities_boundary(
#         domain, fdim, lambda x: x[0] >= (+length - 1e-8))
#     bc_left  = fem.dirichletbc(fem.Constant(domain, -disp_value), ...)
#     bc_right = fem.dirichletbc(fem.Constant(domain,  disp_value), ...)
#     return [bc_left, bc_right]


def dirichlet_bcs(domain, space, disp_value, length):
    """
    Symmetric tensile BCs: automatically detects left and right boundaries
    based on the mesh bounding box along the Y axis.

    The minimum Y coordinate → left (–disp_value); maximum → right (+disp_value).
    A numerical tolerance guards against floating-point round-off on boundary nodes.

    Parameters
    ----------
    domain     : dolfinx.mesh.Mesh
    space      : fem.FunctionSpace – vector displacement space V
    disp_value : array-like – displacement vector (already time-scaled)
    length     : float – half-length of the specimen (kept for signature
                 compatibility; geometry is detected automatically)

    Returns
    -------
    list of two DirichletBC: [bc_left, bc_right]
    """
    fdim = domain.topology.dim - 1

    # Auto-detect geometric extremes along Y
    x_coords = domain.geometry.x[:, 1]
    x_min    = np.min(x_coords)
    x_max    = np.max(x_coords)

    tol = 1e-6  # tolerance against machine-precision round-off
    left_facets  = mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] <= (x_min + tol))
    right_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] >= (x_max - tol))

    bc_left  = fem.dirichletbc(fem.Constant(domain, -disp_value),
                               fem.locate_dofs_topological(space, fdim, left_facets), space)
    bc_right = fem.dirichletbc(fem.Constant(domain,  disp_value),
                               fem.locate_dofs_topological(space, fdim, right_facets), space)
    return [bc_left, bc_right]


def dirichlet_bcs_tensile(domain, space, left_disp_constant, right_disp_constant, coord=0):
    """
    Tensile boundary conditions using pre-allocated fem.Constant objects.

    Unlike dirichlet_bcs, this function accepts *existing* Constant objects
    so that their .value can be updated in-place inside the time loop without
    rebuilding the entire BC list each step.

    Parameters
    ----------
    domain              : dolfinx.mesh.Mesh
    space               : fem.FunctionSpace – vector displacement space V
    left_disp_constant  : fem.Constant – displacement prescribed on the left boundary
    right_disp_constant : fem.Constant – displacement prescribed on the right boundary
    coord               : int – axis along which to detect boundaries (default 0 = X)

    Returns
    -------
    list of two DirichletBC: [bc_left, bc_right]

    Notes
    -----
    The geometric extremes are always detected automatically from
    domain.geometry.x[:, coord], so the function adapts to any mesh size.
    """
    fdim = domain.topology.dim - 1

    # Detect boundary extremes along the chosen coordinate axis
    x_coords = domain.geometry.x[:, coord]
    x_min    = np.min(x_coords)
    x_max    = np.max(x_coords)

    tol = 1e-6
    left_facets  = mesh.locate_entities_boundary(domain, fdim, lambda x: x[coord] <= (x_min + tol))
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: x[coord] >= (x_max - tol))

    # Apply the pre-allocated Constant objects directly; caller updates .value each step
    bc_left  = fem.dirichletbc(left_disp_constant,
                               fem.locate_dofs_topological(space, fdim, left_facets), space)
    bc_right = fem.dirichletbc(right_disp_constant,
                               fem.locate_dofs_topological(space, fdim, right_facets), space)
    return [bc_left, bc_right]


# ===========================================================================
# Variational form and Newton solver
# ===========================================================================
def build_solver(domain, V, model: PlasticityModel, state: PlasticState, bcs,
                 quadrature_degree: int = 1):
    """
    Assemble the quasi-static equilibrium residual and build the Newton solver.

    Variational problem
    -------------------
    Find u ∈ V  s.t.:
        ∫ σ(u) : sym∇v dx = 0   ∀ v ∈ V

    where σ(u) is supplied by model.cauchy_stress(state, u).
    The tangent (Jacobian) J = dF/du is computed by UFL automatic differentiation.

    Parameters
    ----------
    domain            : dolfinx.mesh.Mesh
    V                 : fem.FunctionSpace – vector CG-1 displacement space
    model             : PlasticityModel   – provides cauchy_stress()
    state             : PlasticState      – current history variables (tₙ)
    bcs               : list of DirichletBC
    quadrature_degree : int – integration rule order (1 is sufficient for DG-0
                        internal variables; increase for smoother fields)

    Returns
    -------
    uh      : fem.Function – displacement solution (updated in-place by solver)
    problem : NewtonSolverNonlinearProblem
    solver  : NewtonSolver configured with absolute/relative tolerances 1e-8
    """
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    v  = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    uh = fem.Function(V)  # displacement; starts at zero (correct for incremental BCs)

    # Residual  F = ∫ σ : ∇ₛv dx
    F = ufl.inner(model.cauchy_stress(state, uh), ufl.sym(ufl.grad(v))) * dx
    # Consistent tangent  J = dF/du  (automatic differentiation)
    J = ufl.derivative(F, uh, du)

    problem               = NewtonSolverNonlinearProblem(F, uh, bcs=bcs, J=J)
    solver                = NewtonSolver(domain.comm, problem)
    solver.atol           = 1e-8
    solver.rtol           = 1e-8
    solver.max_it         = 50
    solver.convergence_criterion = "incremental"
    return uh, problem, solver


# ===========================================================================
# Simulation runners
# ===========================================================================

def run_simulation_fast(domain, V, W, WT, config=None, coord=1, model: PlasticityModel = None):
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
    load_amp  = cfg["load_amp"]
    length    = cfg["length"]
    ds = build_right_facet_tag(domain, coord)

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

    if gdim == 2:
        # 2-D: displacement vector has two components (u_x, u_y)
        disp_value       = np.array((0.1 * load_amp, load_amp), dtype=PETSc.ScalarType)
        left_disp_const  = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        right_disp_const = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        bcs = dirichlet_bcs_tensile(domain, V, left_disp_const, right_disp_const, coord)
    else:
        # 3-D: displacement vector has three components (u_x, u_y, u_z)
        disp_value = np.array((0.1 * load_amp, load_amp, 0.0), dtype=PETSc.ScalarType)
        bcs        = dirichlet_bcs(domain, V, disp_value, length)

    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds                  = build_right_facet_tag(domain, coord)

    # -------------------------------------------- time loop --------------
    force_vec  = []
    displ_val  = []
    t_paraview = 0

    # Suppress PETSc/SNES console output for cleaner FEMU logs
    opts = PETSc.Options()
    opts["ksp_monitor"]  = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    for step in range(num_steps + 1):
        t      += dt
        disp_t  = disp_value[1] * t  # current displacement magnitude

        if gdim == 2:
            # Update the existing Constant objects in-place (avoids rebuilding BCs)
            left_disp_const.value  = np.array([0.0, -disp_t], dtype=PETSc.ScalarType)
            right_disp_const.value = np.array([0.0,  disp_t], dtype=PETSc.ScalarType)
        else:
            # 3-D: rebuild the BC list with the new amplitude
            bcs = dirichlet_bcs(domain, V, disp_value * t, length)

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


def run_simulation_relax(domain, V, W, WT, config=None, coord=1,
                         model: PlasticityModel = None, num_supp_steps: int = 50):
    """
    Run the elasto-plastic simulation with a stress-relaxation phase.

    Identical to run_simulation_V3 for the first num_steps steps
    (loading phase), then continues for num_supp_steps additional steps
    with **frozen boundary conditions** to allow elastic unloading / relaxation.

    Parameters
    ----------
    domain, V, W, WT : same as run_simulation_V3
    config           : dict, optional
    coord            : int – loading direction
    model            : PlasticityModel, optional
    num_supp_steps   : int
        Number of additional steps after peak load during which the BCs are
        held constant.  Defaults to 50.

    Returns
    -------
    force_vec : list of float – length = num_steps + 1 + num_supp_steps
    displ_val : list of np.ndarray – same length
    """
    cfg       = {**DEFAULT_CONFIG, **(config or {})}
    t         = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt        = (cfg["T"] - t) / num_steps
    load_amp  = cfg["load_amp"]
    length    = cfg["length"]
    ds = build_right_facet_tag(domain, coord)

    fic = None  # no file output in this variant

    # ----------------------------------------- build plasticity model ----
    if model is None:
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------- boundary conditions --------
    gdim = domain.geometry.dim

    if gdim == 2:
        disp_value       = np.array((0.1 * load_amp, load_amp), dtype=PETSc.ScalarType)
        left_disp_const  = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        right_disp_const = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        bcs = dirichlet_bcs_tensile(domain, V, left_disp_const, right_disp_const, coord)
    else:
        disp_value = np.array((0.1 * load_amp, load_amp, 0.0), dtype=PETSc.ScalarType)
        bcs        = dirichlet_bcs(domain, V, disp_value, length)

    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds                  = build_right_facet_tag(domain, coord)

    # -------------------------------------------- time loop --------------
    force_vec  = []
    displ_val  = []
    t_paraview = 0

    opts = PETSc.Options()
    opts["ksp_monitor"]  = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    # Total steps = loading steps + supplementary (relaxation) steps
    for step in range(num_steps + 1 + num_supp_steps):
        t += dt

        if step <= num_steps:
            # --- Loading phase: ramp up the displacement ---
            disp_t = disp_value[1] * t

            if gdim == 2:
                left_disp_const.value  = np.array([0.0, -disp_t], dtype=PETSc.ScalarType)
                right_disp_const.value = np.array([0.0,  disp_t], dtype=PETSc.ScalarType)
            else:
                bcs = dirichlet_bcs(domain, V, disp_value * t, length)
        else:
            # --- Relaxation phase: BCs remain at their last value (no update) ---
            pass

        problem.bcs = bcs
        solver.solve(uh)

        eps                  = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        current_displ = uh.x.array.copy()
        displ_val.append(current_displ)

        stress = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
        force  = fem.assemble_scalar(fem.form(stress[1, 1] * ds(1)))
        force_vec.append(force)

        model.commit(state, uh)

    if fic is not None:
        fic.close()

    return force_vec, displ_val


def run_simulation_write(domain, V, W, WT, config=None, coord=1,
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
    load_amp  = cfg["load_amp"]
    length    = cfg["length"]

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
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model   = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------- boundary conditions --------
    gdim = domain.geometry.dim

    if gdim == 2:
        disp_value       = np.array((0.1 * load_amp, load_amp), dtype=PETSc.ScalarType)
        left_disp_const  = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        right_disp_const = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
        bcs = dirichlet_bcs_tensile(domain, V, left_disp_const, right_disp_const, coord)
    else:
        disp_value = np.array((0.1 * load_amp, load_amp, 0.0), dtype=PETSc.ScalarType)
        bcs        = dirichlet_bcs(domain, V, disp_value, length)

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
        t      += dt
        disp_t  = disp_value[1] * t

        if gdim == 2:
            left_disp_const.value  = np.array([0.0, -disp_t], dtype=PETSc.ScalarType)
            right_disp_const.value = np.array([0.0,  disp_t], dtype=PETSc.ScalarType)
        else:
            bcs = dirichlet_bcs(domain, V, disp_value * t, length)

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


# ===========================================================================
# Entry point
# ===========================================================================
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
        file_name    = "ref_astar",
        # Elastic constants
        E           = 200_000.0,
        nu          = 0.3,
        # J2 isotropic hardening parameters
        sigma_Y     = 100.0,
        Q_var       = 50.0,
        k_hardening = 1000.0,
    )

    # ---- load mesh and build function spaces -----------------------------
    #with io.XDMFFile(MPI.COMM_WORLD, "".xdmf", "r") as xdmf:
    domain = load_and_write_mesh(config["mesh_file"])
    V, W, WT = build_function_spaces(domain)

    # ---- run simulation with relaxation phase ----------------------------
    forces, _ = run_simulation_write(domain, V, W, WT, config=config)
    np.save("forces_sample.npy", forces)
    # with open("forces_sample.npy", "wb") as f:
    #     np.save(f, forces)
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