"""
simu_tools.py
-------------
Core constitutive models and FEM solver construction for elasto-plastic
simulations using FEniCSx / dolfinx.

Architecture & Abstractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ElasticModel          – Linear-elastic constitutive relation & Lamé constants + UFL expressions
PlasticState (ABC)    – Abstract container for history/internal variables stored as fem.Functions
PlasticityModel (ABC) – Interface for elastoplastic constitutive models
_J2State              – History storage for J2 isotropic hardening (p_old, eps_p_old)
J2IsotropicHardening  – J2 von Mises plasticity with Voce isotropic hardening law

Solver & Function Spaces
~~~~~~~~~~~~~~~~~~~~~~~~
build_function_spaces – Constructs continuous CG-1 displacement and DG-0 internal variable spaces
build_right_facet_tag – Tags upper/right boundary facets for force reaction integration
build_solver          – Assembles non-linear weak form and configures PETSc Newton solver
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, log, mesh
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import pyvista as pv
from time import sleep, time

# ===========================================================================
# Elastic model
# ===========================================================================
class ElasticModel:
    """
    Linear-elastic constitutive model.

    Computes and stores Lamé constants (mu, lambda) from Young's modulus E 
    and Poisson's ratio nu, and exposes UFL symbolic expression builders for
    strain, Cauchy stress, stress deviator, and von Mises equivalent stress.

    Parameters
    ----------
    E    : float
        Young's modulus [MPa or consistent stress units].
    nu   : float
        Poisson ratio [dimensionless, in range (0, 0.5)].
    tdim : int
        Spatial dimension of the mesh (2 for plane stress/strain, 3 for 3D).
    """

    def __init__(self, E: float, nu: float, tdim: int):
        self.E    = E
        self.nu   = nu
        self.tdim = tdim
        
        # Derived Lamé parameters:
        #   mu = E / (2 * (1 + nu))                     [Shear modulus]
        #   lambda = E * nu / ((1 + nu) * (1 - 2*nu))   [First Lamé parameter]
        self.mu   = E / (2.0 * (1.0 + nu))
        self.lam  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(self, v):
        """
        Symmetric gradient – linearised strain tensor ε(v).
        
        ε(v) = ½ (∇v + ∇vᵀ)
        """
        return 0.5 * (ufl.grad(v) + ufl.grad(v).T)

    def sigma(self, eps):
        """
        Cauchy stress tensor for linear isotropic elasticity (Hooke's law).
        
        σ = λ tr(ε) I + 2μ ε
        """
        return (
            self.lam * ufl.tr(eps) * ufl.Identity(self.tdim)
            + 2.0 * self.mu * eps
        )

    def sigma_d(self, s):
        """
        Deviatoric stress tensor s_d.
        
        s_d = s – (1/3) tr(s) I
        """
        return s - (1.0 / 3.0) * ufl.tr(s) * ufl.Identity(self.tdim)

    def von_mises(self, s):
        """
        von Mises equivalent stress J(s).
        
        J(s) = √(3/2 · s_d : s_d)
        """
        return ufl.sqrt(1.5 * ufl.inner(self.sigma_d(s), self.sigma_d(s)))


# ===========================================================================
# Plasticity model interface
# ===========================================================================
class PlasticState(ABC):
    """
    Abstract container for internal (history) variables.

    Concrete subclasses must store internal variables as `fem.Function`
    objects in Discontinuous Galerkin (DG-0) spaces so that UFL symbolic 
    expressions built from them remain valid across time steps.
    """


class PlasticityModel(ABC):
    """
    Abstract base class for elastoplastic constitutive models.

    Parameters
    ----------
    elastic : ElasticModel
        Shared elastic predictor model providing elasticity constants and UFL builders.
    """

    def __init__(self, elastic: ElasticModel):
        self.elastic = elastic

    @abstractmethod
    def create_state(self, domain, W, WT) -> PlasticState:
        """
        Allocate and return internal-variable storage (DG-0 functions) for this model.
        """

    @abstractmethod
    def update(self, state: PlasticState, eps) -> tuple:
        """
        Compute plastic increments from the total strain tensor via return-mapping.

        Parameters
        ----------
        state : PlasticState
            History variables at the previous converged step tₙ.
        eps : UFL expression
            Total strain tensor ε(u) at the current Newton iterate.

        Returns
        -------
        delta_p : UFL expression
            Increment of cumulative plastic strain Δp.
        delta_eps_p : UFL expression
            Increment of plastic strain tensor Δεᵖ.
        """

    @abstractmethod
    def commit(self, state: PlasticState, uh) -> None:
        """
        Advance internal variables from tₙ to tₙ₊₁ after global Newton convergence.
        Updates internal `fem.Function` objects in-place.
        """

    def cauchy_stress(self, state: PlasticState, u) -> object:
        """
        Elastoplastic Cauchy stress: σ = C : (ε(u) – εᵖ_old – Δεᵖ).

        Parameters
        ----------
        state : PlasticState
            History variables at step tₙ.
        u : UFL Function or TrialFunction
            Current displacement field iterate.
        """
        eps = self.elastic.epsilon(u)
        _, delta_eps_p = self.update(state, eps)
        return self.elastic.sigma(eps - (state.eps_p_old + delta_eps_p))


# ===========================================================================
# J2 plasticity with Voce isotropic hardening
# ===========================================================================
class _J2State(PlasticState):
    """
    Internal history variable storage for J2 isotropic hardening.

    Attributes
    ----------
    p_old : fem.Function (scalar DG-0)
        Accumulated equivalent plastic strain p at tₙ.
    eps_p_old : fem.Function (tensor DG-0)
        Plastic strain tensor εᵖ at tₙ.
    """

    def __init__(self, W, WT):
        self.p_old     = fem.Function(W)   # cumulative plastic strain p
        self.eps_p_old = fem.Function(WT)  # plastic strain tensor εᵖ
        self._W        = W
        self._WT       = WT


class J2IsotropicHardening(PlasticityModel):
    """
    J2 plasticity with Voce isotropic hardening.

    Formulation
    -----------
    Yield function   : f(σ, p) = J(σ) – σ_Y – Q · (1 – exp(–k · p))
    Flow rule        : Δεᵖ = Δp · n,   where n = (3/2) s_d / J(σ)
    Hardening radius : R(p) = σ_Y + Q · (1 – exp(–k · p))

    Linearised Return Mapping:
        Trial stress   : σ_tr = C : (ε – εᵖ_old)
        Trial yield    : f_tr = J(σ_tr) – R(p_old)
        Newton update  : Δp = –f_tr / (–R'(p_old) – 3μ)  if f_tr ≥ 0, else 0

    Parameters
    ----------
    elastic : ElasticModel
        Shared elastic predictor instance.
    sigma_Y : float
        Initial yield stress [MPa].
    Q_var : float
        Voce isotropic hardening saturation stress [MPa].
    k : float
        Voce hardening rate exponent.
    """

    def __init__(self, elastic: ElasticModel, sigma_Y: float, Q_var: float, k: float):
        super().__init__(elastic)
        self.sigma_Y = sigma_Y
        self.Q_var   = Q_var
        self.k       = k

    def _yield_func(self, sigma, p):
        """Yield function f = J(σ) - σ_Y - Q*(1 - exp(-k*p))."""
        R = self.Q_var * (1.0 - ufl.exp(-self.k * p))
        return self.elastic.von_mises(sigma) - self.sigma_Y - R

    def _flow_normal(self, sigma):
        """Associated plastic flow direction n = (3/2) * s_d / von_mises(σ)."""
        return (3.0 / (2.0 * self.elastic.von_mises(sigma))) * self.elastic.sigma_d(sigma)

    def create_state(self, domain, W, WT) -> _J2State:
        return _J2State(W, WT)

    def update(self, state: _J2State, eps) -> tuple:
        """
        Radial return-mapping step expressed via UFL symbolic conditionals.
        
        Using UFL conditionals allows FEniCSx to compute the consistent elastoplastic
        tangent tensor automatically via symbolic differentiation `ufl.derivative()`.
        """
        # 1. Elastic trial predictor assuming no plastic increment in current step
        sigma_tr = self.elastic.sigma(eps - state.eps_p_old)
        f_val    = self._yield_func(sigma_tr, state.p_old)
        n        = self._flow_normal(sigma_tr)
        
        # 2. Derivative of hardening law: dR/dp = Q * k * exp(-k * p_old)
        R_prime  = self.Q_var * self.k * ufl.exp(-self.k * state.p_old)
        
        # 3. Linearised consistency denominator: df/d(delta_p) = -R' - 3*mu
        f_prime  = -R_prime - 3.0 * self.elastic.mu
        delta_p0 = -(1.0 / f_prime) * f_val
        
        # 4. UFL conditional evaluation: plastic update if f_val >= 0, else zero
        delta_eps_p = ufl.conditional(ufl.ge(f_val, 0.0), delta_p0 * n, 0.0 * n)
        delta_p     = ufl.sqrt(2.0 / 3.0 * ufl.inner(delta_eps_p, delta_eps_p))
        return delta_p, delta_eps_p

    def commit(self, state: _J2State, uh) -> None:
        """
        Interpolate and advance plastic history variables (p and εᵖ) in-place.
        """
        eps                  = self.elastic.epsilon(uh)
        delta_p, delta_eps_p = self.update(state, eps)
        
        # In-place interpolation into the DG-0 scalar space W
        state.p_old.interpolate(
            fem.Expression(state.p_old + delta_p, state._W.element.interpolation_points)
        )
        
        # In-place interpolation into the DG-0 tensor space WT
        delta_eps_p_proj = fem.Function(state._WT)
        delta_eps_p_proj.interpolate(
            fem.Expression(delta_eps_p, state._WT.element.interpolation_points)
        )
        state.eps_p_old.x.array[:] += delta_eps_p_proj.x.array[:]


# ===========================================================================
# Function spaces
# ===========================================================================
def build_function_spaces(domain):
    """
    Construct finite element function spaces for the non-linear simulation.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The active mesh domain.

    Returns
    -------
    V : fem.FunctionSpace
        Continuous Galerkin (CG-1) vector space for nodal displacement field u.
    W : fem.FunctionSpace
        Discontinuous Galerkin (DG-0) scalar space for cell-wise internal variables (p).
    WT : fem.FunctionSpace
        Discontinuous Galerkin (DG-0) tensor space for cell-wise plastic strain (εᵖ).
    """
    tdim = domain.topology.dim
    V    = fem.functionspace(domain, ("CG", 1, (tdim,)))
    W    = fem.functionspace(domain, ("DG", 0))
    WT   = fem.functionspace(domain, ("DG", 0, (tdim, tdim)))
    return V, W, WT


# ===========================================================================
# Boundary condition utilities
# ===========================================================================
def build_right_facet_tag(domain, coord=1):
    """
    Tag boundary facets at the maximum coordinate along specified axis (coord).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    coord : int
        Axis index (0=X, 1=Y, 2=Z). Default is 1 (Y axis top boundary).

    Returns
    -------
    ds : ufl.Measure
        Subdomain measure restricted to tagged boundary facets (tag=1).
    """
    x_coords  = domain.geometry.x[:, coord]
    x_max     = np.max(x_coords)
    facets    = locate_entities(domain, domain.topology.dim - 1,
                                lambda x: x[coord] >= (x_max - 1e-8))
    facet_tag = meshtags(domain, domain.topology.dim - 1,
                         facets, np.full_like(facets, 1))
    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


# ===========================================================================
# Variational form and Newton solver
# ===========================================================================
def build_solver(domain, V, model: PlasticityModel, state: PlasticState, bcs,
                 quadrature_degree: int = 1):
    """
    Assemble non-linear weak form and construct PETSc Newton solver.

    Variational Formulation
    -----------------------
    Find u ∈ V such that:
        F(u; v) = ∫ ( σ(u) : sym(∇v) ) dx = 0    ∀ v ∈ V_0

    Consistent Tangent
    ------------------
    J = dF/du = ufl.derivative(F, u, du)

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    V : fem.FunctionSpace
        Vector displacement space.
    model : PlasticityModel
        Constitutive model supplying Cauchy stress computation.
    state : PlasticState
        Current internal history state.
    bcs : list of DirichletBC
        Active boundary conditions.
    quadrature_degree : int
        Integration rule degree (default=1).

    Returns
    -------
    uh : fem.Function
        Displacement solution vector (updated in-place).
    problem : NewtonSolverNonlinearProblem
        FEniCSx PETSc non-linear problem definition.
    solver : NewtonSolver
        Configured PETSc Newton solver instance.
    """
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    v  = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    uh = fem.Function(V)

    # Weak form residual F
    F = ufl.inner(model.cauchy_stress(state, uh), ufl.sym(ufl.grad(v))) * dx
    # Automatic symbolic differentiation for exact tangent matrix J
    J = ufl.derivative(F, uh, du)

    problem               = NewtonSolverNonlinearProblem(F, uh, bcs=bcs, J=J)
    solver                = NewtonSolver(domain.comm, problem)
    solver.atol           = 1e-8
    solver.rtol           = 1e-8
    solver.max_it         = 50
    solver.convergence_criterion = "incremental"
    return uh, problem, solver



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
