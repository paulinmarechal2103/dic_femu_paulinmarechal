import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, nls, default_scalar_type
from dolfinx.io import XDMFFile
from mpi4py import MPI
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# ==============================================================================
# 1. Mesh and Geometry Setup
# ==============================================================================
L, H = 1.0, 0.2
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    points=[(0.0, 0.0), (L, H)],
    n=[50, 10],
    cell_type=mesh.CellType.triangle
)
gdim = domain.geometry.dim

# ==============================================================================
# 2. Material Parameters (Von Mises + Linear Isotropic Hardening)
# ==============================================================================
E = 200.0e3       # Young's modulus (MPa)
nu = 0.3          # Poisson's ratio
sigma_0 = 250.0   # Yield stress (MPa)
H_mod = 1000.0    # Isotropic hardening modulus (MPa)

# Lamé parameters & bulk modulus
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
K = lmbda + (2.0 / 3.0) * mu

# ==============================================================================
# 3. Finite Element Spaces (Basix API)
# ==============================================================================
# Displacement space (Vector CG1)
v_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(gdim,))
V = fem.functionspace(domain, v_elem)

# History space for cumulative plastic strain p (Scalar DG0)
p_elem = basix.ufl.element("DG", domain.topology.cell_name(), 0)
W_scalar = fem.functionspace(domain, p_elem)

# History space for plastic strain tensor eps_p (Tensor DG0)
ep_elem = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(gdim, gdim))
W_tensor = fem.functionspace(domain, ep_elem)

# ==============================================================================
# 4. Problem Variables
# ==============================================================================
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)

# History variables stored from step n
p_old = fem.Function(W_scalar, name="Cumulative_Plastic_Strain")
eps_p_old = fem.Function(W_tensor, name="Plastic_Strain_Tensor")

# Parameter for prescribed displacement load
disp_bar = fem.Constant(domain, default_scalar_type(0.0))

# ==============================================================================
# 5. Boundary Conditions
# ==============================================================================
left_facets = mesh.locate_entities_boundary(domain, gdim - 1, lambda x: np.isclose(x[0], 0.0))
right_facets = mesh.locate_entities_boundary(domain, gdim - 1, lambda x: np.isclose(x[0], L))

left_dofs = fem.locate_dofs_topological(V, gdim - 1, left_facets)
right_dofs_x = fem.locate_dofs_topological(V.sub(0), gdim - 1, right_facets)

# Clamped left end (ux = 0, uy = 0)
bc_left = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), left_dofs, V)
# Prescribed displacement in x-direction at right end
bc_right = fem.dirichletbc(disp_bar, right_dofs_x, V.sub(0))

bcs = [bc_left, bc_right]

# ==============================================================================
# 6. Constitutive Equations & Radial Return Mapping
# ==============================================================================
def eps(v_field):
    return ufl.sym(ufl.grad(v_field))

def dev(tensor):
    return tensor - (1.0 / 3.0) * ufl.tr(tensor) * ufl.Identity(gdim)

def von_mises(s_tensor):
    # Small term (+1e-10) prevents division-by-zero during symbolic differentiation
    return ufl.sqrt(1.5 * ufl.inner(s_tensor, s_tensor) + 1.0e-10)

def return_mapping(u_field, eps_p_n, p_n):
    # Trial strain and trial deviatoric stress
    ep_trial = eps(u_field) - eps_p_n
    tr_ep = ufl.tr(ep_trial)
    s_trial = 2.0 * mu * dev(ep_trial)
    sigma_eq_trial = von_mises(s_trial)

    # Yield criterion
    f_trial = sigma_eq_trial - (sigma_0 + H_mod * p_n)

    # Plastic increment (MacAulay bracket)
    dp = ufl.conditional(ufl.gt(f_trial, 0.0), f_trial / (3.0 * mu + H_mod), 0.0)

    # Updated stress tensor
    sigma = K * tr_ep * ufl.Identity(gdim) + (1.0 - 3.0 * mu * dp / sigma_eq_trial) * s_trial

    # Updated state variables
    n_elast = s_trial / sigma_eq_trial
    eps_p_new = eps_p_n + 1.5 * dp * n_elast
    p_new = p_n + dp

    return sigma, eps_p_new, p_new

# Compute updated stress and state expressions
sigma, eps_p_new, p_new = return_mapping(u, eps_p_old, p_old)

# ==============================================================================
# 7. Variational Formulation & Solver Setup
# ==============================================================================
# Définition de la forme résiduelle et du jacobien
import dolfinx
dx = ufl.Measure("dx", domain=domain)
F = ufl.inner(sigma, eps(v)) * dx
J = ufl.derivative(F, u)

# 1. Instanciation du problème non-linéaire (gère SNES en interne)
problem = dolfinx.fem.petsc.NonlinearProblem(
    F, u, bcs=bcs, J=J, petsc_options_prefix="elastoplastic_"
)

# 2. Configuration directe du solveur SNES de PETSc
snes = problem._snes
snes.setType("newtonls")  # Newton avec line-search
snes.setTolerances(rtol=1e-6, max_it=20)
snes.getKSP().setType("preonly")
snes.getKSP().getPC().setType("lu")
# ==============================================================================
# 8. Load Stepping & Incremental Solution
# ==============================================================================
num_steps = 20
max_disp = 0.005  # Total horizontal extension (m)
load_steps = np.linspace(0.0, max_disp, num_steps + 1)

# Expressions for post-processing and updating state variables
p_expr = fem.Expression(p_new, W_scalar.element.interpolation_points)
ep_expr = fem.Expression(eps_p_new, W_tensor.element.interpolation_points)

# XDMF File for visualization
xdmf = XDMFFile(domain.comm, "elastoplasticity_results.xdmf", "w")
xdmf.write_mesh(domain)

print("--- Starting Elastoplastic Simulation ---")
for step, t in enumerate(load_steps[1:], start=1):
    disp_bar.value = default_scalar_type(t)

    # La résolution s'effectue directement via problem.solve()
    n_iter, converged = problem.solve()
    if not converged:
        raise RuntimeError(f"Solver failed to converge at load step {step}")

    # Update history variables at converged equilibrium state
    p_old.interpolate(p_expr)
    eps_p_old.interpolate(ep_expr)

    # Save output
    xdmf.write_function(u, t)
    xdmf.write_function(p_old, t)

    if domain.comm.rank == 0:
        p_max = np.max(p_old.x.array)
        print(f"Step {step:02d}/{num_steps} | Disp: {t:.4f} m | Newton Iter: {n_iter} | Max Plastic Strain: {p_max:.6e}")

xdmf.close()
print("--- Simulation Completed Successfully ---")