import numpy as np
import pyvista as pv
import ufl
import basix.ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem import form, Expression
from dolfinx.fem.petsc import assemble_vector, NonlinearProblem
from mpi4py import MPI
import petsc4py.PETSc as PETSc

from plasticity_simu import get_vtu_files_from_pvd


def fenicsx_elastoplastic_solver(domain, V, W_scalar, W_tensor, run_cfg):
    """
    Stateless FEniCSx J2 elastoplastic solver compliant with the FEMU toolbox interface:
    f_sim, sim_multiblock = solver(domain, V, W, WT, run_cfg)
    """
    # 1. Safely extract material & simulation parameters
    E = float(run_cfg.get("E", 200000.0))
    nu = float(run_cfg.get("nu", 0.3))
    sigma_0 = float(run_cfg.get("sigma_0", run_cfg.get("sigma_Y", 250.0)))
    H_mod = float(run_cfg.get("H_mod", run_cfg.get("k_hardening", 1000.0)))
    pvd_path = run_cfg["pvd_file_path"]

    vtu_files = get_vtu_files_from_pvd(pvd_path)
    num_steps = min(int(run_cfg.get("num_steps", len(vtu_files))), len(vtu_files))

    gdim = domain.geometry.dim
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    K = lmbda + (2.0 / 3.0) * mu

    # 2. Locate boundary DOFs (+ymax and -ymax boundaries)
    y_coords = domain.geometry.x[:, 1]
    ymax = np.max(y_coords)
    ymin = np.min(y_coords)

    def boundary_ymax(x):
        return np.isclose(x[1], ymax, atol=1e-5)

    def boundary_sides(x):
        return np.isclose(x[1], ymax, atol=1e-5) | np.isclose(x[1], ymin, atol=1e-5)

    dofs_ymax = fem.locate_dofs_geometrical(V, boundary_ymax)
    bc_dofs_all = fem.locate_dofs_geometrical(V, boundary_sides)

    # 3. Define Variational Problem & Return Mapping
    u = fem.Function(V, name="Displacement")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    p_old = fem.Function(W_scalar, name="Cumulative_Plastic_Strain")
    eps_p_old = fem.Function(W_tensor, name="Plastic_Strain_Tensor")

    def eps(v_field):
        return ufl.sym(ufl.grad(v_field))

    def dev(tensor):
        return tensor - (1.0 / 3.0) * ufl.tr(tensor) * ufl.Identity(gdim)

    def von_mises(s_tensor):
        return ufl.sqrt(1.5 * ufl.inner(s_tensor, s_tensor) + 1.0e-10)

    # Radial return mapping formulation
    ep_trial = eps(u) - eps_p_old
    tr_ep = ufl.tr(ep_trial)
    s_trial = 2.0 * mu * dev(ep_trial)
    sigma_eq_trial = von_mises(s_trial)

    f_trial = sigma_eq_trial - (sigma_0 + H_mod * p_old)
    dp = ufl.conditional(ufl.gt(f_trial, 0.0), f_trial / (3.0 * mu + H_mod), 0.0)

    sigma = K * tr_ep * ufl.Identity(gdim) + (1.0 - 3.0 * mu * dp / sigma_eq_trial) * s_trial

    n_elast = s_trial / sigma_eq_trial
    eps_p_new = eps_p_old + 1.5 * dp * n_elast
    p_new = p_old + dp

    dx = ufl.Measure("dx", domain=domain)
    F_form = ufl.inner(sigma, eps(v)) * dx
    
    eps_du = eps(du)
    sigma_du = K * ufl.tr(eps_du) * ufl.Identity(gdim) + 2.0 * mu * dev(eps_du)
    J_form = ufl.inner(sigma_du, eps(v)) * dx

    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bc_dofs_all)

    # 4. Instantiate NonlinearProblem and configure SNES
    problem = NonlinearProblem(
        F_form, u, bcs=[bc], J=J_form, petsc_options_prefix="elastoplastic_"
    )
    snes = problem._snes
    snes.setType("newtonls")
    snes.setTolerances(rtol=1e-6, max_it=25)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")

    L_compiled = form(F_form)

    # Expressions for plastic state updates (valeur totale corrigée)
    p_expr = Expression(p_new, W_scalar.element.interpolation_points)
    ep_expr = Expression(eps_p_new, W_tensor.element.interpolation_points)

    f_sim = []
    sim_multiblock = pv.MultiBlock()
    bs = V.dofmap.bs
    dofs_ymax_y = dofs_ymax[dofs_ymax % bs == 1]

    # 5. Incremental Time-Stepping Loop
    for t in range(num_steps):
        ref_grid = pv.read(vtu_files[t])
        disp_proj = ref_grid.point_data["displacement_projected"]

        u_bc.x.array.fill(0.0)
        for dof in bc_dofs_all:
            node_id = dof // bs
            comp = dof % bs
            # Coordonnée du nœud dans le domaine FEniCSx
            coord = domain.geometry.x[node_id]
            
            # Trouver le point correspondant dans PyVista (ou utiliser directement l'index si l'ordre est conservé)
            # Si l'ordre des points du VTU n'a pas permuté :
            if disp_proj.ndim == 2:
                u_bc.x.array[dof] = disp_proj[node_id, comp]
            else:
                u_bc.x.array[dof] = disp_proj[node_id]

        u_bc.x.scatter_forward()

        # Solve nonlinear step directly via problem.solve()
        problem.solve()
        
        # Check convergence via PETSc SNES
        converged = snes.getConvergedReason() > 0
        if not converged:
            raise RuntimeError(f"Solver failed to converge at load step {t} (Reason: {snes.getConvergedReason()})")
        
        # Mise à jour des variables d'historique (Affectation directe de la valeur totale)
        p_val = fem.Function(W_scalar)
        p_val.interpolate(p_expr)
        p_old.x.array[:] = p_val.x.array[:]

        eps_p_val = fem.Function(W_tensor)
        eps_p_val.interpolate(ep_expr)
        eps_p_old.x.array[:] = eps_p_val.x.array[:]

        # Reaction force vector assembly
        R = assemble_vector(L_compiled)
        R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        f_sim_t = float(np.sum(R.array[dofs_ymax_y]))
        f_sim.append(f_sim_t)

        # Build PyVista mesh preserving node correspondence
        sim_grid = ref_grid.copy()
        u_nodes = u.x.array.reshape(-1, bs)

        if ref_grid.points.shape[1] == 3 and bs == 2:
            u_disp_3d = np.zeros((ref_grid.n_points, 3), dtype=np.float64)
            u_disp_3d[:, :2] = u_nodes
            sim_grid.point_data["displacement"] = u_disp_3d
        else:
            sim_grid.point_data["displacement"] = u_nodes

        sim_multiblock.append(sim_grid)

    return np.array(f_sim, dtype=np.float64), sim_multiblock


if __name__ == "__main__":
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"

    run_cfg = {
        "E": 200000.0,
        "nu": 0.3,
        "sigma_0": 250.0,
        "H_mod": 1000.0,
        "pvd_file_path": PVD_FILE,
        "num_steps": 55,
    }

    from simu_tools import build_function_spaces,animer_deformee
    from plasticity_simu import get_vtu_files_from_pvd, load_domain_from_vtu

    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    forces, multiblock = fenicsx_elastoplastic_solver(domain, V, W, WT, run_cfg)
    print(f"Simulation completed successfully. Evaluated force steps: {len(forces)}")
    animer_deformee(multiblock, factor=10.0, component=None, fps=20)

    import matplotlib.pyplot as plt
    plt.plot(forces)
    plt.xlabel("Load Step")
    plt.ylabel("Reaction Force")
    plt.title("Reaction Force vs Load Step")
    plt.grid()
    plt.show()
