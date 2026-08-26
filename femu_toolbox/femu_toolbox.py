"""
femu_toolbox.py
----------------
Generic, solver-agnostic FEMU (Finite Element Model Updating) toolbox.

Unlike `femu_generic.py`, this module has no knowledge of any particular
constitutive model: there is no `MODEL_REGISTRY` and no import of a
specific material model class (`Hill48Model`, `J2IsotropicHardening`, ...).
Instead, the user supplies their own finite-element `solver` callable,
which is responsible for building whatever material model it needs from
the current parameter set and for running the FE simulation.

This makes the toolbox reusable across projects/models: only the `solver`
argument changes.

Public API
~~~~~~~~~~
femu_res_toolbox(PVD_file, FORCE_file, Optim_params, Bounds, solver, config=None)
    -> scipy.optimize.OptimizeResult
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import least_squares

from plasticity_simu import DEFAULT_CONFIG, get_vtu_files_from_pvd, load_domain_from_vtu
from simu_tools import build_function_spaces


# ---------------------------------------------------------------------------
# 1. Parameter normalisation helpers
# ---------------------------------------------------------------------------

def normalize_params(params, bounds):
    """
    Map physical parameters to the normalized unit hypercube [0, 1].

    Parameters
    ----------
    params : list of float
        Physical parameter values.
    bounds : list of tuple (min, max)
        Physical parameter lower and upper bounds.

    Returns
    -------
    list of float
        Normalized parameter values in range [0, 1].
    """
    return [
        (params[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        for i in range(len(bounds))
    ]


def denormalize_params(params_norm, bounds):
    """
    Map normalized unit hypercube values [0, 1] back to physical units.

    Parameters
    ----------
    params_norm : list of float
        Normalized parameter values in range [0, 1].
    bounds : list of tuple (min, max)
        Physical parameter lower and upper bounds.

    Returns
    -------
    list of float
        Physical parameter values.
    """
    return [
        params_norm[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        for i in range(len(bounds))
    ]


# ---------------------------------------------------------------------------
# 2. DIC displacement + force residuals (model-independent)
# ---------------------------------------------------------------------------

def compute_u_f_residuals_is_imported(
    ref_multiblock, 
    sim_multiblock, 
    f_ref, 
    f_sim, 
    vtu_function_name="displacement_projected",
    sim_function_name=None,
    weight_u=1.0,
    weight_f=1.0,
    normalize=True
):
    """
    Compute concatenated residual vector [res_u | res_f] between DIC reference and FE simulation.

    Masking logic:
      1. Node mask `is_imported ≈ 0.1` (filters out CAD nodes outside experimental DIC region).
      2. Surface mask `z ≈ 0.0` for 3D meshes (filters top surface nodes).

    Normalisation logic:
      Res_u = (u_sim - u_ref) * weight_u / ( std(res_u) * sqrt(len(res_u)) )
      Res_f = (f_sim - f_ref) * weight_f / ( std(f_ref) * sqrt(len(res_f)) )

    Parameters
    ----------
    ref_multiblock : pyvista.MultiBlock
        Experimental DIC displacement fields across timesteps.
    sim_multiblock : pyvista.MultiBlock
        Simulated FEM displacement fields across timesteps.
    f_ref : array-like
        Measured experimental force curve array.
    f_sim : array-like
        Simulated reaction force curve array.
    vtu_function_name : str
        Data key in reference VTU point_data (default: "displacement_projected").
    sim_function_name : str
        Data key in simulation PyVista block point_data (default: "displacement").
    weight_u, weight_f : float
        Relative weights balancing kinematic field error and force error.
    normalize : bool
        Whether to divide residuals by signal standard deviation and vector length.

    Returns
    -------
    residuals : np.ndarray
        1D concatenated residual array.
    """
    res_u_list = []
    f_ref = np.asarray(f_ref, dtype=np.float64)
    f_sim = np.asarray(f_sim, dtype=np.float64)
    
    num_steps = min(len(ref_multiblock), len(sim_multiblock), len(f_ref), len(f_sim))
    
    if sim_function_name is None:
        sim_function_name = vtu_function_name

    ref_grid_0 = ref_multiblock[0]
    sim_grid_0 = sim_multiblock[0]

    if vtu_function_name not in ref_grid_0.point_data:
        raise KeyError(
            f"Field '{vtu_function_name}' not found in REF point_data. "
            f"Available keys: {list(ref_grid_0.point_data.keys())}"
        )
    if sim_function_name not in sim_grid_0.point_data:
        raise KeyError(
            f"Field '{sim_function_name}' not found in SIM point_data. "
            f"Available keys: {list(sim_grid_0.point_data.keys())}"
        )

    # Loop over timesteps to extract valid surface nodes and compute errors
    for t in range(num_steps):
        ref_grid = ref_multiblock[t]
        sim_grid = sim_multiblock[t]
        
        # 1. Mask for nodes within imported DIC region
        if "is_imported" in ref_grid.point_data:
            is_imported = ref_grid.point_data["is_imported"]
            mask_imported = np.isclose(is_imported, 0.1, atol=1e-6)
        else:
            mask_imported = np.ones(ref_grid.n_points, dtype=bool)
            
        # 2. Surface mask z=0 for 3D meshes
        if ref_grid.points.shape[1] >= 3:
            mask_z = np.isclose(ref_grid.points[:, 2], 0.0, atol=1e-6)
            mask = mask_imported & mask_z
        else:
            mask = mask_imported

        u_ref_t = ref_grid.point_data[vtu_function_name][mask]
        u_sim_t = sim_grid.point_data[sim_function_name][mask]
        
        diff_u = (u_sim_t - u_ref_t).ravel()
        res_u_list.append(diff_u)
        
    res_u = np.concatenate(res_u_list)
    res_f = (f_sim[:num_steps] - f_ref[:num_steps]).ravel()
    
    # Apply standard deviation scaling and relative weighting
    if normalize:
        scale_u = np.std(res_u) if np.std(res_u) > 1e-12 else 1.0
        scale_f = np.std(f_ref[:num_steps]) if np.std(f_ref[:num_steps]) > 1e-12 else 1.0
        
        norm_factor_u = (1.0 / (scale_u * np.sqrt(len(res_u)))) * weight_u
        norm_factor_f = (1.0 / (scale_f * np.sqrt(len(res_f)))) * weight_f
        
        res_u = res_u * norm_factor_u
        res_f = res_f * norm_factor_f
    else:
        res_u = res_u * weight_u
        res_f = res_f * weight_f

    return np.concatenate([res_u, res_f])




def compute_residuals_toolbox(
    f_ref,
    ref_multiblock,
    solver,
    free_param_names,
    free_param_values,
    fixed_params,
    config=None,
):
    """
    Recompose parameter set, invoke solver(run_cfg), and compute residuals.
    """
    if not callable(solver):
        raise TypeError("`solver` must be a callable with signature: solver(run_cfg)")

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    full_params = {**fixed_params, **dict(zip(free_param_names, free_param_values))}
    run_cfg = {**cfg, **full_params}

    # Estimate penalty size directly from PyVista reference grid (solver-agnostic)
    ref_grid = ref_multiblock[0]
    is_imported = ref_grid.point_data.get("is_imported", np.ones(len(ref_grid.points)))
    mask_imported = np.isclose(is_imported, 0.1, atol=1e-6)
    if ref_grid.points.shape[1] >= 3:
        mask_z = np.isclose(ref_grid.points[:, 2], 0.0, atol=1e-6)
        n_masked = (mask_imported & mask_z).sum()
    else:
        n_masked = mask_imported.sum()

    # Determine vector dimension from reference data array shape
    ref_field = ref_grid.point_data.get("displacement_projected", None)
    active_comps_len = ref_field.shape[1] if (ref_field is not None and ref_field.ndim > 1) else 1

    expected_u_size = n_masked * active_comps_len * len(ref_multiblock)
    expected_f_size = len(ref_multiblock)
    expected_total_size = expected_u_size + expected_f_size

    try:
        # Generic solver invocation
        f_sim, sim_multiblock = solver(run_cfg)

        weight_u = run_cfg.get("weight_u", 1.0)
        weight_f = run_cfg.get("weight_f", 1.0)

        error = compute_u_f_residuals_is_imported(
            ref_multiblock,
            sim_multiblock,
            f_ref,
            f_sim,
            vtu_function_name="displacement_projected",
            sim_function_name="displacement",
            weight_u=weight_u,
            weight_f=weight_f,
        )
    except Exception as e:
        print(f"--> [Solver Exception] {e}. Applying penalty error.")
        error = np.ones(expected_total_size) * 1e3
        f_sim = np.zeros_like(f_ref)

    return error, f_sim


def femu_res_toolbox(
    PVD_file,
    FORCE_file,
    Optim_params,
    Bounds,
    solver,
    config=None,
):
    """
    Identify material parameters by FEMU (Finite Element Model Updating),
    fitting a user-supplied FE `solver` against experimental DIC
    displacement fields and a measured force curve.

    This function is fully model-agnostic: it doesn't know or care what
    constitutive model is being fitted. That knowledge lives entirely in
    `solver`, which the caller constructs (e.g. as a small wrapper/closure
    around their own simulation routine).

    Parameters
    ----------
    PVD_file : str
        Path to the .pvd file describing the reference DIC time series
        (experimental displacement fields).
    FORCE_file : str
        Path to a .npy file containing the measured experimental force curve.
    Optim_params : dict
        Every material parameter involved in the simulation, mapped to a
        single value each:
          - for parameters to be identified, the value is the *initial
            guess* used to start the optimization;
          - for parameters that should stay constant, the value is the
            *fixed value* used throughout the simulation.
        Whether a parameter is free or fixed is decided by `Bounds` (see
        below), not by its position in this dict.
    Bounds : dict
        `{param_name: (min, max)}` physical bounds for the parameters to
        optimize. Only parameters that appear here (with a key also present
        in `Optim_params`) are treated as free/optimized; every other key
        in `Optim_params` is treated as fixed. `Bounds` must not contain
        names that are absent from `Optim_params`.
    solver : callable
        User-supplied finite-element solver with signature:

            f_sim, sim_multiblock = solver(domain, V, W, WT, run_cfg)

        where `run_cfg` is a dict merging `config`, the internal simulation
        defaults, and every entry of `Optim_params` (fixed values plus the
        free values currently being tested). The solver is responsible for
        building whatever constitutive model it needs from `run_cfg` and
        for returning the simulated reaction force curve and the simulated
        displacement fields (as a `pyvista.MultiBlock`, with per-timestep
        blocks exposing a `"displacement"` point-data field).
    config : dict, optional
        Additional simulation configuration options (e.g. number of load
        steps, total simulated time `T`, residual weights `weight_u` /
        `weight_f`, ...), merged on top of the internal defaults.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result of `scipy.optimize.least_squares`, with in addition:
        - `x` : physical (denormalized) values of the identified
          parameters, in the order of the free parameters found in
          `Optim_params` / `Bounds`;
        - `param_names` : names of the identified parameters, aligned
          with `x`;
        - `fixed_params` : dict of the parameters that were held fixed
          and their value.

    Raises
    ------
    ValueError
        If `Optim_params` is empty, if `Bounds` references a parameter not
        present in `Optim_params`, if no parameter ends up free (i.e. none
        of the `Optim_params` keys have a matching entry in `Bounds`), or
        if any bound is invalid (`max <= min`).
    TypeError
        If `solver` is not callable.

    Examples
    --------
    >>> vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    >>> domain = load_domain_from_vtu(vtu_files[0])
    >>> V, W, WT = build_function_spaces(domain)
    >>> def my_solver(run_cfg):
    ...     model = J2IsotropicHardening(
    ...         elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
    ...         sigma_Y=run_cfg["sigma_Y"], Q_var=run_cfg["Q_var"], k=run_cfg["k_hardening"],
    ...     )
    ...     return run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)
    >>> result = femu_res_toolbox(
    ...     "ref.pvd", "force.npy",
    ...     Optim_params={"E": 200_000.0, "nu": 0.3, "sigma_Y": 100.0,
    ...                    "Q_var": 50.0, "k_hardening": 1_000.0},
    ...     Bounds={"E": (150_000.0, 250_000.0), "sigma_Y": (20.0, 300.0)},
    ...     solver=my_solver,
    ... )
    # Here "nu", "Q_var" and "k_hardening" have no entry in Bounds, so they
    # stay fixed at their Optim_params value; only "E" and "sigma_Y" are
    # identified by the optimizer.
    """
    if not isinstance(Optim_params, dict) or not Optim_params:
        raise ValueError("`Optim_params` must be a non-empty dict of {param_name: initial_or_fixed_value}.")

    Bounds = Bounds or {}
    if not isinstance(Bounds, dict):
        raise ValueError("`Bounds` must be a dict of {param_name: (min, max)}.")

    unknown_bounds = set(Bounds) - set(Optim_params)
    if unknown_bounds:
        raise ValueError(
            f"`Bounds` contains parameter(s) not present in `Optim_params`: {sorted(unknown_bounds)}"
        )

    # A parameter is "free" (optimized) iff it has a matching Bounds entry;
    # otherwise it is treated as fixed at its Optim_params value.
    free_param_names = [name for name in Optim_params if name in Bounds]
    fixed_param_names = [name for name in Optim_params if name not in Bounds]

    if not free_param_names:
        raise ValueError(
            "No free parameters to optimize: none of the entries in `Optim_params` "
            "have a matching entry in `Bounds`."
        )

    for name in free_param_names:
        lo, hi = Bounds[name]
        if hi <= lo:
            raise ValueError(f"Invalid bounds for parameter '{name}': ({lo}, {hi}).")

    if not callable(solver):
        raise TypeError("`solver` must be a callable with signature solver(domain, V, W, WT, run_cfg).")

    fixed_params = {k: Optim_params[k] for k in fixed_param_names}
    params0 = [Optim_params[k] for k in free_param_names]
    bounds_free = [tuple(Bounds[k]) for k in free_param_names]

    # --- 1. Load reference VTU files described by the PVD ---
    vtu_files = get_vtu_files_from_pvd(PVD_file)

    print("Pre-loading reference VTU files into memory...")
    ref_multiblock = pv.MultiBlock()
    for f_vtu in vtu_files:
        ref_multiblock.append(pv.read(f_vtu))

    f_ref = np.load(FORCE_file)

    cfg = {
        "pvd_file_path": PVD_file,
        "num_steps": len(vtu_files) - 1,
        "T": 3.0,
        "weight_u": 1.0,
        "weight_f": 1.0,
        **(config or {}),
    }

    # --- Interactive dashboard setup (error, force fitting, per-param traces) ---
    n_free = len(free_param_names)
    n_total_plots = n_free + 2
    n_cols = 4
    n_rows = int(np.ceil(n_total_plots / n_cols))

    plt.ion()
    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    ax_err = fig.add_subplot(gs[0, 0])
    ax_force = fig.add_subplot(gs[0, 1])

    ax_params = []
    for i in range(n_free):
        slot_idx = i + 2
        row, col = divmod(slot_idx, n_cols)
        ax_params.append(fig.add_subplot(gs[row, col]))

    history_err = []
    vtu_files = get_vtu_files_from_pvd(PVD_file)

    # --- 2. Load DOLFINx domain + function spaces ---
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    # --- 3. Pre-load experimental DIC reference fields into memory ---
    print("Pré-chargement des fichiers VTU de référence en mémoire...")
    ref_multiblock = pv.MultiBlock()
    for f_vtu in vtu_files:
        ref_multiblock.append(pv.read(f_vtu))
    print(f"Chargé {len(ref_multiblock)} pas de temps de référence.")

    # --- 4. Pre-load experimental force curve ---
    print(f"Chargement du fichier de forces : {FORCE_file}")
    f_ref = np.load(FORCE_file)
    print(f"Chargé {len(f_ref)} points de force.")

    cfg = {
        "pvd_file_path": PVD_file,
        "num_steps": len(vtu_files) - 1,
        "T": 3.0,
        "weight_u": 1.0,
        "weight_f": 1.0,
        **(config or {}),
    }
    history_params = []

    # --- Normalisation ---
    params0_norm = normalize_params(params0, bounds_free)
    bounds_norm = ScipyBounds([0.0] * len(params0), [1.0] * len(params0))

    def objective_function(params_norm):
        params_phys = denormalize_params(params_norm, bounds_free)

        print(f"\n{datetime.now()} | Simu n°{len(history_err)}")
        print("Paramètres libres testés :", dict(zip(free_param_names, params_phys)))
        if fixed_params:
            print("Paramètres fixes :", fixed_params)

        residuals, f_sim = compute_residuals_toolbox(
            f_ref=f_ref,
            ref_multiblock=ref_multiblock,
            solver=solver,
            free_param_names=free_param_names,
            free_param_values=params_phys,
            fixed_params=fixed_params,
            config=cfg,
        )

        error_scalar = np.sum(np.square(residuals))
        history_err.append(error_scalar)
        history_params.append(params_phys)
        data_p = np.array(history_params)

        # Update live visual plots
        try:
            ax_err.clear()
            ax_err.plot(history_err, color='firebrick', lw=1.5)
            ax_err.set_yscale('log')
            ax_err.set_title(r"Norme Résidus (Log $\sum r^2$)")
            ax_err.grid(True, which="both", ls="-", alpha=0.2)

            ax_force.clear()
            steps_ref = np.arange(len(f_ref))
            steps_sim = np.arange(len(f_sim))
            ax_force.plot(steps_ref, f_ref, 'k--', label="F exp", lw=1.5)
            ax_force.plot(steps_sim, f_sim, 'r-', label="F sim", lw=1.5)
            ax_force.set_title("Fitting Force")
            ax_force.set_xlabel("Pas de temps")
            ax_force.set_ylabel("Force [N]")
            ax_force.legend(fontsize=8)
            ax_force.grid(True, alpha=0.2)

            for i, name in enumerate(free_param_names):
                ax_params[i].clear()
                ax_params[i].plot(data_p[:, i], color='royalblue')
                ax_params[i].set_title(f"{name}: {params_phys[i]:.2e}", fontsize=9)
                ax_params[i].grid(True, alpha=0.2)

            plt.tight_layout()
            plt.pause(0.001)
        except Exception as e_plot:
            print(f"Erreur d'affichage : {e_plot}")

        print(f"Norme des résidus : {error_scalar:.6e}")
        return residuals

    # Execute Trust-Region Reflective non-linear least-squares optimization
    result_norm = least_squares(
        objective_function,
        params0_norm,
        method='trf',
        bounds=bounds_norm,
        ftol=1e-7, gtol=1e-8, max_nfev=500, verbose=2, x_scale=1.0, diff_step=5e-3, xtol=None,
    )

    plt.ioff()
    plt.show()

    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds_free))
    result_phys.param_names = free_param_names
    result_phys.fixed_params = fixed_params

    return result_phys









# ---------------------------------------------------------------------------
# 5. Manual test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Model-specific bits live only here, in the solver you build. Nothing
    # in femu_toolbox.py needs to change if you swap J2IsotropicHardening
    # for Hill48Model (or any other model): just write a different solver.
    # ------------------------------------------------------------------
    from simu_tools import ElasticModel, J2IsotropicHardening
    from hill48_model import Hill48Model
    from plasticity_simu import run_simulation_bc_vtu_fast
    from elastoplastic_test import fenicsx_elastoplastic_solver


    PVD_FILE   = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    FORCE_FILE = "MAINTEST/pyvista_exports/csv_projection_A305_FULL/forces_sample_A305.npy"

    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)
    
    def my_solver(run_cfg):
        """
        Build a J2 isotropic-hardening model from the current parameter
        set in `run_cfg` and run the FE simulation.
 
        Must return (f_sim, sim_multiblock), exactly what
        `run_simulation_bc_vtu_fast` already returns.
        """
        model = J2IsotropicHardening(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            Q_var=run_cfg["Q_var"],
            k=run_cfg["k_hardening"],
        )
        f_sim, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)

        return 10*np.array(f_sim), sim_multiblock
 
    def my_solver2(domain, V, W, WT, run_cfg):
        """
        Build a J2 isotropic-hardening model from the current parameter
        set in `run_cfg` and run the FE simulation.
 
        Must return (f_sim, sim_multiblock), exactly what
        `run_simulation_bc_vtu_fast` already returns.
        """
        model = Hill48Model(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            Q_var=run_cfg["Q_var"],
            k_hardening=run_cfg["k_hardening"],
            F=run_cfg["F"],
            G=run_cfg["G"],
            H=1-run_cfg["G"],
            L=1.5,
            M=1.5,
            N=run_cfg["N"],
        )
        return run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)

    # --- Inputs: adjust these paths to your own data ---

 
    # Every parameter the model needs. Values are the initial guess for
    # parameters you want identified, or the fixed value otherwise.
    Optim_params = {
        "E": 210_000.0,          # will be identified (has bounds below)
        "nu": 0.30,               # fixed: no entry in Bounds
        "sigma_Y": 200.0,         # will be identified
        "Q_var": 50.0,            # fixed
        "k_hardening": 500.0,   # fixed
    }
 
    # Only parameters listed here are optimized. Anything in Optim_params
    # that is NOT listed here is treated as fixed automatically.
    Bounds = {
        "Q_var": (10.0, 1000.0),
        "sigma_Y": (10.0, 1000.0),
        "k_hardening": (10.0, 1_500.0),
        "E": (100_000.0, 300_000.0),
    }
 
    config = {
        "num_steps": 54,
        "T": 3.0,
        "weight_u": 1.0,
        "weight_f": 3.0,
    }
 
    result = femu_res_toolbox(
        PVD_file=PVD_FILE,
        FORCE_file=FORCE_FILE,
        Optim_params=Optim_params,
        Bounds=Bounds,
        solver=my_solver,
        config=config,
    )
 
    print("\n" + "=" * 60)
    print("Identification terminée.")
    for name, val in zip(result.param_names, result.x):
        print(f"  {name:>15s} = {val:.4f}")
    if result.fixed_params:
        print("Paramètres fixes utilisés :")
        for name, val in result.fixed_params.items():
            print(f"  {name:>15s} = {val}")
    print(f"  cout final (sum r^2) = {float(np.sum(result.fun ** 2)):.6e}")
    print("=" * 60)
