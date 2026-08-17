"""
femu_generic.py
---------------
Generic FEMU (Finite Element Model Updating) optimisation framework.

This module provides:
  1. Parameter normalisation/denormalisation helpers for unit hypercube mapping [0, 1]^N.
  2. Model registry (`MODEL_REGISTRY`) mapping model names to constructor closures, 
     default parameter values, and physical parameter bounds.
  3. Joint residual computation (`compute_u_f_residuals_is_imported`) combining DIC displacement 
     field errors and reaction force discrepancies.
  4. Main optimization routine (`femu_res_generic`) executing Scipy `least_squares` 
     with live matplotlib monitoring.

Public API
~~~~~~~~~~
femu_res_generic(PVD_FILE, FORCE_FILE, model_name, ...)  -> OptimizeResult
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from dolfinx import fem
from scipy.optimize import Bounds, least_squares

from hill48_model import Hill48Model
from plasticity_simu import (
    DEFAULT_CONFIG,
    get_vtu_files_from_pvd,
    load_domain_from_vtu,
    run_simulation_bc_vtu_fast,
)
from simu_tools import (
    ElasticModel,
    J2IsotropicHardening,
    build_function_spaces,
)


# ---------------------------------------------------------------------------
# Parameter normalisation helpers
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
# 1. Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "J2IsotropicHardening": {
        "builder": lambda run_cfg, domain: J2IsotropicHardening(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            Q_var=run_cfg["Q_var"],
            k=run_cfg["k_hardening"],
        ),
        "params_default": {
            "t_start":     0.0,
            "E":           200_000.0,
            "nu":          0.3,
            "sigma_Y":     100.0,
            "Q_var":       50.0,
            "k_hardening": 1_000.0,
            "ux_up":       0.0,
            "uy_up":       0.01,
            "uz_up":       0.0,
            "ux_down":     0.0,
            "uy_down":     -0.01,
            "uz_down":     0.0,
        },
        "bounds": {
            "t_start":     (-0.2, 0.5),
            "E":           (150_000.0, 250_000.0),
            "nu":          (0.2, 0.45),
            "sigma_Y":     (20.0, 300.0),
            "Q_var":       (0.0, 300.0),
            "k_hardening": (10.0, 5_000.0),
            "ux_up":       (-0.01, 0.01),
            "uy_up":       (0.0001, 0.05),
            "uz_up":       (-0.01, 0.01),
            "ux_down":     (-0.01, 0.01),
            "uy_down":     (-0.05, -0.0001),
            "uz_down":     (-0.01, 0.01),
        },
    },
    "Hill48": {
        "builder": lambda run_cfg, domain: Hill48Model(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            H=run_cfg["H"],
            F=run_cfg["F"],
            G=run_cfg["G"],
            L=run_cfg["L"],
            M=run_cfg["M"],
            N=run_cfg["N"],
            Q_var=run_cfg["Q_var"],
            k_hardening=run_cfg["k_hardening"],
        ),
        "params_default": {
            "t_start":     0.0,
            "E":           200_000.0,
            "nu":          0.3,
            "sigma_Y":     100.0,
            "Q_var":       50.0,
            "k_hardening": 1_000.0,
            "F":  0.900, "G": 0.600, "H": 0.400,
            "L":  1.700, "M": 1.300, "N": 1.350,
            "ux_up": 0.0, "uy_up": 0.01, "uz_up": 0.0,
            "ux_down": 0.0, "uy_down": -0.01, "uz_down": 0.0,
        },
        "bounds": {
            "t_start":     (-0.2, 0.5),
            "E":           (150_000.0, 250_000.0),
            "nu":          (0.2, 0.45),
            "sigma_Y":     (20.0, 300.0),
            "Q_var":       (0.0, 300.0),
            "k_hardening": (10.0, 5_000.0),
            "F": (0.0, 2.0), "G": (0.0, 2.0), "H": (0.0, 2.0),
            "L": (0.1, 5.0), "M": (0.1, 5.0), "N": (0.1, 5.0),
            "ux_up": (-0.01, 0.01), "uy_up": (0.0001, 0.05), "uz_up": (-0.01, 0.01),
            "ux_down": (-0.01, 0.01), "uy_down": (-0.05, -0.0001), "uz_down": (-0.01, 0.01),
        },
    },
    "Hill48_2": {
        # Variant: H is constrained to (1 - G) to enforce a normalization condition
        "builder": lambda run_cfg, domain: Hill48Model(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            G=run_cfg["G"],
            F=run_cfg["F"],
            H=1 - run_cfg["G"],
            L=run_cfg["L"],
            M=run_cfg["M"],
            N=run_cfg["N"],
            Q_var=run_cfg["Q_var"],
            k_hardening=run_cfg["k_hardening"],
        ),
        "params_default": {
            "t_start":     0.0,
            "E":           200_000.0,
            "nu":          0.3,
            "sigma_Y":     100.0,
            "Q_var":       50.0,
            "k_hardening": 1_000.0,
            "F":  0.500, "G": 0.500,
            "L":  1.500, "M": 1.500, "N": 1.500,
            "ux_up": 0.0, "uy_up": 0.01, "uz_up": 0.0,
            "ux_down": 0.0, "uy_down": -0.01, "uz_down": 0.0,
        },
        "bounds": {
            "t_start":     (-0.2, 0.5),
            "E":           (150_000.0, 250_000.0),
            "nu":          (0.2, 0.45),
            "sigma_Y":     (20.0, 300.0),
            "Q_var":       (0.0, 300.0),
            "k_hardening": (10.0, 5_000.0),
            "F": (0.0, 1.0), "G": (0.0, 1.0),
            "L": (0.1, 5.0), "M": (0.1, 5.0), "N": (0.1, 5.0),
            "ux_up": (-0.01, 0.01), "uy_up": (0.0001, 0.05), "uz_up": (-0.01, 0.01),
            "ux_down": (-0.01, 0.01), "uy_down": (-0.05, -0.0001), "uz_down": (-0.01, 0.01),
        },
    },
}


# ---------------------------------------------------------------------------
# 2. Mixed displacement + force residual
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


# ---------------------------------------------------------------------------
# 3. Objective function wrapper for optimization iterations
# ---------------------------------------------------------------------------

def compute_residuals_generic_DIC_BC(
    domain, V, W, WT, f_ref, ref_multiblock,
    model_name, free_param_names, free_param_values, fixed_params,
    config=None
):
    """
    Evaluate the FEM model for candidate parameter values and compute residual vector.

    Includes exception handling for Newton solver divergence: if a numerical simulation
    diverges due to unphysical parameter values, a large constant penalty vector (1e3)
    is returned to guide the optimizer away from invalid parameter regions.

    Returns
    -------
    error : np.ndarray
        Residual vector.
    f_sim : np.ndarray
        Simulated reaction force curve.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Merge fixed and free physical parameters
    full_params = {**fixed_params, **dict(zip(free_param_names, free_param_values))}
    run_cfg = {**cfg, **full_params}

    # Reconstruct 3D prescribed velocity vectors
    run_cfg["disp_value_up"] = [
        run_cfg["ux_up"],
        run_cfg["uy_up"],
        run_cfg["uz_up"],
    ]
    run_cfg["disp_value_down"] = [
        run_cfg["ux_down"],
        run_cfg["uy_down"],
        run_cfg["uz_down"],
    ]

    model = model_info["builder"](run_cfg, domain)

    # Determine expected residual length for divergence penalty vector fallback
    ref_grid = ref_multiblock[0]
    is_imported = ref_grid.point_data.get("is_imported", np.ones(len(ref_grid.points)))
    mask_imported = np.isclose(is_imported, 0.1, atol=1e-6)
    if ref_grid.points.shape[1] >= 3:
        mask_z = np.isclose(ref_grid.points[:, 2], 0.0, atol=1e-6)
        n_masked = (mask_imported & mask_z).sum()
    else:
        n_masked = mask_imported.sum()

    vtu_function_name = run_cfg.get("vtu_function_name", "displacement_projected")

    if vtu_function_name in ref_grid.point_data:
        active_comps_len = ref_grid.point_data[vtu_function_name].shape[1]
    else:
        active_comps_len = 3

    expected_u_size = n_masked * active_comps_len * len(ref_multiblock)
    expected_f_size = len(ref_multiblock)
    expected_total_size = expected_u_size + expected_f_size

    try:
        # Run non-linear FE simulation with current material parameters
        f_sim, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)
        
        weight_u = run_cfg.get("weight_u", 1.0)
        weight_f = run_cfg.get("weight_f", 1.0)
        
        error = compute_u_f_residuals_is_imported(
            ref_multiblock, sim_multiblock, f_ref, f_sim,
            vtu_function_name="displacement_projected",
            sim_function_name="displacement",
            weight_u=weight_u,
            weight_f=weight_f
        )
    except Exception as e:
        print(f"--> [Simulation/Newton Divergence] Exception : {e}. Applying penalty error.")
        error = np.ones(expected_total_size) * 1e3
        f_sim = np.zeros_like(f_ref)

    return error, f_sim


# ---------------------------------------------------------------------------
# 4. Main FEMU optimisation loop
# ---------------------------------------------------------------------------

def femu_res_generic(
    PVD_FILE,
    FORCE_FILE,
    model_name,
    free_param_names=None,
    fixed_param_overrides=None,
    params0_overrides=None,
    bounds_overrides=None,
    config=None
):
    """
    Run FEMU (Finite Element Model Updating) parameter identification loop.

    Pre-loads experimental DIC displacement fields (PVD/VTU) and force curves (.npy),
    normalizes free parameters to [0, 1]^N, executes Trust-Region Reflective (TRF)
    least-squares minimization, and updates a live Matplotlib dashboard.

    Parameters
    ----------
    PVD_FILE : str
        Path to PVD file containing projected DIC displacement fields.
    FORCE_FILE : str
        Path to .npy file containing measured force curve.
    model_name : str
        Model key registered in `MODEL_REGISTRY` ("J2IsotropicHardening", "Hill48", "Hill48_2").
    free_param_names : list of str, optional
        Names of parameters to optimize (defaults to all model parameters).
    fixed_param_overrides : dict, optional
        Fixed parameter values held constant.
    params0_overrides : dict, optional
        Initial guesses for optimized parameters.
    bounds_overrides : dict, optional
        Custom (min, max) physical parameter bounds.
    config : dict, optional
        Simulation configuration options (e.g. weights `weight_u`, `weight_f`).

    Returns
    -------
    result : OptimizeResult
        Scipy optimization result object with identified physical parameters attached as `result.x`.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]
    all_param_names = list(model_info["params_default"].keys())

    if free_param_names is None:
        free_param_names = all_param_names
    else:
        unknown = set(free_param_names) - set(all_param_names)
        if unknown:
            raise ValueError(f"Unknown parameters for '{model_name}': {unknown}")

    fixed_param_names = [p for p in all_param_names if p not in free_param_names]

    params_default = {**model_info["params_default"], **(fixed_param_overrides or {})}
    bounds_all = {**model_info["bounds"], **(bounds_overrides or {})}

    fixed_params = {k: params_default[k] for k in fixed_param_names}

    params0_dict = {**params_default, **(params0_overrides or {})}
    params0 = [params0_dict[k] for k in free_param_names]
    bounds_free = [bounds_all[k] for k in free_param_names]

    # Pre-load experimental DIC reference fields into memory
    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    print("Pré-chargement des fichiers VTU de référence en mémoire...")
    ref_multiblock = pv.MultiBlock()
    for f_vtu in vtu_files:
        ref_multiblock.append(pv.read(f_vtu))
    print(f"Chargé {len(ref_multiblock)} pas de temps de référence.")

    # Pre-load experimental force vector
    print(f"Chargement du fichier de forces : {FORCE_FILE}")
    f_ref = np.load(FORCE_FILE)
    print(f"Chargé {len(f_ref)} points de force.")

    cfg = {
        "pvd_file_path": PVD_FILE,
        "num_steps": len(vtu_files) - 1,
        "T": 3.0,
        "weight_u": 1.0,
        "weight_f": 1.0,
        **(config or {})
    }

    # Setup interactive Matplotlib visualization dashboard
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
    history_params = []

    # Scale initial parameter values and bounds to normalized unit range [0, 1]
    params0_norm = normalize_params(params0, bounds_free)
    bounds_norm = Bounds([0.0] * len(params0), [1.0] * len(params0))

    def objective_function(params_norm):
        # Convert normalized optimization variables back to physical values
        params_phys = denormalize_params(params_norm, bounds_free)

        print(f"\n{datetime.now()} | Simu n°{len(history_err)} | Modèle: {model_name}")
        print("Paramètres libres testés :", dict(zip(free_param_names, params_phys)))
        if fixed_params:
            print("Paramètres fixes :", fixed_params)

        residuals, f_sim = compute_residuals_generic_DIC_BC(
            domain, V, W, WT, f_ref, ref_multiblock,
            model_name=model_name,
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
        ftol=1e-7, gtol=1e-8, max_nfev=500, verbose=2, x_scale=1.0, diff_step=5e-3, xtol=None
    )

    plt.ioff()
    plt.show()

    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds_free))
    result_phys.param_names = free_param_names
    result_phys.fixed_params = fixed_params

    return result_phys
