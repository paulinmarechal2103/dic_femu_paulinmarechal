import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import least_squares, Bounds
from scipy.spatial import KDTree
import dolfinx
from dolfinx import fem, mesh
from petsc4py import PETSc

from femu import *
from femu_DIC import *
from plasticity_simu_DIC_BC import *

# ---------------------------------------------------------------------------
# 1. Registre des modèles disponibles (Comportement + CLs Vectorielles)
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
            # Paramètres matériau
            "E": 200_000.0,
            "nu": 0.3,
            "sigma_Y": 100.0,
            "Q_var": 50.0,
            "k_hardening": 1_000.0,
            # Composantes du vecteur vitesse/déplacement par s [mm/s] (Bord Supérieur)
            "ux_up": 0.0,
            "uy_up": 0.01,
            "uz_up": 0.0,
            # Composantes du vecteur vitesse/déplacement par s [mm/s] (Bord Inférieur)
            "ux_down": 0.0,
            "uy_down": -0.01,
            "uz_down": 0.0,
        },
        "bounds": {
            "E": (150_000.0, 250_000.0),
            "nu": (0.2, 0.45),
            "sigma_Y": (20.0, 300.0),
            "Q_var": (0.0, 300.0),
            "k_hardening": (10.0, 5_000.0),
            # Bornes pour les composantes vectorielles de CL
            "ux_up": (-0.01, 0.01),
            "uy_up": (0.0001, 0.05),
            "uz_up": (-0.01, 0.01),
            "ux_down": (-0.01, 0.01),
            "uy_down": (-0.05, -0.0001),
            "uz_down": (-0.01, 0.01),
        },
    },
}

# ---------------------------------------------------------------------------
# 2. Calcul des résidus mixtes (Déplacement DIC + Force)
# ---------------------------------------------------------------------------

def compute_u_f_residuals_is_imported(
        ref_multiblock, 
        sim_multiblock, 
        f_ref, 
        f_sim, 
        vtu_function_name="displacement_projected",
        sim_function_name=None,  # Si le nom dans la simu diffère de la ref
        weight_u=1.0,
        weight_f=1.0,
        normalize=True):
    
    res_u_list = []
    f_ref = np.asarray(f_ref, dtype=np.float64)
    f_sim = np.asarray(f_sim, dtype=np.float64)
    
    num_steps = min(len(ref_multiblock), len(sim_multiblock), len(f_ref), len(f_sim))
    
    # Nom du champ dans la simulation (par défaut le même que la référence)
    if sim_function_name is None:
        sim_function_name = vtu_function_name

    # Diagnostic au premier pas si la clé manque
    ref_grid_0 = ref_multiblock[0]
    sim_grid_0 = sim_multiblock[0]

    if vtu_function_name not in ref_grid_0.point_data:
        raise KeyError(
            f"Champ '{vtu_function_name}' introuvable dans REF point_data. "
            f"Clés disponibles dans REF : {list(ref_grid_0.point_data.keys())}"
        )
    if sim_function_name not in sim_grid_0.point_data:
        raise KeyError(
            f"Champ '{sim_function_name}' introuvable dans SIM point_data. "
            f"Clés disponibles dans SIM : {list(sim_grid_0.point_data.keys())}"
        )

    for t in range(num_steps):
        ref_grid = ref_multiblock[t]
        sim_grid = sim_multiblock[t]
        
        # Filtre sur la zone mesurée par DIC
        if "is_imported" in ref_grid.point_data:
            is_imported = ref_grid.point_data["is_imported"]
            mask_imported = np.isclose(is_imported, 0.1, atol=1e-6)
        else:
            mask_imported = np.ones(ref_grid.n_points, dtype=bool)
            
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
# 3. Fonction objectif de l'optimiseur
# ---------------------------------------------------------------------------
def compute_residuals_generic_DIC_BC(
        domain, V, W, WT, f_ref, ref_multiblock,
        model_name, free_param_names, free_param_values, fixed_params,
        config=None):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu : '{model_name}'. Disponibles : {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Fusion des paramètres optimisés et fixes
    full_params = {**fixed_params, **dict(zip(free_param_names, free_param_values))}
    run_cfg = {**cfg, **full_params}

    # Reconstitution des vecteurs 3D [ux, uy, uz]
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

    ref_grid = ref_multiblock[0]
    is_imported = ref_grid.point_data.get("is_imported", np.ones(len(ref_grid.points)))
    mask_imported = np.isclose(is_imported, 0.1, atol=1e-6)
    if ref_grid.points.shape[1] >= 3:
        mask_z = np.isclose(ref_grid.points[:, 2], 0.0, atol=1e-6)
        n_masked = (mask_imported & mask_z).sum()
    else:
        n_masked = mask_imported.sum()

    active_comps_len = 2 if V.dofmap.bs >= 2 else 1
    expected_u_size = n_masked * active_comps_len * len(ref_multiblock)
    expected_f_size = len(ref_multiblock)
    expected_total_size = expected_u_size + expected_f_size

    try:
        f_sim, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)
        
        vtu_function_name = run_cfg.get("vtu_function_name", "displacement_projected")
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
        print(f"--> [Simulation/Newton Divergence] Exception : {e}. Pénalisation de l'erreur.")
        error = np.ones(expected_total_size) * 1e3
        f_sim = np.zeros_like(f_ref) # Sécurité en cas de divergence

    return error, f_sim


# ---------------------------------------------------------------------------
# 4. Boucle FEMU principale
# ---------------------------------------------------------------------------
def femu_res_generic(
        PVD_FILE,
        FORCE_FILE,
        model_name,
        free_param_names=None,
        fixed_param_overrides=None,
        params0_overrides=None,
        bounds_overrides=None,
        config=None):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu : '{model_name}'. Disponibles : {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]
    all_param_names = list(model_info["params_default"].keys())

    if free_param_names is None:
        free_param_names = all_param_names
    else:
        unknown = set(free_param_names) - set(all_param_names)
        if unknown:
            raise ValueError(f"Paramètres inconnus pour '{model_name}' : {unknown}")

    fixed_param_names = [p for p in all_param_names if p not in free_param_names]

    params_default = {**model_info["params_default"], **(fixed_param_overrides or {})}
    bounds_all = {**model_info["bounds"], **(bounds_overrides or {})}

    fixed_params = {k: params_default[k] for k in fixed_param_names}

    params0_dict = {**params_default, **(params0_overrides or {})}
    params0 = [params0_dict[k] for k in free_param_names]
    bounds_free = [bounds_all[k] for k in free_param_names]

    # Chargement des champs de déplacement DIC
    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    print("Pré-chargement des fichiers VTU de référence en mémoire...")
    ref_multiblock = pv.MultiBlock()
    for f_vtu in vtu_files:
        ref_multiblock.append(pv.read(f_vtu))
    print(f"Chargé {len(ref_multiblock)} pas de temps de référence.")

    # Chargement des données de force expérimentales
    print(f"Chargement du fichier de forces : {FORCE_FILE}")
    f_ref = np.load(FORCE_FILE)
    print(f"Chargé {len(f_ref)} points de force.")

    cfg = {
        "pvd_file_path": PVD_FILE,
        "num_steps": len(vtu_files) - 1,
        "t_start": 0.0,
        "T": 3.0,
        "weight_u": 1.0,  # Pondération relative U
        "weight_f": 1.0,  # Pondération relative Force
        **(config or {})
    }

    n_free = len(free_param_names)
    # On ajoute 2 emplacements réservés : 1 pour les résidus + 1 pour la courbe de force
    n_total_plots = n_free + 2
    n_cols = 4
    n_rows = int(np.ceil(n_total_plots / n_cols))

    plt.ion()
    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    # Graphique 1 (gs[0, 0]) : Convergence des résidus
    ax_err = fig.add_subplot(gs[0, 0])
    
    # Graphique 2 (gs[0, 1]) : Fitting Force / Pas de temps
    ax_force = fig.add_subplot(gs[0, 1])

    # Graphiques 3 à N : Évolution des paramètres
    ax_params = []
    for i in range(n_free):
        slot_idx = i + 2  # Décalage de 2 slots
        row, col = divmod(slot_idx, n_cols)
        ax_params.append(fig.add_subplot(gs[row, col]))

    history_err = []
    history_params = []

    params0_norm = normalize_params(params0, bounds_free)
    bounds_norm = Bounds([0.0] * len(params0), [1.0] * len(params0))

    def objective_function(params_norm):
        params_phys = denormalize_params(params_norm, bounds_free)

        print(f"\n{datetime.now()} | Simu n°{len(history_err)} | Modèle: {model_name}")
        print("Paramètres libres testés :", dict(zip(free_param_names, params_phys)))
        if fixed_params:
            print("Paramètres fixes :", fixed_params)

        # Récupération des résidus ET de la force simulée
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

        try:
            # --- 1. Graphique Erreur globale ---
            ax_err.clear()
            ax_err.plot(history_err, color='firebrick', lw=1.5)
            ax_err.set_yscale('log')
            ax_err.set_title(r"Norme Résidus (Log $\sum r^2$)")
            ax_err.grid(True, which="both", ls="-", alpha=0.2)

            # --- 2. Graphique Fitting Force / Pas de temps ---
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

            # --- 3. Graphiques Paramètres ---
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
        
        # scipy.optimize.least_squares attend uniquement le tableau 1D de résidus
        return residuals

    result_norm = least_squares(
        objective_function,
        params0_norm,
        method='trf',
        bounds=bounds_norm,
        ftol=1e-6, gtol=1e-8, max_nfev=150, verbose=2, x_scale=1.0, diff_step=1e-2,
    )

    plt.ioff()
    plt.show()

    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds_free))
    result_phys.param_names = free_param_names
    result_phys.fixed_params = fixed_params

    return result_phys


# ---------------------------------------------------------------------------
# 5. Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    FORCE_FILE = "forces_sample.npy"

    print("Lancement de l'optimisation FEMU mixte (Champs u + Forces F)...")

    optimizer_result = femu_res_generic(
        PVD_FILE,
        FORCE_FILE,
        model_name="J2IsotropicHardening",
        params0_overrides={
            "sigma_Y": 90.0,
            "Q_var": 45.0,
            "k_hardening": 1_100.0,
            
        },
        free_param_names=[
            "sigma_Y",
            "Q_var",
            "k_hardening",
            "uy_up",
            "uy_down",
            "ux_down",
            "ux_up"
        ],
        fixed_param_overrides={
            "E": 200_000.0,
            "nu": 0.3,
            "uz_up": 0.0,
            "uz_down": 0.0,
        },
        config={
            "weight_u": 1.0,  # Ajustable si vous souhaitez donner plus de poids à la force ou au déplacement
            "weight_f": 5.0,
        }
    )

    print("\n================ OPTIMISATION TERMINÉE ================")
    print("Paramètres identifiés :", dict(zip(optimizer_result.param_names, optimizer_result.x)))