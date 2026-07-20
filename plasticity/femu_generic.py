"""
FEMU générique : identification de paramètres élastoplastiques par recalage
sur des champs de déplacement DIC (fichiers PVD/VTU).

Ce module suppose que les objets suivants existent déjà dans ton code
(mêmes noms que dans ton script d'origine) :
    - ElasticModel, J2IsotropicHardening
    - DEFAULT_CONFIG
    - get_vtu_files_from_pvd, load_domain_from_vtu, build_function_spaces
    - run_simulation_bc_vtu_fast
    - normalize_params, denormalize_params

Deux nouveautés par rapport à ton code :
    1. MODEL_REGISTRY : un dictionnaire qui décrit chaque modèle
       (comment le construire, ses paramètres par défaut, ses bornes).
       -> pour ajouter un nouveau modèle, il suffit d'ajouter une entrée ici,
       le reste du code n'a pas à changer.
    2. `free_param_names` : la liste des paramètres que tu veux réellement
       identifier. Tout ce qui n'est pas dans cette liste reste fixé à sa
       valeur par défaut (ou à la valeur que tu passes en override).
"""

import numpy as np
from datetime import datetime
from scipy.spatial import KDTree
from scipy.optimize import least_squares, Bounds
import matplotlib.pyplot as plt

from femu import *
from plasticity_simu_DIC_BC import *

# ---------------------------------------------------------------------------
# 1. Registre des modèles disponibles
# ---------------------------------------------------------------------------
# Pour ajouter un nouveau modèle : ajouter une clé ici avec
#   - "builder"        : fonction(run_cfg, domain) -> instance du modèle
#   - "params_default" : dict {nom_param: valeur_par_defaut}
#   - "bounds"          : dict {nom_param: (min, max)}
MODEL_REGISTRY = {
    "J2IsotropicHardening": {
        "builder": lambda run_cfg, domain: J2IsotropicHardening(
            elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
            sigma_Y=run_cfg["sigma_Y"],
            Q_var=run_cfg["Q_var"],
            k=run_cfg["k_hardening"],
        ),
        "params_default": {
            "E": 200_000.0,
            "nu": 0.3,
            "sigma_Y": 100.0,
            "Q_var": 50.0,
            "k_hardening": 1_000.0,
        },
        "bounds": {
            "E": (150_000.0, 250_000.0),
            "nu": (0.2, 0.45),
            "sigma_Y": (20.0, 300.0),
            "Q_var": (0.0, 300.0),
            "k_hardening": (10.0, 5_000.0),
        },
    },

    # Exemple pour ajouter un autre modèle plus tard (à adapter/décommenter
    # une fois que tu as la classe correspondante) :
    #
    # "ElasticOnly": {
    #     "builder": lambda run_cfg, domain: ElasticModel(
    #         run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim
    #     ),
    #     "params_default": {"E": 200_000.0, "nu": 0.3},
    #     "bounds": {"E": (150_000.0, 250_000.0), "nu": (0.2, 0.45)},
    # },
}


# ---------------------------------------------------------------------------
# 2. Résidus géométriques (inchangé par rapport à ton code d'origine)
# ---------------------------------------------------------------------------
def compute_u_residuals_is_imported(ref_multiblock, sim_multiblock,
                                     vtu_function_name="displacement_projected",
                                     mask_name="is_imported",
                                     mask_value=0.1, atol=1e-6, tol_max_dist=1e-5, atol_z=1e-6):
    """
    Calcule les résidus de déplacement (composantes X et Y uniquement) entre le MultiBlock de référence
    et le MultiBlock de simulation, pour les points où is_imported == mask_value ET z == 0.
    """
    ref_grid = ref_multiblock[0]
    sim_grid = sim_multiblock[0]

    if mask_name not in ref_grid.point_data:
        raise KeyError(f"Le champ de masque '{mask_name}' est introuvable dans le VTU de référence.")

    is_imported = ref_grid.point_data[mask_name]
    mask_imported = np.isclose(is_imported, mask_value, atol=atol)

    ref_points = ref_grid.points
    gdim = ref_points.shape[1]

    if gdim >= 3:
        z_coords = ref_points[:, 2]
        mask_z = np.isclose(z_coords, 0.0, atol=atol_z)
        final_mask = mask_imported & mask_z
    else:
        final_mask = mask_imported

    if final_mask.sum() == 0:
        raise ValueError(f"Aucun point avec {mask_name} ≈ {mask_value} ET Z ≈ 0 trouvé.")

    masked_ref_points = ref_points[final_mask]

    sim_points = sim_grid.points
    tree = KDTree(sim_points)
    distances, sim_indices = tree.query(masked_ref_points)

    if np.any(distances > tol_max_dist):
        max_d = np.max(distances)
        raise ValueError(
            f"Erreur d'appariement géométrique : distance max ({max_d:.2e}) > tolérance ({tol_max_dist:.2e}). "
            f"Vérifie que la géométrie du maillage correspond bien."
        )

    errors = []
    num_steps = len(ref_multiblock)

    if len(sim_multiblock) < num_steps:
        raise ValueError(f"Le MultiBlock de simulation a moins de pas ({len(sim_multiblock)}) que la référence ({num_steps}).")

    for step in range(num_steps):
        step_ref = ref_multiblock[step]
        step_sim = sim_multiblock[step]

        if vtu_function_name not in step_ref.point_data:
            raise KeyError(f"Le champ '{vtu_function_name}' est introuvable au pas {step} de la référence.")

        d1 = step_ref.point_data[vtu_function_name][final_mask][:, :2]
        d2 = step_sim.point_data["displacement"][sim_indices][:, :2]

        diff = (d1 - d2).flatten()
        errors.append(diff)

    return np.concatenate(errors)


# ---------------------------------------------------------------------------
# 3. Résidus "physiques" génériques : indépendants du modèle constitutif
# ---------------------------------------------------------------------------
def compute_residuals_generic_DIC_BC(
        domain, V, W, WT, ref_multiblock,
        model_name, free_param_names, free_param_values, fixed_params,
        config=None):
    """
    Version générique de compute_J2_residuals_DIC_BC.
    `model_name` sélectionne le modèle dans MODEL_REGISTRY.
    `free_param_names` / `free_param_values` : paramètres identifiés par l'optimiseur.
    `fixed_params` : le reste des paramètres du modèle, à valeur fixe.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu : '{model_name}'. Disponibles : {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # On recompose l'ensemble complet des paramètres du modèle : fixes + libres
    full_params = {**fixed_params, **dict(zip(free_param_names, free_param_values))}
    run_cfg = {**cfg, **full_params}

    model = model_info["builder"](run_cfg, domain)

    # Pré-calcul de la taille attendue du vecteur résidu (pour pénaliser en cas de plantage)
    ref_grid = ref_multiblock[0]
    is_imported = ref_grid.point_data.get("is_imported", np.ones(len(ref_grid.points)))
    mask_imported = np.isclose(is_imported, 0.1, atol=1e-6)
    if ref_grid.points.shape[1] >= 3:
        mask_z = np.isclose(ref_grid.points[:, 2], 0.0, atol=1e-6)
        n_masked = (mask_imported & mask_z).sum()
    else:
        n_masked = mask_imported.sum()

    active_comps_len = 2 if V.dofmap.bs >= 2 else 1
    expected_size = n_masked * active_comps_len * len(ref_multiblock)

    try:
        _, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)
        vtu_function_name = run_cfg.get("vtu_function_name", "displacement_projected")
        error = compute_u_residuals_is_imported(
            ref_multiblock, sim_multiblock, vtu_function_name=vtu_function_name
        )
    except Exception as e:
        print(e)
        print("--> [Simulation/Newton Divergence] Paramètres instables détectés. Pénalisation de l'erreur.")
        error = np.ones(expected_size) * 1e3

    return error


# ---------------------------------------------------------------------------
# 4. Boucle FEMU générique
# ---------------------------------------------------------------------------
def femu_res_generic(
        PVD_FILE,
        model_name,
        free_param_names=None,       # None -> tous les paramètres du modèle sont identifiés
        fixed_param_overrides=None,  # pour changer la valeur d'un paramètre fixe
        params0_overrides=None,      # valeurs initiales custom pour les paramètres libres
        bounds_overrides=None,       # bornes custom, dict {nom: (min, max)}
        config=None):
    """
    Identifie par FEMU (Finite Element Model Updating) les paramètres d'un
    modèle de comportement, en recalant la simulation DOLFINx sur des champs
    de déplacement expérimentaux (DIC) chargés depuis un fichier PVD/VTU.
 
    Le modèle à utiliser est choisi via `model_name` (voir MODEL_REGISTRY),
    et seul le sous-ensemble de paramètres passé dans `free_param_names` est
    réellement identifié par l'optimiseur (least_squares) ; tous les autres
    paramètres du modèle restent fixés à leur valeur par défaut, ou à la
    valeur fournie dans `fixed_param_overrides`.
 
    Paramètres
    ----------
    PVD_FILE : str
        Chemin vers le fichier .pvd décrivant la série temporelle de VTU
        de référence (champs de déplacement expérimentaux issus de la DIC).
    model_name : str
        Nom du modèle à utiliser, doit être une clé de `MODEL_REGISTRY`
        (ex. "J2IsotropicHardening").
    free_param_names : list[str], optionnel
        Noms des paramètres à identifier (doivent exister dans
        `MODEL_REGISTRY[model_name]["params_default"]`). Si None (défaut),
        tous les paramètres du modèle sont identifiés.
    fixed_param_overrides : dict, optionnel
        Valeurs à utiliser pour les paramètres non identifiés, à la place
        de leur valeur par défaut. Ex. {"nu": 0.33}.
    params0_overrides : dict, optionnel
        Valeurs initiales custom pour un ou plusieurs paramètres libres,
        à la place de leur valeur par défaut. Ex. {"E": 210_000.0}.
    bounds_overrides : dict, optionnel
        Bornes custom (min, max) pour un ou plusieurs paramètres (libres
        ou fixes), à la place de celles définies dans le registre.
        Ex. {"E": (180_000.0, 220_000.0)}.
    config : dict, optionnel
        Options complémentaires fusionnées à la configuration par défaut
        de la simulation (ex. nombre de pas de temps, durée totale T...).
 
    Retour
    ------
    scipy.optimize.OptimizeResult
        Résultat de `scipy.optimize.least_squares`, avec en plus :
        - `x` : valeurs physiques (dénormalisées) des paramètres identifiés,
          dans l'ordre de `free_param_names` ;
        - `param_names` : noms des paramètres identifiés, alignés avec `x` ;
        - `fixed_params` : dict des paramètres non identifiés et leur valeur.
 
    Lève
    ----
    ValueError
        Si `model_name` n'est pas dans MODEL_REGISTRY, ou si
        `free_param_names` contient un nom de paramètre inconnu pour ce
        modèle.
 
    Exemples
    --------
    # Identifier seulement E et sigma_Y du modèle J2, tout le reste fixe
    >>> femu_res_generic("ref.pvd", "J2IsotropicHardening",
    ...                   free_param_names=["E", "sigma_Y"])
 
    # Identifier tous les paramètres du modèle J2 (comportement d'origine)
    >>> femu_res_generic("ref.pvd", "J2IsotropicHardening")
 
    # Identifier E et nu, avec une valeur initiale et une borne différentes pour E
    >>> femu_res_generic("ref.pvd", "J2IsotropicHardening",
    ...                   free_param_names=["E", "nu"],
    ...                   params0_overrides={"E": 210_000.0},
    ...                   bounds_overrides={"E": (180_000.0, 220_000.0)})
    """
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

    # --- 1. Chargement des fichiers VTU de référence ---
    vtu_files = get_vtu_files_from_pvd(PVD_FILE)

    # --- 2. Chargement du domaine DOLFINx ---
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    # --- 3. Pré-chargement de la référence expérimentale ---
    print("Pré-chargement des fichiers VTU de référence en mémoire...")
    import pyvista as pv
    ref_multiblock = pv.MultiBlock()
    for f_vtu in vtu_files:
        ref_multiblock.append(pv.read(f_vtu))
    print(f"Chargé {len(ref_multiblock)} pas de temps de référence.")

    cfg = {
        "pvd_file_path": PVD_FILE,
        "num_steps": len(vtu_files) - 1,
        "t_start": 0.0,
        "T": 3.0,
        **(config or {})
    }

    # --- Configuration du plot interactif (grille dynamique selon nb de params libres) ---
    n_free = len(free_param_names)
    n_cols = 4
    n_rows = int(np.ceil((n_free + 1) / n_cols))

    plt.ion()
    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    ax_err = fig.add_subplot(gs[0, 0])
    ax_params = []
    for i in range(1, n_free + 1):
        row, col = divmod(i, n_cols)
        ax_params.append(fig.add_subplot(gs[row, col]))

    history_err = []
    history_params = []

    # --- Normalisation ---
    params0_norm = normalize_params(params0, bounds_free)
    bounds_norm = Bounds([0.0] * len(params0), [1.0] * len(params0))

    def objective_function(params_norm):
        params_phys = denormalize_params(params_norm, bounds_free)

        print(f"\n{datetime.now()} | Simu n°{len(history_err)} | Modèle: {model_name}")
        print("Paramètres libres testés :", dict(zip(free_param_names, params_phys)))
        if fixed_params:
            print("Paramètres fixes :", fixed_params)

        residuals = compute_residuals_generic_DIC_BC(
            domain, V, W, WT, ref_multiblock,
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
            ax_err.clear()
            ax_err.plot(history_err, color='firebrick', lw=1.5)
            ax_err.set_yscale('log')
            ax_err.set_title(r"Norme Résidus (Log $\sum r^2$)")
            ax_err.grid(True, which="both", ls="-", alpha=0.2)

            for i, name in enumerate(free_param_names):
                ax_params[i].clear()
                ax_params[i].plot(data_p[:, i], color='royalblue')
                ax_params[i].set_title(f"{name}: {params_phys[i]:.2e}", fontsize=9)
                ax_params[i].grid(True, alpha=0.2)

            plt.tight_layout()
            plt.pause(0.001)
        except Exception:
            pass

        print(f"Norme des résidus : {error_scalar:.6e}")
        return residuals

    result_norm = least_squares(
        objective_function,
        params0_norm,
        method='trf',
        bounds=bounds_norm,
        ftol=1e-6, gtol=1e-6, max_nfev=150, verbose=2, x_scale=1.0, diff_step=1e-3,
    )

    plt.ioff()
    plt.show()

    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds_free))
    result_phys.param_names = free_param_names
    result_phys.fixed_params = fixed_params

    return result_phys



if __name__ == "__main__":
    from random import uniform, seed
    import numpy as np

    # 2. Passage au fichier PVD (au lieu du XDMF)
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    # PVD_FILE = "results/projection_cad_temporelle_mask.pvd"
    
    real_params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]
    params_names = ["E", "nu", "sigma_Y", "Q_var", "k_hardening"]

    print("Lancement de l'optimisation FEMU via le pipeline PyVista...")

    # 4. Lancement de l'optimisation avec le fichier PVD
    optimizer_result = femu_res_generic(
            PVD_FILE,
            model_name="J2IsotropicHardening",
            free_param_names=["sigma_Y"],
            fixed_param_overrides={"E": 200_000.0, "nu": 0.3,"Q_var": 50.0, "k_hardening": 1_000.0},
        )

    
    # 5. Affichage des résultats et calcul de l'erreur
    print("\n================ OPTIMISATION TERMINÉE ================")
    print("Optimized parameters (phys):", optimizer_result.x)
    print("Normalized error:")
    for i in range(len(real_params)):
        err_percent = abs(optimizer_result.x[i] - real_params[i]) / abs(real_params[i]) * 100
        print(f"  - {params_names[i]} : {err_percent:.5f}%")