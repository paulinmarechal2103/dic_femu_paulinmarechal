import h5py
#from imagecodecs import NONE
import numpy as np


from plasticity_simu_DIC_BC import *
from femu import *
from hill48_model import Hill48Model,Hill48state

from scipy.optimize import minimize,least_squares, Bounds

from image_calibration import *
from dolfinx import geometry
from scipy.spatial import KDTree

from scipy.optimize import least_squares, Bounds
from datetime import datetime
import matplotlib.pyplot as plt



bounds_ref = [
    (150_000, 250_000),   # E [MPa]
    (0.25, 0.35),         # nu 
    (10.0, 500.0),        # sigma_Y [MPa]
    (0.0, 400.0),         # Q_var [MPa]
    (5.0, 50.0),          # k_hardening
    (0.3, 1.3),           # F : Hill, resserré (évite les rapports d'anisotropie > 3)
    (0.3, 1.3),           # G : Hill, resserré
    (0.2, 1.0),           # H : Hill, resserré
    (0.8, 1.8),           # L : cisaillement hors-plan, resserré
    (0.8, 1.8),           # M : cisaillement hors-plan, resserré
    (0.6, 1.6),           # N : cisaillement plan, cohérent avec H et resserré
]

bounds_ref_J2 = [
    (199000, 201_000),   # E [MPa]
    (0.25, 0.35),         # nu 
    (10.0, 500.0),        # sigma_Y [MPa]
    (20.0, 400.0),         # Q_var [MPa]
    (10.0, 1500.0),          # k_hardening
]



def compute_u_residuals_is_imported(ref_multiblock, sim_multiblock, 
                                    vtu_function_name="displacement_projected",
                                    mask_name="is_imported",
                                    mask_value=0.1, atol=1e-6, tol_max_dist=1e-5, atol_z=1e-6):
    """
    Calcule les résidus de déplacement (composantes X et Y uniquement) entre le MultiBlock de référence
    et le MultiBlock de simulation, pour les points où is_imported == mask_value ET z == 0.
    """
    # On utilise le premier pas de temps pour définir la géométrie et le masque
    ref_grid = ref_multiblock[0]
    sim_grid = sim_multiblock[0]
    
    # --- 1. Extraction du masque is_imported ---
    if mask_name not in ref_grid.point_data:
        raise KeyError(f"Le champ de masque '{mask_name}' est introuvable dans le VTU de référence.")
    
    is_imported = ref_grid.point_data[mask_name]
    mask_imported = np.isclose(is_imported, mask_value, atol=atol)
    
    # --- 2. Extraction de la géométrie et filtre Z=0 ---
    ref_points = ref_grid.points
    gdim = ref_points.shape[1]
    
    if gdim >= 3:
        z_coords = ref_points[:, 2]
        mask_z = np.isclose(z_coords, 0.0, atol=atol_z)
        final_mask = mask_imported & mask_z
    else:
        # En 2D, Z=0 est implicite
        final_mask = mask_imported
        
    if final_mask.sum() == 0:
        raise ValueError(f"Aucun point avec {mask_name} ≈ {mask_value} ET Z ≈ 0 trouvé.")
        
    masked_ref_points = ref_points[final_mask]
    
    # --- 3. Appariement géométrique des points (KDTree) ---
    # On associe chaque point masqué du VTU de référence au nœud correspondant de la simulation (DOFs de V)
    sim_points = sim_grid.points
    tree = KDTree(sim_points)
    distances, sim_indices = tree.query(masked_ref_points)
    
    if np.any(distances > tol_max_dist):
        max_d = np.max(distances)
        raise ValueError(
            f"Erreur d'appariement géométrique : distance max ({max_d:.2e}) > tolérance ({tol_max_dist:.2e}). "
            f"Vérifie que la géométrie du maillage correspond bien."
        )
        
    # --- 4. Calcul des résidus sur X et Y pour tous les pas de temps ---
    errors = []
    num_steps = len(ref_multiblock)
    
    if len(sim_multiblock) < num_steps:
        raise ValueError(f"Le MultiBlock de simulation a moins de pas ({len(sim_multiblock)}) que la référence ({num_steps}).")
        
    for step in range(num_steps):
        step_ref = ref_multiblock[step]
        step_sim = sim_multiblock[step]
        
        # Extraction du déplacement de référence (composantes X=0 et Y=1)
        if vtu_function_name not in step_ref.point_data:
            raise KeyError(f"Le champ '{vtu_function_name}' est introuvable au pas {step} de la référence.")
        
        # Shape attendue : (N_points, 3) -> on sélectionne les points masqués et les colonnes X, Y
        d1 = step_ref.point_data[vtu_function_name][final_mask][:, :2]
        
        # Extraction du déplacement simulé (déjà formaté en 3D par notre fonction d'affichage)
        d2 = step_sim.point_data["displacement"][sim_indices][:, :2]
        
        diff = (d1 - d2).flatten()
        errors.append(diff)
        
    return np.concatenate(errors)

def compute_J2_residuals_DIC_BC(domain, V, W, WT, ref_multiblock, config=None, params=[200_000.0, 0.3, 100.0, 50.0, 1_000.0]): 
    """
    Calcule la différence de déplacement total entre le MultiBlock de référence
    et le MultiBlock simulé pour un jeu de paramètres élastoplastiques donné.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    # On met à jour la configuration avec les paramètres de cette itération
    run_cfg = {
        **cfg,
        "E": params[0],
        "nu": params[1],
        "sigma_Y": params[2],
        "Q_var": params[3],
        "k_hardening": params[4],
    }

    model = J2IsotropicHardening(
        elastic=ElasticModel(run_cfg["E"], run_cfg["nu"], tdim=domain.topology.dim),
        sigma_Y=run_cfg["sigma_Y"],
        Q_var=run_cfg["Q_var"],
        k=run_cfg["k_hardening"]
    )

    # Pré-calcul de la taille attendue du vecteur résidu en cas de plantage (pour SciPy least_squares)
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
        # On lance la simulation qui renvoie maintenant (force_vec, sim_multiblock)
        _, sim_multiblock = run_simulation_bc_vtu_fast(domain, V, W, WT, run_cfg, model=model)
        #animer_deformee(multiblock=sim_multiblock)
        # Calcul de l'erreur via l'appariement des deux multiblocks
        vtu_function_name = run_cfg.get("vtu_function_name", "displacement_projected")
        error = compute_u_residuals_is_imported(
            ref_multiblock, 
            sim_multiblock, 
            vtu_function_name=vtu_function_name
        )
    except Exception as e:
        print(e)
        print(f"--> [Simulation/Newton Divergence] Paramètres instables détectés. Pénalisation de l'erreur.")
        # On renvoie un vecteur de pénalité de la taille exacte requise par least_squares
        error = np.ones(expected_size) * 1e3
        
    return error

def femu_res_J2_DIC_BC(
        PVD_FILE,
        params0=[200_500.0, 0.29, 102.0, 52.0, 1_010.0],
        bounds=bounds_ref_J2,
        params_names=["E", "nu", "sigma_Y", "Q_var", "k_hardening"],
        config=None
    ):
    """
    Optimisation par FEMU à partir de fichiers de déplacement expérimentaux PVD/VTU.
    Charge le domaine directement depuis le premier fichier VTU de référence.
    """
    # 1. Extraction de la liste des fichiers VTU
    vtu_files = get_vtu_files_from_pvd(PVD_FILE)
    
    # 2. Chargement du domaine DOLFINx depuis la géométrie du 1er VTU
    domain = load_domain_from_vtu(vtu_files[0])
    V, W, WT = build_function_spaces(domain)

    # 3. Pré-chargement de la référence expérimentale (Gain d'I/O massif !)
    print("Pré-chargement des fichiers VTU de référence en mémoire...")
    import pyvista as pv
    ref_multiblock = pv.MultiBlock()
    for f_vtu in vtu_files:
        ref_multiblock.append(pv.read(f_vtu))
    print(f"Chargé {len(ref_multiblock)} pas de temps de référence.")

    # Configuration par défaut
    cfg = {
        "pvd_file_path": PVD_FILE,
        "num_steps": len(vtu_files) - 1,
        "t_start": 0.0,
        "T": 3.0,
        **(config or {})
    }

    # --- Configuration du Plot Interactif ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0])
    
    ax_params = []
    for i in range(1, 6):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []    
    history_params = [] 

    # --- Préparation de la Normalisation ---
    params0_norm = normalize_params(params0, bounds)
    bounds_norm = Bounds([0.0]*len(params0), [1.0]*len(params0))

    def objective_function(params_norm):
        params_phys = denormalize_params(params_norm, bounds)
        
        print(f"\n{str(datetime.now())} | Simu n°{len(history_err)}")
        print(f"Paramètres testés : {params_phys}")
        
        # Calcul du vecteur résidu
        residuals = compute_J2_residuals_DIC_BC(
            domain, V, W, WT, ref_multiblock, config=cfg, params=params_phys
        )
        
        # Norme pour l'affichage graphique
        error_scalar = np.sum(np.square(residuals))
        
        history_err.append(error_scalar)
        history_params.append(params_phys)
        data_p = np.array(history_params)
        
        # Rafraîchissement des courbes
        try:
            ax_err.clear()
            ax_err.plot(history_err, color='firebrick', lw=1.5)
            ax_err.set_yscale('log')
            ax_err.set_title("Norme Résidus (Log $\sum r^2$)")
            ax_err.grid(True, which="both", ls="-", alpha=0.2)

            for i in range(len(params_phys)):
                ax_params[i].clear()
                ax_params[i].plot(data_p[:, i], color='royalblue')
                ax_params[i].set_title(f"{params_names[i]}: {params_phys[i]:.2e}", fontsize=9)
                ax_params[i].grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.pause(0.001)
        except Exception:
            pass
                
        print(f"Norme des résidus : {error_scalar:.6e}")
        return residuals

    # Optimisation par méthode Trust Region Reflective (TRF)
    result_norm = least_squares(
        objective_function,
        params0_norm,
        method='trf',
        bounds=bounds_norm,
        ftol=1e-6, gtol=1e-6, max_nfev=150, verbose=2, x_scale=1.0, diff_step=1e-2,
    )
    
    plt.ioff()
    plt.show()
    
    # Dénormalisation du résultat final
    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds))
    
    return result_phys



if __name__ == "__main__":
    print("hey")
    from random import uniform, seed
    import numpy as np

    # 1. Définition des bornes (inchangées)
    bounds_ref_J2_centr= [
            (200000, 200000+1e-6),   # E [MPa]
            (0.3, 0.3+1e-10),         # nu 
            (10.0, 500.0),        # sigma_Y [MPa]
            (5.0, 400.0),         # Q_var [MPa]
            (10.0, 1500.0),          # k_hardening
        ]

    # 2. Passage au fichier PVD (au lieu du XDMF)
    PVD_FILE = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd"
    # PVD_FILE = "results/projection_cad_temporelle_mask.pvd"
    
    real_params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]
    params_names = ["E", "nu", "sigma_Y", "Q_var", "k_hardening"]
    
    # 3. Génération de la perturbation (inchangée)
    seed(43)  # Pour la reproductibilité
    perturbation_percentage = 0.1  # 15% de perturbation aléatoire
    
    normalized_result = normalize_params(real_params, bounds_ref_J2_centr)
    normalized_disturbed = [i + uniform(-perturbation_percentage, perturbation_percentage) for i in normalized_result]
    normalized_disturbed = [min(max(i, 0.0), 1.0) for i in normalized_disturbed]  # Clamp entre 0 et 1
    parameters_disturbed = denormalize_params(normalized_disturbed, bounds_ref_J2_centr)
    
    print("Paramètres initiaux perturbés (physiques) :", parameters_disturbed)
    print("Lancement de l'optimisation FEMU via le pipeline PyVista...")

    # 4. Lancement de l'optimisation avec le fichier PVD
    optimizer_result = femu_res_J2_DIC_BC(
        PVD_FILE, 
        bounds=bounds_ref_J2_centr, 
        params0=real_params, #parameters_disturbed,
        params_names=params_names
    )
    
    # 5. Affichage des résultats et calcul de l'erreur
    print("\n================ OPTIMISATION TERMINÉE ================")
    print("Optimized parameters (phys):", optimizer_result.x)
    print("Normalized error:")
    for i in range(len(real_params)):
        err_percent = abs(optimizer_result.x[i] - real_params[i]) / abs(real_params[i]) * 100
        print(f"  - {params_names[i]} : {err_percent:.5f}%")