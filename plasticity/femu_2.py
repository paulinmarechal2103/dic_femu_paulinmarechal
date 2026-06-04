import h5py
#from imagecodecs import NONE
import numpy as np


from plasticity_simu import *
from hill48_model import Hill48Model,Hill48state

from scipy.optimize import minimize

from image_calibration import *
from dolfinx import geometry
import dolfinx
import pandas as pd

import os


def compute_u_sim_raw_h5_diff(f, u_sim, base_path = "Function/displacement"):
    """
    Compute the total displacement difference between 
    H5 reference raw file extracted with h5py
    and simulation output array.
    """
    errors = []
    
    step = 0
    while str(step) in f[base_path]:
        d1 = f[f"{base_path}/{step}"][:]
        d2 = u_sim[step]
        
        # FIX: Reshape the flattened 1D array to match (num_nodes, 3)
        d2 = d2.reshape(d1.shape)
        
        # Calcul de la norme de la différence
        #diff = np.linalg.norm(d1 - d2)**2
        diff = d1-d2
        rmse = np.mean(diff**2)
        diff = rmse / (np.mean(d1**2) + 1e-12)


        errors.append(diff)
        
        #print(f"Pas {step} : Différence = {diff}")
        step += 1
            
    return np.sum(errors)



def evaluate_sim_at_dic_points(domain, u_sim, pts_dic_img, tform_cad_to_img_4d):
    """Calcule la position des points DIC dans le repère CAD via l'inverse 
    de la matrice 4x4, et y évalue le champ de déplacement FE (u_sim).

    Args:
        domain: dolfinx.mesh.Mesh du domaine de simulation (maillage CAD)
        u_sim: La fonction dolfinx du champ de déplacement simulé
        pts_dic_img: Tableau (N, 2) des coordonnées (x, y) de la DIC (pixels)
        tform_cad_to_img_4d: Matrice de transformation 4x4 (CAD -> Image)
    Returns:
        u_sim_at_dic: Tableau (N, 2) des déplacements simulés évalués aux positions DIC transformées en CAD
        valid_indices: Indices des points DIC qui ont été trouvés à l'intérieur du maillage CAD et pour lesquels u_sim a été évalué
    """

    # 1. Inversion de la matrice pour passer de l'Image -> CAD
    tform_img_to_cad = np.linalg.inv(tform_cad_to_img_4d)
    
    # 2. Passage des points DIC en coordonnées homogènes 3D (N, 4)
    N = pts_dic_img.shape[0]
    pts_hom = np.ones((N, 4))
    pts_hom[:, 0] = pts_dic_img[:, 0] # x
    pts_hom[:, 1] = pts_dic_img[:, 1] # y
    pts_hom[:, 2] = 0.0               # z = 0 en DIC 2D

    # 3. Transformation des positions vers le repère CAD
    pts_cad = (tform_img_to_cad @ pts_hom.T).T[:, :3]

    # 4. Détection des collisions avec les éléments du maillage EF
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_cad)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_cad)

    valid_pts_cad = []
    cells_for_eval = []
    valid_indices = []

    # Filtrage : on ne garde que les points DIC qui tombent VRAIMENT dans le maillage CAD
    for i, pt in enumerate(pts_cad):
        if len(colliding_cells.links(i)) > 0:
            cells_for_eval.append(colliding_cells.links(i)[0])
            valid_pts_cad.append(pt)
            valid_indices.append(i)

    if len(valid_pts_cad) == 0:
        print("[Attention] Aucun point de la DIC n'a été trouvé à l'intérieur du maillage CAD.")
        return np.array([]), [], tform_img_to_cad

    valid_pts_cad = np.array(valid_pts_cad)

    # 5. Évaluation native de u_sim aux coordonnées CAD valides (rend un tableau Nx2)
    u_sim_at_dic = u_sim.eval(valid_pts_cad, cells_for_eval)

    return u_sim_at_dic, valid_indices, tform_img_to_cad





bounds_ref_hill48 = [
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
    (150_000, 250_000),   # E [MPa]
    (0.25, 0.35),         # nu 
    (10.0, 500.0),        # sigma_Y [MPa]
    (20.0, 400.0),         # Q_var [MPa]
    (10.0, 1500.0),          # k_hardening
]

def compute_J2_dic_error_from_parameters(domain, V, W, WT, dic_csv_path, tform_cad_to_img_4d, params=[200_000.0, 0.3, 100.0, 50.0, 1_000.0]):
    """Calcule l'écart de déplacement uniquement sur les points de la ROI de la DIC

    entre la simulation Hill48 et les mesures expérimentales du CSV.
    """
    # 1. Chargement et filtrage initial des données du CSV DIC actuel
    data = pd.read_csv(dic_csv_path)
    names = [s.replace('"', "").replace(" ", "") for s in data.columns]
    d = {names[i]: data.values[:, i] for i in range(len(names))}

    valid_mask = (d["u"] != 0) & (~np.isnan(d["u"])) & (~np.isnan(d["x"]))
    
    pts_dic_img = np.stack([d["x"][valid_mask], d["y"][valid_mask]], axis=-1)
    u_dic_img = np.stack([d["u"][valid_mask], d["v"][valid_mask], np.zeros_like(d["u"][valid_mask])], axis=-1)

    # 2. Configuration des paramètres pour le solveur EF
    J2_params = dict(
        t_start=0.0, T=3.0, num_steps=50, load_amp=0.01, length=10.0,
        mesh_file="Flat_specimen_refined.msh", output_dir="results_plasticity", file_name="donnes_ref",
        E=params[0], nu=params[1], sigma_Y=params[2], Q_var=params[3], k_hardening=params[4],
    )

    model = J2IsotropicHardening(
        elastic=ElasticModel(J2_params["E"], J2_params["nu"], tdim=domain.topology.dim),
        sigma_Y=J2_params["sigma_Y"], Q_var=J2_params["Q_var"], k=J2_params["k_hardening"]
    )

    try:
        # 3. Résolution Éléments Finis (FEniCSx)
        _, u_sim = run_simulation_V3(domain, V, W, WT, J2_params, model=model, write_output=False)
        
        # 4. Évaluation de la simulation sur les points de la DIC et récupération de la matrice inverse
        u_sim_at_dic, valid_indices, tform_img_to_cad = evaluate_sim_at_dic_points(
            domain, u_sim, pts_dic_img, tform_cad_to_img_4d
        )
        
        if len(u_sim_at_dic) == 0:
            return 1e3 # Pénalité forte si aucun point ne intersecte
        u_sim_at_dic_2d = u_sim_at_dic[:, :2] # On ne garde que U et V de la simulation 3D
        error = np.sum((u_sim_at_dic_2d - u_dic_cad_valid) ** 2)
        # 5. Transformation des VECTEURS déplacement DIC vers le repère CAD
        # On utilise le bloc de rotation/homothétie 3x3 supérieur gauche de la matrice inverse
        R_img_to_cad = tform_img_to_cad[:3, :3]
        u_dic_cad_all = (R_img_to_cad @ u_dic_img.T).T
        
        # Slicing pour ne garder que les points valides et les composantes (u, v) en 2D
        u_dic_cad_valid = u_dic_cad_all[valid_indices, :2]

        # 6. Calcul de l'erreur des moindres carrés brute sur la ROI
        error = np.sum((u_sim_at_dic_2d - u_dic_cad_valid) ** 2)

    except RuntimeError as e:
        print(f"--> [Newton Divergence] Paramètres instables détectés. Pénalisation de l'erreur.")
        error = 1e3 # Valeur de pénalisation pour SciPy
        
    return error


def compute_J2_temporal_dic_error(domain, V, W, WT, dic_csv_files, t_dic, tform_cad_to_img_4d, params):
    """Calcule l'erreur cumulée sur l'espace et le temps entre la simulation

    et les fichiers CSV de la DIC non-alignés temporellement.
    
    Args:
        dic_csv_files: Liste de str, chemins vers les fichiers CSV (".000", ".001", ...)
        t_dic: Array (N_frames,) temps physique associé à chaque CSV
        tform_cad_to_img_4d: Matrice 4x4 de calibration
    """
    # 1. Configuration et exécution de la simulation EF
    J2_params = dict(
        t_start=0.0, T=3.0, num_steps=50, load_amp=0.01, length=10.0,
        mesh_file="Flat_specimen_refined.msh", output_dir="results_plasticity", file_name="donnes_ref",
        E=params[0], nu=params[1], sigma_Y=params[2], Q_var=params[3], k_hardening=params[4],
    )
    
    # Génération du vecteur de temps de la simulation
    t_sim = np.linspace(J2_params["t_start"], J2_params["T"], J2_params["num_steps"] + 1)

    model = J2IsotropicHardening(
        elastic=ElasticModel(J2_params["E"], J2_params["nu"], tdim=domain.topology.dim),
        sigma_Y=J2_params["sigma_Y"], Q_var=J2_params["Q_var"], k=J2_params["k_hardening"]
    )

    try:
        # run_simulation_V3 doit retourner l'historique des déplacements pour chaque pas de temps
        # u_sim_history: liste de fonctions dolfinx [u(t=0), u(t=1), ..., u(t=T)]
        _, u_sim_history = run_simulation_V3(domain, V, W, WT, J2_params, model=model, write_output=False)
    except RuntimeError:
        print(f"--> [Newton Divergence] Paramètres instables.")
        return 1e6
    u_func_pre = dolfinx.fem.Function(V)
    u_func_post = dolfinx.fem.Function(V)
    total_error = 0.0
    
    # Extraction de la matrice de rotation/homothétie pour les vecteurs déplacements
    tform_img_to_cad = np.linalg.inv(tform_cad_to_img_4d)
    R_img_to_cad = tform_img_to_cad[:3, :3]

    # 2. Boucle sur chaque image (CSV) de la DIC
    for csv_path, tau in zip(dic_csv_files, t_dic):
        
        # Ignorer les images hors de l'intervalle de la simulation
        if tau < t_sim[0] or tau > t_sim[-1]:
            continue
            
        # Charger le CSV de la DIC pour ce pas de temps tau
        data = pd.read_csv(csv_path)
        names = [s.replace('"', "").replace(" ", "") for s in data.columns]
        d = {names[i]: data.values[:, i] for i in range(len(names))}
        
        valid_mask = (d["u"] != 0) & (~np.isnan(d["u"])) & (~np.isnan(d["x"]))
        if not np.any(valid_mask):
            continue
            
        pts_dic_img = np.stack([d["x"][valid_mask], d["y"][valid_mask]], axis=-1)
        u_dic_img = np.stack([d["u"][valid_mask], d["v"][valid_mask], np.zeros_like(d["u"][valid_mask])], axis=-1)
        
        # 3. INTERPOLATION TEMPORELLE DU CHAMP EF
        # Trouver les indices des pas EF encadrant le temps DIC 'tau'
        idx_post = np.searchsorted(t_sim, tau)
        idx_pre = idx_post - 1
        
        t_pre = t_sim[idx_pre]
        t_post = t_sim[idx_post]
        
        # Calcul du poids d'interpolation linéaire
        if t_post == t_pre:
            alpha = 0.0
        else:
            alpha = (tau - t_pre) / (t_post - t_pre)
            
        # --- MODIFICATION : Remplir les objets Function avec les tableaux numpy ---
        u_func_pre.x.array[:] = u_sim_history[idx_pre]
        u_func_post.x.array[:] = u_sim_history[idx_post]
        
        # 4. ÉVALUATION SPATIALE AUX POINTS DIC
        # On passe maintenant les objets Function (u_func_pre, u_func_post)
        u_spatial_pre, valid_indices, _ = evaluate_sim_at_dic_points(domain, u_func_pre, pts_dic_img, tform_cad_to_img_4d)
        u_spatial_post, _, _ = evaluate_sim_at_dic_points(domain, u_func_post, pts_dic_img, tform_cad_to_img_4d)
        # --------------------------------------------------------------------------
        
        if len(u_spatial_pre) == 0:
            continue
            
        # Combinaison linéaire temporelle des déplacements simulés interpolés dans l'espace
        u_sim_interp = (1.0 - alpha) * u_spatial_pre + alpha * u_spatial_post
        
        # 5. Préparation de la mesure expérimentale correspondante
        u_dic_cad_all = (R_img_to_cad @ u_dic_img.T).T
        u_dic_cad_valid = u_dic_cad_all[valid_indices, :2]
        
        # Accumulation de l'erreur des moindres carrés pour ce pas de temps
        total_error += np.sum((u_sim_interp - u_dic_cad_valid) ** 2)
        
    return total_error

def normalize_params(params, bounds):
    """Normalize parameters to [0, 1] range based on given bounds."""
    return [(params[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) for i in range(len(bounds))]

def denormalize_params(params_norm, bounds):
    """Denormalize parameters from [0, 1] range back to original scale based on given bounds."""
    return [params_norm[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0] for i in range(len(bounds))]  


from datetime import datetime


def femu2(
        domain,
        V, W, WT,
        ref_img,
        h5_file,
        params0=[200_500.0, 0.29, 102.0, 52.0, 1_010.0],
        bounds=bounds_ref_J2,
    ):
    """
    domain : dolfinx.mesh.Mesh déjà créé à partir du maillage PyVista
    V, W, WT : espaces de fonctions déjà construits pour ce maillage
    h5_file : chemin vers le fichier de référence contenant les déplacements mesurés
    params0 : liste des paramètres initiaux pour l'optimisation (E, nu, sigma_Y, Q_var, k_hardening)
    bounds : liste des tuples (min, max) pour chaque paramètre, utilisée pour la normalisation et les contraintes de l'optimiseur
    """
    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))

    tform_cad_to_img_4d = calibrate_2d(domain, ref_img,min_scale = 0.7, max_scale = 1.3)

    
    # 11 paramètres + 1 erreur = 12 slots (3 lignes x 4 colonnes)
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0]) # Erreur en haut à gauche
    
    # On crée les axes pour les 11 paramètres
    ax_params = []
    for i in range(1, 6):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []
    history_params = [] # On va stocker les paramètres PHYSIQUES (dénormalisés) pour le plot

    # --- Préparation de la Normalisation pour SciPy ---
    # L-BFGS-B va travailler entre 0 et 1 pour chaque paramètre
    params0_norm = normalize_params(params0, bounds)
    bounds_norm = [(0.0, 1.0) for _ in range(len(bounds))]

    with h5py.File(h5_file, 'r') as f:
        def objective_function(params_norm):
            # 1. Dénormalisation pour retrouver les valeurs physiques
            params_phys = denormalize_params(params_norm, bounds)
            
            print(f"{str(datetime.now())} \nCurrent params (phys): {params_phys}\n")
            
            # 2. Calcul de l'erreur avec les valeurs physiques
            error = compute_J2_dic_error_from_parameters(domain,V,W,WT,dic_csv_path,tform_cad_to_img_4d, params_phys)
            
            # 3. Stockage pour l'historique
            history_err.append(error)
            history_params.append(params_phys)
            data_p = np.array(history_params)
            
            # 4. Mise à jour graphique (avec les valeurs physiques)
            try:
                # Plot Erreur
                ax_err.clear()
                ax_err.plot(history_err, color='firebrick', lw=1.5)
                ax_err.set_yscale('log')
                ax_err.set_title("Erreur (Log)")
                ax_err.grid(True, which="both", ls="-", alpha=0.2)

                # Plot Paramètres (Physiques)
                for i in range(len(params_phys)):
                    ax_params[i].clear()
                    ax_params[i].plot(data_p[:, i], color='royalblue')
                    ax_params[i].set_title(f"P{i}: {params_phys[i]:.2e}", fontsize=9)
                    ax_params[i].grid(True, alpha=0.2)
                
                plt.tight_layout()
                plt.pause(0.001)
            except:
                # Permet de continuer si la fenêtre est fermée
                pass
                
            print(f"Error: {error}")
            return error

        # L'optimiseur reçoit les versions normalisées (0 à 1)
        result_norm = minimize(
            objective_function,
            params0_norm,
            method='L-BFGS-B',
            bounds=bounds_norm,
            options={'ftol': 1e-4, 'gtol': 1e-3, 'maxiter': 150, 'disp': True, 'eps': 1e-3}
        )
        
    plt.ioff()
    plt.show()
    
    # --- Post-traitement ---
    # On reconstruit l'objet résultat pour renvoyer les paramètres physiques optimaux
    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds))
    
    return result_phys

def femu2_2(
        domain,
        V, W, WT,
        ref_img,
        dic_csv_files, 
        t_dic,
        params0=[200_500.0, 0.29, 102.0, 52.0, 1_010.0],
        bounds=bounds_ref_J2,
    ):
    """
    domain : dolfinx.mesh.Mesh déjà créé à partir du maillage PyVista
    V, W, WT : espaces de fonctions déjà construits pour ce maillage
    h5_file : chemin vers le fichier de référence contenant les déplacements mesurés
    params0 : liste des paramètres initiaux pour l'optimisation (E, nu, sigma_Y, Q_var, k_hardening)
    bounds : liste des tuples (min, max) pour chaque paramètre, utilisée pour la normalisation et les contraintes de l'optimiseur
    """
    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))

    tform_cad_to_img_4d = calibrate_2d(domain, ref_img, min_scale = 0.7, max_scale = 1.3)

    
    # 11 paramètres + 1 erreur = 12 slots (3 lignes x 4 colonnes)
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0]) # Erreur en haut à gauche
    
    # On crée les axes pour les 11 paramètres
    ax_params = []
    for i in range(1, 6):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []
    history_params = [] # On va stocker les paramètres PHYSIQUES (dénormalisés) pour le plot

    # --- Préparation de la Normalisation pour SciPy ---
    # L-BFGS-B va travailler entre 0 et 1 pour chaque paramètre
    params0_norm = normalize_params(params0, bounds)
    bounds_norm = [(0.0, 1.0) for _ in range(len(bounds))]

    def objective_function(params_norm):
        # 1. Dénormalisation pour retrouver les valeurs physiques
        params_phys = denormalize_params(params_norm, bounds)
        
        print(f"{str(datetime.now())} \nCurrent params (phys): {params_phys}\n")
        
        # 2. Calcul de l'erreur avec les valeurs physiques
        error = compute_J2_temporal_dic_error(domain,V,W,WT,dic_csv_files,t_dic,tform_cad_to_img_4d,params_phys)
        
        # 3. Stockage pour l'historique
        history_err.append(error)
        history_params.append(params_phys)
        data_p = np.array(history_params)
        
        # 4. Mise à jour graphique (avec les valeurs physiques)
        try:
            # Plot Erreur
            ax_err.clear()
            ax_err.plot(history_err, color='firebrick', lw=1.5)
            ax_err.set_yscale('log')
            ax_err.set_title("Erreur (Log)")
            ax_err.grid(True, which="both", ls="-", alpha=0.2)

            # Plot Paramètres (Physiques)
            for i in range(len(params_phys)):
                ax_params[i].clear()
                ax_params[i].plot(data_p[:, i], color='royalblue')
                ax_params[i].set_title(f"P{i}: {params_phys[i]:.2e}", fontsize=9)
                ax_params[i].grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.pause(0.001)
        except:
            # Permet de continuer si la fenêtre est fermée
            pass
            
        print(f"Error: {error}")
        return error

    # L'optimiseur reçoit les versions normalisées (0 à 1)
    result_norm = minimize(
        objective_function,
        params0_norm,
        method='L-BFGS-B',
        bounds=bounds_norm,
        options={'ftol': 1e-4, 'gtol': 1e-3, 'maxiter': 150, 'disp': True, 'eps': 1e-3}
    )
        
    plt.ioff()
    plt.show()
    
    # --- Post-traitement ---
    # On reconstruit l'objet résultat pour renvoyer les paramètres physiques optimaux
    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds))
    
    return result_phys



if __name__ == "__main__":
    domain = load_and_write_mesh("astar_2D.msh")
    ref_img = skimage.io.imread("femu_files/csv_imgs/VK03-1-16-0001_0.tif", as_gray=True)
    
    V, W, WT = build_function_spaces(domain)
    dic_csv_files = [
        "femu_files/csv_imgs/VK03-1-16-0201_0.csv",
    ]

    t_dic = np.array([2.99])

    from random import uniform,seed
    seed(42)  # Pour la reproductibilité
    perturbation_percentage = 0.05  # 5% de perturbation aléatoire
    normalized_result = normalize_params([200_000.0, 0.3, 100.0, 50.0, 1_000.0], bounds_ref_J2)
    normalized_disturbed = [i + uniform(-perturbation_percentage, perturbation_percentage) for i in normalized_result]
    parameters_disturbed = denormalize_params(normalized_disturbed, bounds_ref_J2)
    optimizer_result = femu2_2(domain,V, W, WT,ref_img, dic_csv_files, t_dic, parameters_disturbed, bounds_ref_J2)
    print("Optimized parameters (phys):", optimizer_result.x)


