import h5py
#from imagecodecs import NONE
import numpy as np


from plasticity_simu import *
from hill48_model import Hill48Model,Hill48state

from scipy.optimize import minimize

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

def compute_direct_h5_diff(h5_file1, h5_file2):
    """
    """
    errors = []
    
    with h5py.File(h5_file1, 'r') as f1, h5py.File(h5_file2, 'r') as f2:
        # Dans votre fichier, le chemin est : /Function/displacement/0, 1, 2...
        base_path = "Function/displacement"
        
        step = 0
        while str(step) in f1[base_path]:
            # Lecture des arrays numpy directs
            d1 = f1[f"{base_path}/{step}"][:]
            d2 = f2[f"{base_path}/{step}"][:]
            
            # Calcul de la norme de la différence
            diff = np.linalg.norm(d1 - d2)
            errors.append(diff)
            
            print(f"Pas {step} : Différence = {diff}")
            step += 1
            
    return np.sum(errors)

# # Utilisation (remplacez par vos noms de fichiers .h5)
# diffs = compute_direct_h5_diff("femu_files/res.h5", "femu_files/donnes_ref.h5")
# print(f"Différence totale : {diffs}")

#_, u = run_simulation()


def compute_u_sim_h5_diff(h5_file, u_sim, base_path = "Function/displacement"):
    """Compute the total displacement difference between H5 reference and simulation output."""
    errors = []
    with h5py.File(h5_file, 'r') as f:
        
        
        step = 0
        while str(step) in f[base_path]:
            d1 = f[f"{base_path}/{step}"][:]
            d2 = u_sim[step]
            
            # FIX: Reshape the flattened 1D array to match (num_nodes, 3)
            d2 = d2.reshape(d1.shape)
            
            # Calcul de la norme de la différence
            diff = np.linalg.norm(d1 - d2)
            errors.append(diff)
            
            print(f"Pas {step} : Différence = {diff}")
            step += 1
            
    return np.sum(errors)


#diffs = compute_u_sim_h5_diff("femu_files/donnes_ref.h5", u)
#print(f"Différence totale : {diffs}")

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
        diff = np.linalg.norm(d1 - d2)
        errors.append(diff)
        
        #print(f"Pas {step} : Différence = {diff}")
        step += 1
            
    return np.sum(errors)


# with h5py.File("femu_files/donnes_ref.h5", 'r') as f:
#     diffs = compute_u_sim_raw_h5_diff(f, u)
#     print(f"Différence totale : {diffs}")



def compute_hill_raw_h5_error_from_parameters(f, params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0, 0.900, 0.600, 0.400, 1.7, 1.3, 1.350]): 
    """
    Compute the total displacement difference between 
    H5 reference raw file extracted with h5py
    and simulation output array for a given set of Hill48 parameters.

    params should be a list or array containing the following parameters in order:
    (E, nu, sigma_Y, Q_var, k_hardening, F, G, H, L, M, N)
    
    """
    hill_params = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name    = "donnes_ref",
    # Elastic constants (used when no model is supplied)
    E           = params[0],
    nu          = params[1],
    # J2 isotropic hardening parameters (used when no model is supplied)
    sigma_Y     = params[2],
    Q_var       = params[3],
    k_hardening = params[4],
    F = params[5],  # Anisotropie dans le plan transverse
    G = params[6],  # Anisotropie dans le plan longitudinal
    H = params[7],  # Terme d'interaction (souvent proche de 0.5)
    L = params[8],  # Cisaillement hors-plan (souvent supposé isotrope = 1.5)
    M = params[9],  # Cisaillement hors-plan (souvent supposé isotrope = 1.5)
    N = params[10] 
    )

    modèle_hill48 = Hill48Model(
        elastic=ElasticModel(hill_params["E"], hill_params["nu"], tdim=3),
        sigma_Y=hill_params["sigma_Y"],
        H=hill_params["H"],
        F=hill_params["F"],
        G=hill_params["G"],
        L=hill_params["L"],
        M=hill_params["M"],
        N=hill_params["N"],
        Q_var=hill_params["Q_var"],
        k_hardening=hill_params["k_hardening"]
    )
    _, u_sim = run_simulation_V2(hill_params, model=modèle_hill48, write_output=False)
    error = compute_u_sim_raw_h5_diff(f, u_sim)
    return error

def get_hill48_from_yield_ratios(sigma_Y, R22=1.0, R33=1.0, R12=1/np.sqrt(3), R23=1/np.sqrt(3), R31=1/np.sqrt(3)):
    """
    Calcule les coefficients F, G, H, L, M, N de Hill48 à partir des rapports 
    de contraintes d'écoulement. Garantit la convexité physique.
    
    sigma_Y : Limite d'élasticité de référence (direction 1, généralement optimisée)
    R22     : sigma_22^y / sigma_Y
    R33     : sigma_33^y / sigma_Y
    R12     : tau_12^y / sigma_Y
    R23     : tau_23^y / sigma_Y
    R31     : tau_31^y / sigma_Y
    """
    s1 = sigma_Y
    s2 = sigma_Y * R22
    s3 = sigma_Y * R33
    t12 = sigma_Y * R12
    t23 = sigma_Y * R23
    t31 = sigma_Y * R31
    
    # Formules brutes (unité 1/MPa²)
    F_raw = 0.5 * (1/s2**2 + 1/s3**2 - 1/s1**2)
    G_raw = 0.5 * (1/s3**2 + 1/s1**2 - 1/s2**2)
    H_raw = 0.5 * (1/s1**2 + 1/s2**2 - 1/s3**2)
    L_raw = 0.5 / t23**2
    M_raw = 0.5 / t31**2
    N_raw = 0.5 / t12**2
    
    # ⚠️ Adimensionnalisation standard (multiplication par sigma_Y²)
    # La plupart des implémentations FEniCSx/UMAT attendent des coefficients sans unité.
    # Si votre Hill48Model attend les valeurs brutes, retirez simplement "* sY2".
    sY2 = sigma_Y**2
    return F_raw*sY2, G_raw*sY2, H_raw*sY2, L_raw*sY2, M_raw*sY2, N_raw*sY2

def compute_hill_raw_h5_error_from_parameters_yield_ratios(f, params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0, 1.0, 1.0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]): 
    """
    Compute the total displacement difference between 
    H5 reference raw file extracted with h5py
    and simulation output array for a given set of Hill48 parameters derived from yield ratios.

    params should be a list or array containing the following parameters in order:
    (E, nu, sigma_Y, Q_var, k_hardening, F, G, H, L, M, N)
    
    """
    hill_from_yield_ratios = get_hill48_from_yield_ratios(
        sigma_Y=params[2],
        R22=params[5],
        R33=params[6],
        R12=params[7],
        R23=params[8],
        R31=params[9]
    )

    hill_params = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name    = "donnes_ref",
    # Elastic constants (used when no model is supplied)
    E           = params[0],
    nu          = params[1],
    # J2 isotropic hardening parameters (used when no model is supplied)
    sigma_Y     = params[2],
    Q_var       = params[3],
    k_hardening = params[4],
    F = hill_from_yield_ratios[0],  # Anisotropie dans le plan transverse
    G = hill_from_yield_ratios[1],  # Anisotropie dans le plan longitudinal
    H = hill_from_yield_ratios[2],  # Terme d'interaction (souvent proche de 0.5)
    L = hill_from_yield_ratios[3],  # Cisaillement hors-plan (souvent supposé isotrope = 1.5)
    M = hill_from_yield_ratios[4],  # Cisaillement hors-plan (souvent supposé isotrope = 1.5)
    N = hill_from_yield_ratios[5]  # Cisaillement plan (souvent supposé isotrope = 1.5)
    )

    modèle_hill48 = Hill48Model(
        elastic=ElasticModel(hill_params["E"], hill_params["nu"], tdim=3),
        sigma_Y=hill_params["sigma_Y"],
        H=hill_params["H"],
        F=hill_params["F"],
        G=hill_params["G"],
        L=hill_params["L"],
        M=hill_params["M"],
        N=hill_params["N"],
        Q_var=hill_params["Q_var"],
        k_hardening=hill_params["k_hardening"]
    )
    _, u_sim = run_simulation_V2(hill_params, model=modèle_hill48, write_output=False)
    error = compute_u_sim_raw_h5_diff(f, u_sim)
    return error


# with h5py.File("femu_files/res.h5", 'r') as f:
#     diffs = compute_hill_raw_h5_error_from_parameters(f)
#     print(f"Différence totale : {diffs}")




def femu_V1(h5_file, params0 = [200_000.0, 0.3, 100.0, 50.0, 1_000.0, 0.900, 0.600, 0.400, 1.7, 1.3, 1.350]):
    with h5py.File(h5_file, 'r') as f:
        def objective_function(params):
            print(params)
            return compute_hill_raw_h5_error_from_parameters(f, params)
        result = minimize(objective_function, params0, method='BFGS')
        print("Optimized parameters:", result.x)
        print("Minimum error:", result.fun)
    return result

bounds_ref = [
    (150_000, 250_000),   # E [MPa] : acier, OK
    (0.25, 0.35),         # nu : métaux typiques
    (10.0, 500.0),        # sigma_Y [MPa] : large mais valide
    (0.0, 750.0),         # Q_var [MPa] : découplé, à contraindre séparément si besoin
    (5.0, 50),          # k_hardening : CORRECTION CRITIQUE
    (0.3, 1.2),           # F : Hill, élargi légèrement
    (0.3, 1.2),           # G : Hill
    (0.4, 1.0),           # H : Hill, restreint pour éviter R45 extrême
    (1.0, 2.5),           # L : cisaillement hors-plan
    (0.8, 2.0),           # M : cisaillement hors-plan
    (0.6, 1.8),           # N : cisaillement plan, cohérent avec H
]


def femu_V2(
        h5_file,
        params0=[200_000.0, 0.3, 100.0, 50.0, 1_000.0, 0.900, 0.600, 0.400, 1.7, 1.3, 1.350],
        bounds = bounds_ref
    ):
    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    
    # 11 paramètres + 1 erreur = 12 slots (3 lignes x 4 colonnes)
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0]) # Erreur en haut à gauche
    
    # On crée les axes pour les 11 paramètres
    ax_params = []
    for i in range(1, 12):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []
    history_params = []

    with h5py.File(h5_file, 'r') as f:
        def objective_function(params):

            print(f"Current params: {params}")
            error = compute_hill_raw_h5_error_from_parameters(f, params)
            
            history_err.append(error)
            history_params.append(params)
            data_p = np.array(history_params)
            
            # Mise à jour graphique
            try:
                # Plot Erreur
                ax_err.clear()
                ax_err.plot(history_err, color='firebrick', lw=1.5)
                ax_err.set_yscale('log')
                ax_err.set_title("Erreur (Log)")
                ax_err.grid(True, which="both", ls="-", alpha=0.2)

                # Plot Paramètres
                for i in range(len(params)):
                    ax_params[i].clear()
                    ax_params[i].plot(data_p[:, i], color='royalblue')
                    ax_params[i].set_title(f"P{i}: {params[i]:.2e}", fontsize=9)
                    ax_params[i].grid(True, alpha=0.2)
                
                plt.tight_layout()
                plt.pause(0.001)
            except:
                # Permet de continuer si la fenêtre est fermée
                pass
                
            print(f"Error: {error}")
            return error

        result = minimize(
            objective_function,
            params0,
            method='L-BFGS-B',
            bounds=bounds_ref,
            options={'ftol': 1e-5, 'maxiter': 150, 'disp': True, 'eps': 1e-3}
        )
        
    plt.ioff()
    plt.show()
    return result




def femu_V3(
        h5_file,
        params0=[200_002, 0.29, 102, 52, 8, 0.52, 0.52, 0.48, 1.52, 1.48, 1.45],
        bounds = bounds_ref
    ):

    dimensions = [
        Real(bounds_ref[i][0], bounds_ref[i][1], name=f"p_{i}") 
        for i in range(len(bounds_ref))
    ]

    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    
    # 11 paramètres + 1 erreur = 12 slots (3 lignes x 4 colonnes)
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0]) # Erreur en haut à gauche
    
    # On crée les axes pour les 11 paramètres
    ax_params = []
    for i in range(1, 12):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []
    history_params = []

    with h5py.File(h5_file, 'r') as f:
        def objective_function(params):

            print(f"Current params: {params}")
            error = compute_hill_raw_h5_error_from_parameters(f, params)
            
            history_err.append(error)
            history_params.append(params)
            data_p = np.array(history_params)
            
            # Mise à jour graphique
            try:
                # Plot Erreur
                ax_err.clear()
                ax_err.plot(history_err, color='firebrick', lw=1.5)
                ax_err.set_yscale('log')
                ax_err.set_title("Erreur (Log)")
                ax_err.grid(True, which="both", ls="-", alpha=0.2)

                # Plot Paramètres
                for i in range(len(params)):
                    ax_params[i].clear()
                    ax_params[i].plot(data_p[:, i], color='royalblue')
                    ax_params[i].set_title(f"P{i}: {params[i]:.2e}", fontsize=9)
                    ax_params[i].grid(True, alpha=0.2)
                
                plt.tight_layout()
                plt.pause(0.001)
            except:
                # Permet de continuer si la fenêtre est fermée
                pass
                
            print(f"Error: {error}")
            return error

        result = gp_minimize(
            objective_function,          # La fonction à minimiser
            dimensions,                  # L'espace des paramètres (les bornes)
            n_calls=50,                  # Nombre TOTAL d'évaluations (simulations) max
            n_random_starts=10,          # Nombre de points initiaux aléatoires (pour démarrer le modèle)
            acq_func="EI",               # Fonction d'acquisition : Expected Improvement
            x0=params0,                  # On peut optionnellement lui donner votre point initial
            random_state=42,             # Pour la reproductibilité
            verbose=True
        )
        
    plt.ioff()
    plt.show()
    return result



def normalize_params(params, bounds):
    """Normalize parameters to [0, 1] range based on given bounds."""
    return [(params[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) for i in range(len(bounds))]

def denormalize_params(params_norm, bounds):
    """Denormalize parameters from [0, 1] range back to original scale based on given bounds."""
    return [params_norm[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0] for i in range(len(bounds))]  


def femu_V4(
        h5_file,
        params0=[200_002.0, 0.29, 102.0, 52.0, 8.0, 0.52, 0.52, 0.48, 1.52, 1.48, 1.45],
        bounds=bounds_ref
    ):

    # - Normalisation des paramètres pour une meilleure convergence -
    bounds_norm = [(0.0, 1.0) for _ in bounds]
    
    params0_norm = normalize_params(params0, bounds)
    
    dimensions = [
        Real(bounds_norm[i][0], bounds_norm[i][1], name=f"p_{i}") 
        for i in range(len(bounds_norm))
    ]

    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0])
    
    ax_params = []
    for i in range(1, 12):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []
    history_params = []

    with h5py.File(h5_file, 'r') as f:
        
        # Recommandé pour s'assurer du format des données envoyées par skopt
        @use_named_args(dimensions)
        def objective_function(**kwargs):
            # Extraction propre des variables de la dimension skopt
            params_norm = [kwargs[f"p_{i}"] for i in range(len(bounds))]
            # Dénormalisation pour obtenir les paramètres dans leur échelle réelle
            params = denormalize_params(params_norm, bounds)

            print(f"\nCurrent params: {[round(p, 2) for p in params]}")
            
            # --- STRATÉGIE DE PÉNALISATION (Try / Except) ---
            try:
                error = compute_hill_raw_h5_error_from_parameters(f, params)
                print(f"Error: {error}")
            
            except RuntimeError as e:
                # Intercepte spécifiquement l'échec du solveur de Newton (PETSc / dolfinx)
                print(f"--> [DIVERGENCE FEniCSx] Le solveur de Newton n'a pas convergé.")
                print(f"--> Attribution d'une pénalité de 500.0")
                return 500.0  # Renvoyer une valeur élevée pour forcer skopt à fuir cette zone
            
            except Exception as e:
                # Intercepte d'autres exceptions numériques potentielles
                print(f"--> [ERREUR INATTENDUE] : {e}")
                return 500.0

            # --- Enregistrement de l'historique et tracé si la simulation réussit ---
            history_err.append(error)
            history_params.append(params)
            data_p = np.array(history_params)
            
            try:
                # Plot Erreur
                ax_err.clear()
                ax_err.plot(history_err, color='firebrick', lw=1.5)
                ax_err.set_yscale('log')
                ax_err.set_title("Erreur (Log)")
                ax_err.grid(True, which="both", ls="-", alpha=0.2)

                # Plot Paramètres
                for i in range(len(params)):
                    ax_params[i].clear()
                    ax_params[i].plot(data_p[:, i], color='royalblue')
                    ax_params[i].set_title(f"P{i}: {params[i]:.2e}", fontsize=9)
                    ax_params[i].grid(True, alpha=0.2)
                
                plt.tight_layout()
                plt.pause(0.001)
            except:
                pass
                
            return error

        # result = gp_minimize(
        #     objective_function,          
        #     dimensions,                  
        #     n_calls=50,                  
        #     n_random_starts=10,          
        #     acq_func="EI",               
        #     x0=params0_norm,                  
        #     random_state=42,             
        #     verbose=True
        # )
        result = gp_minimize(
            objective_function,          
            dimensions,                  
            n_calls=80,                  # Augmenté pour laisser l'algorithme exploiter après l'initialisation
            n_random_starts=25,          # Recommandé pour 11 variables (environ 2d)
            initial_point_generator="lhs", # <-- CRITIQUE : Force un échantillonnage spatial optimal (Hypercube Latin)
            acq_func="EI",               
            x0=params0_norm,                  
            random_state=42,             
            verbose=True
        )


    plt.ioff()
    plt.show()
    return denormalize_params(result.x, bounds)


from skopt import Optimizer  # <-- Nouvel import requis à la place de gp_minimize

def femu_V5(
        h5_file,
        params0=[200_002.0, 0.29, 102.0, 52.0, 8.0, 0.52, 0.52, 0.48, 1.52, 1.48, 1.45],
        bounds=bounds_ref
    ):
    """
    pour garantir que gp_minimize dispose d'un jeu complet de N simulations réussies 
    pour entraîner correctement son processus gaussien, plutôt que de polluer son historique avec des pénalités à 500.0.
    Pour obtenir ce comportement sans perturber le fonctionnement interne de scikit-optimize, 
    la meilleure méthode consiste à ne pas utiliser directement la boucle automatique de gp_minimize, 
    mais à utiliser la classe Optimizer de skopt. 
    Elle permet de piloter l'optimisation manuellement avec une boucle while selon le schéma "Demander un point -> Évaluer -> 
    Communiquer le résultat" (Ask and Tell).
    De cette manière, si une simulation échoue, on ignore simplement le point (ou on n'incrémente pas le compteur) 
    et on redemande un autre point à l'optimiseur.

    """

    dimensions = [
        Real(bounds[i][0], bounds[i][1], name=f"p_{i}") 
        for i in range(len(bounds))
    ]

    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0])
    
    ax_params = []
    for i in range(1, 12):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []
    history_params = []

    # 1. Initialisation explicite de l'optimiseur skopt
    opt = Optimizer(
        dimensions=dimensions,
        n_random_starts=25,            # Vos itérations initiales LHS
        initial_point_generator="lhs",
        acq_func="EI",
        random_state=42
    )

    # Paramétrage de vos objectifs de convergence
    n_successful_calls_target = 60    # On veut par exemple 60 VRAIES simulations réussies au total
    successful_calls = 0
    iteration_total = 0

    with h5py.File(h5_file, 'r') as f:
        
        # On injecte manuellement le point initial params0 au premier coup s'il est valide
        if params0 is not None:
            next_x = params0
            params0 = None # Pour ne le faire qu'une seule fois
        else:
            next_x = opt.ask()

        # Boucle tant que le quota de simulations réussies n'est pas atteint
        while successful_calls < n_successful_calls_target:
            iteration_total += 1
            print(f"\n--- Itération globale n°{iteration_total} (Succès cumulés: {successful_calls}/{n_successful_calls_target}) ---")
            print(f"Paramètres testés : {[round(p, 2) for p in next_x]}")
            
            try:
                # 2. Exécution de FEniCSx
                error = compute_hill_raw_h5_error_from_parameters(f, next_x)
                print(f"-> Succès ! Error calculée : {error}")
                
                # Envoi du résultat valide à l'optimiseur
                opt.tell(next_x, error)
                
                # Mise à jour des compteurs et historique graphique
                successful_calls += 1
                history_err.append(error)
                history_params.append(next_x)
                
                # Génération du point suivant pour l'itération d'après
                if successful_calls < n_successful_calls_target:
                    next_x = opt.ask()
                
                # Rafraîchissement des graphiques
                try:
                    data_p = np.array(history_params)
                    ax_err.clear()
                    ax_err.plot(history_err, color='firebrick', lw=1.5)
                    ax_err.set_yscale('log')
                    ax_err.set_title("Erreur (Log)")
                    ax_err.grid(True, which="both", ls="-", alpha=0.2)

                    for i in range(len(next_x)):
                        ax_params[i].clear()
                        ax_params[i].plot(data_p[:, i], color='royalblue')
                        ax_params[i].set_title(f"P{i}: {next_x[i]:.2e}", fontsize=9)
                        ax_params[i].grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    plt.pause(0.001)
                except:
                    pass

            except RuntimeError as e:
                # 3. GESTION DE LA DIVERGENCE (Newton max iterations, etc.)
                print(f"--> [DIVERGENCE FEniCSx] Le solveur n'a pas convergé.")
                print(f"--> Rejet de ce set. Demande d'un nouveau point sans incrémenter le compteur de succès.")
                
                # STRATÉGIE : On assigne une très mauvaise valeur (pénalité) à ce point précis 
                # pour que l'optimiseur apprenne instantanément que c'est une zone interdite...
                opt.tell(next_x, 500.0)
                
                # ...MAIS on ne compte pas cette itération comme un succès et on redemande immédiatement un point !
                next_x = opt.ask()
                
            except Exception as e:
                print(f"--> [ERREUR INATTENDUE] : {e}")
                opt.tell(next_x, 500.0)
                next_x = opt.ask()

    plt.ioff()
    plt.show()
    
    # Récupération du meilleur résultat final à partir de l'objet Optimizer
    best_index = np.argmin(opt.yi)
    print("\n--- OPTIMISATION TERMINÉE ---")
    print(f"Nombre total d'essais (Valides + Échecs) : {iteration_total}")
    print(f"Nombre de simulations réussies enregistrées : {successful_calls}")
    print("Meilleurs paramètres trouvés :", opt.Xi[best_index])
    print("Plus petite erreur obtenue :", opt.yi[best_index])
    
    return opt


from skopt.sampler import Lhs

def femu_V6(
        h5_file,
        params0=[200_500.0, 0.29, 105.0, 52.0, 8.0, 0.52, 0.52, 0.48, 1.52, 1.48, 1.45],
        bounds=bounds_ref,
        n_lhs_target=25,
        n_successful_calls_target=60,
    ):
    """
    Par défaut, si l'échantillonnage par Hypercube Latin (LHS) génère 25 points et que 5 d'entre eux font planter FEniCSx, 
    gp_minimize considère que sa phase d'apprentissage est terminée au bout de 25 itérations, 
    même s'il n'a que 20 vraies simulations réussies en mémoire. Le modèle de substitution démarre 
    alors sa phase d'exploitation avec un sérieux déficit d'apprentissage.
    Pour garantir que vous obtenez exactement 25 points initiaux LHS réussis, il faut légèrement adapter la boucle Ask & Tell.
    scikit-optimize ne permet pas de régénérer un point LHS à la demande à l'intérieur de l'objet Optimizer.
    L'astuce consiste à générer l'intégralité de vos points LHS à l'avance, puis à piocher dedans et à en recréer de nouveaux
    si certains échouent.
    """

    dimensions = [
        Real(bounds[i][0], bounds[i][1], name=f"p_{i}") 
        for i in range(len(bounds))
    ]

    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0])
    ax_params = [fig.add_subplot(gs[divmod(i, 4)[0], divmod(i, 4)[1]]) for i in range(1, 12)]
    
    history_err = []
    history_params = []

    # 1. Configuration de l'Optimizer (on désactive son n_random_starts interne car on gère à la main)
    opt = Optimizer(
        dimensions=dimensions,
        n_random_starts=0,  # Gestion manuelle
        acq_func="EI",
        random_state=42
    )

    # Objectifs de la simulation
    
    successful_calls = 0
    iteration_total = 0

    # 2. Génération manuelle de la réserve de points LHS
    # On en génère un peu plus (ex: 50) au cas où FEniCSx divergerait sur certains points
    lhs_sampler = Lhs(criterion="maximin")
    lhs_points = lhs_sampler.generate(dimensions, n_samples=50, random_state=42)
    lhs_index = 0

    with h5py.File(h5_file, 'r') as f:
        
        # Détermination du tout premier point à tester
        if params0 is not None:
            next_x = params0
            params0 = None
            using_lhs = False
        else:
            next_x = lhs_points[lhs_index]
            lhs_index += 1
            using_lhs = True

        # Boucle principale basée uniquement sur les VRAIS succès
        while successful_calls < n_successful_calls_target:
            iteration_total += 1
            
            # Message d'affichage pour suivre les deux phases distinctes
            if successful_calls < n_lhs_target:
                phase_str = f"Phase APPRENTISSAGE LHS (Succès: {successful_calls}/{n_lhs_target})"
            else:
                phase_str = f"Phase EXPLOITATION BAYÉSIENNE (Succès: {successful_calls}/{n_successful_calls_target})"
                using_lhs = False
                
            print(f"\n--- Itération globale n°{iteration_total} | {phase_str} ---")
            print(f"Paramètres testés : {[round(p, 2) for p in next_x]}")
            
            try:
                # 3. Exécution FEniCSx
                error = compute_hill_raw_h5_error_from_parameters(f, next_x)
                print(f"-> Succès ! Error calculée : {error}")
                
                # Envoi du succès à l'optimiseur pour qu'il mette à jour son processus gaussien
                opt.tell(next_x, error)
                
                successful_calls += 1
                history_err.append(error)
                history_params.append(next_x)
                
                # Choix du prochain point selon la phase actuelle
                if successful_calls < n_successful_calls_target:
                    if successful_calls < n_lhs_target:
                        # On continue de piocher dans notre réserve LHS
                        next_x = lhs_points[lhs_index]
                        lhs_index += 1
                    else:
                        # L'apprentissage est validé, on laisse l'optimiseur décider (Exploitation)
                        next_x = opt.ask()
                
                # Rafraîchissement graphique
                try:
                    data_p = np.array(history_params)
                    ax_err.clear()
                    ax_err.plot(history_err, color='firebrick', lw=1.5)
                    ax_err.set_yscale('log')
                    ax_err.set_title("Erreur (Log)")

                    for i in range(len(next_x)):
                        ax_params[i].clear()
                        ax_params[i].plot(data_p[:, i], color='royalblue')
                        ax_params[i].set_title(f"P{i}: {next_x[i]:.2e}", fontsize=9)
                        ax_params[i].grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    plt.pause(0.001)
                except:
                    pass

            except RuntimeError as e:
                # 4. GESTION DES DIVERGENCES
                print(f"--> [DIVERGENCE FEniCSx] Le solveur n'a pas convergé.")
                print(f"--> Point rejeté. Le compteur de succès n'augmente pas.")
                
                # On pénalise le point pour que skopt n'y revienne plus
                opt.tell(next_x, 500.0)
                
                # Calcul du point de remplacement
                if successful_calls < n_lhs_target:
                    # En phase d'apprentissage : on passe simplement au point LHS suivant de notre réserve
                    print("--> Phase d'apprentissage : Pioche du point LHS suivant dans la réserve.")
                    next_x = lhs_points[lhs_index]
                    lhs_index += 1
                    
                    # Sécurité : si on arrive au bout des 50 points de la réserve, on en régénère
                    if lhs_index >= len(lhs_points):
                        print("--> Réserve LHS épuisée, génération de points supplémentaires...")
                        lhs_points = lhs_sampler.generate(dimensions, n_samples=20, random_state=iteration_total)
                        lhs_index = 0
                else:
                    # En phase d'exploitation : on redemande une suggestion à l'algorithme
                    next_x = opt.ask()
                
            except Exception as e:
                print(f"--> [ERREUR INATTENDUE] : {e}")
                opt.tell(next_x, 500.0)
                if successful_calls < n_lhs_target:
                    next_x = lhs_points[lhs_index]
                    lhs_index += 1
                else:
                    next_x = opt.ask()

    plt.ioff()
    plt.show()
    
    # Extraction du meilleur résultat
    best_index = np.argmin(opt.yi)
    print("\n--- OPTIMISATION TERMINÉE ---")
    print(f"Nombre total d'essais (Valides + Échecs) : {iteration_total}")
    print(f"Succès en phase LHS (Apprentissage) : {n_lhs_target}")
    print(f"Succès totaux enregistrés : {successful_calls}")
    print("Meilleurs paramètres trouvés :", opt.Xi[best_index])
    print("Plus petite erreur obtenue :", opt.yi[best_index])
    
    return opt


bounds_ref_yield = [
    (150_000, 250_000),   # E [MPa]
    (0.25, 0.35),         # nu
    (10.0, 500.0),        # sigma_Y [MPa] (référence direction 1)
    (0.0, 750.0),         # Q_var [MPa]
    (5.0, 50.0),          # k_hardening
    (0.7, 1.3),           # R22 = sigma_22^y / sigma_Y
    (0.7, 1.3),           # R33 = sigma_33^y / sigma_Y
    (0.5, 1.5),           # R12 = tau_12^y / sigma_Y
    (0.5, 1.5),           # R23 = tau_23^y / sigma_Y
    (0.5, 1.5),           # R31 = tau_31^y / sigma_Y
]

def femu(
        h5_file,
        params0=None,
        bounds=bounds_ref_yield,
        n_lhs_target=35,
        n_successful_calls_target=70,
    ):
    """
    Par défaut, si l'échantillonnage par Hypercube Latin (LHS) génère 25 points et que 5 d'entre eux font planter FEniCSx, 
    gp_minimize considère que sa phase d'apprentissage est terminée au bout de 25 itérations, 
    même s'il n'a que 20 vraies simulations réussies en mémoire. Le modèle de substitution démarre 
    alors sa phase d'exploitation avec un sérieux déficit d'apprentissage.
    Pour garantir que vous obtenez exactement 25 points initiaux LHS réussis, il faut légèrement adapter la boucle Ask & Tell.
    scikit-optimize ne permet pas de régénérer un point LHS à la demande à l'intérieur de l'objet Optimizer.
    L'astuce consiste à générer l'intégralité de vos points LHS à l'avance, puis à piocher dedans et à en recréer de nouveaux
    si certains échouent.
    """

    dimensions = [
        Real(bounds[i][0], bounds[i][1], name=f"p_{i}") 
        for i in range(len(bounds))
    ]

    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0])
    ax_params = [fig.add_subplot(gs[divmod(i, 4)[0], divmod(i, 4)[1]]) for i in range(1, 11)]
    
    history_err = []
    history_params = []

    # 1. Configuration de l'Optimizer (on désactive son n_random_starts interne car on gère à la main)
    opt = Optimizer(
        dimensions=dimensions,
        n_random_starts=0,  # Gestion manuelle
        acq_func="EI",
        random_state=42
    )

    # Objectifs de la simulation
    
    successful_calls = 0
    iteration_total = 0

    # 2. Génération manuelle de la réserve de points LHS
    # On en génère un peu plus (ex: 50) au cas où FEniCSx divergerait sur certains points
    lhs_sampler = Lhs(criterion="maximin")
    lhs_points = lhs_sampler.generate(dimensions, n_samples=50, random_state=42)
    lhs_index = 0

    with h5py.File(h5_file, 'r') as f:
        
        # Détermination du tout premier point à tester
        if params0 is not None:
            next_x = params0
            params0 = None
            using_lhs = False
        else:
            next_x = lhs_points[lhs_index]
            lhs_index += 1
            using_lhs = True

        # Boucle principale basée uniquement sur les VRAIS succès
        while successful_calls < n_successful_calls_target:
            iteration_total += 1
            
            # Message d'affichage pour suivre les deux phases distinctes
            if successful_calls < n_lhs_target:
                phase_str = f"Phase APPRENTISSAGE LHS (Succès: {successful_calls}/{n_lhs_target})"
            else:
                phase_str = f"Phase EXPLOITATION BAYÉSIENNE (Succès: {successful_calls}/{n_successful_calls_target})"
                using_lhs = False
                
            print(f"\n--- Itération globale n°{iteration_total} | {phase_str} ---")
            print(f"Paramètres testés : {[round(p, 2) for p in next_x]}")
            print(f"equivalents hill : {get_hill48_from_yield_ratios(sigma_Y=params[2], R22=params[5], R33=params[6],R12=params[7],R23=params[8],R31=params[9])}")
            
            try:
                # 3. Exécution FEniCSx
                error = compute_hill_raw_h5_error_from_parameters_yield_ratios(f, next_x)
                print(f"-> Succès ! Error calculée : {error}")
                
                # Envoi du succès à l'optimiseur pour qu'il mette à jour son processus gaussien
                opt.tell(next_x, error)
                
                successful_calls += 1
                history_err.append(error)
                history_params.append(next_x)
                
                # Choix du prochain point selon la phase actuelle
                if successful_calls < n_successful_calls_target:
                    if successful_calls < n_lhs_target:
                        # On continue de piocher dans notre réserve LHS
                        next_x = lhs_points[lhs_index]
                        lhs_index += 1
                    else:
                        # L'apprentissage est validé, on laisse l'optimiseur décider (Exploitation)
                        next_x = opt.ask()
                
                # Rafraîchissement graphique
                try:
                    data_p = np.array(history_params)
                    ax_err.clear()
                    ax_err.plot(history_err, color='firebrick', lw=1.5)
                    ax_err.set_yscale('log')
                    ax_err.set_title("Erreur (Log)")

                    for i in range(len(next_x)):
                        ax_params[i].clear()
                        ax_params[i].plot(data_p[:, i], color='royalblue')
                        ax_params[i].set_title(f"P{i}: {next_x[i]:.2e}", fontsize=9)
                        ax_params[i].grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    plt.pause(0.001)
                except:
                    pass

            except RuntimeError as e:
                # 4. GESTION DES DIVERGENCES
                print(f"--> [DIVERGENCE FEniCSx] Le solveur n'a pas convergé.")
                print(f"--> Point rejeté. Le compteur de succès n'augmente pas.")
                
                # On pénalise le point pour que skopt n'y revienne plus
                opt.tell(next_x, 500.0)
                
                # Calcul du point de remplacement
                if successful_calls < n_lhs_target:
                    # En phase d'apprentissage : on passe simplement au point LHS suivant de notre réserve
                    print("--> Phase d'apprentissage : Pioche du point LHS suivant dans la réserve.")
                    next_x = lhs_points[lhs_index]
                    lhs_index += 1
                    
                    # Sécurité : si on arrive au bout des 50 points de la réserve, on en régénère
                    if lhs_index >= len(lhs_points):
                        print("--> Réserve LHS épuisée, génération de points supplémentaires...")
                        lhs_points = lhs_sampler.generate(dimensions, n_samples=20, random_state=iteration_total)
                        lhs_index = 0
                else:
                    # En phase d'exploitation : on redemande une suggestion à l'algorithme
                    next_x = opt.ask()
                
            except Exception as e:
                print(f"--> [ERREUR INATTENDUE] : {e}")
                opt.tell(next_x, 500.0)
                if successful_calls < n_lhs_target:
                    next_x = lhs_points[lhs_index]
                    lhs_index += 1
                else:
                    next_x = opt.ask()

    plt.ioff()
    plt.show()
    
    # Extraction du meilleur résultat
    best_index = np.argmin(opt.yi)
    print("\n--- OPTIMISATION TERMINÉE ---")
    print(f"Nombre total d'essais (Valides + Échecs) : {iteration_total}")
    print(f"Succès en phase LHS (Apprentissage) : {n_lhs_target}")
    print(f"Succès totaux enregistrés : {successful_calls}")
    print("Meilleurs paramètres trouvés :", opt.Xi[best_index])
    print("Plus petite erreur obtenue :", opt.yi[best_index])
    
    return opt


print(get_hill48_from_yield_ratios(100))
if __name__ == "__main__":
    optimizer_result = femu("femu_files/res.h5", None, bounds_ref, n_lhs_target=40, n_successful_calls_target=80)