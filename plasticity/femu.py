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
            diff = np.linalg.norm(d1 - d2)**2
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

def is_hill48_physically_valid(params):
    # params = [E, nu, sigma_Y, Q, k, F, G, H, L, M, N]
    F, G, H, L, M, N = params[5:11]
    
    # 1. Positivité stricte
    if any(p <= 1e-6 for p in [F, G, H, L, M, N]):
        return False
    
    # 2. Conditions de convexité pratique (ratios raisonnables)
    if (F+G) <= 0 or (G+H) <= 0 or (H+F) <= 0:
        return False
        
    # Évite les anisotropies trop brutales qui font exploser le solveur plastique
    if max(F,G,H,L,M,N) / min(F,G,H,L,M,N) > 3.0:  
        return False
        
    # 3. Cohérence écrouissage / élasticité
    E, Q, k = params[0], params[3], params[4]
    if Q * k > 0.15 * E:  
        return False
        
    return True

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

    # if not is_hill48_physically_valid(params):
    #     print("--> [REJET PRÉ-FEM] Paramètres non physiques ou Hill48 non convexe.")
    #     raise ValueError("Hill48 non convexe ou paramètres non physiques")

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


def compute_J2_raw_h5_error_from_parameters(f, params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]): 
    """
    Compute the total displacement difference between 
    H5 reference raw file extracted with h5py
    and simulation output array for a given set of Hill48 parameters.

    params should be a list or array containing the following parameters in order:
    (E, nu, sigma_Y, Q_var, k_hardening)
    
    """
    hill_params = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 2,
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
    )

    # if not is_hill48_physically_valid(params):
    #     print("--> [REJET PRÉ-FEM] Paramètres non physiques ou Hill48 non convexe.")
    #     raise ValueError("Hill48 non convexe ou paramètres non physiques")

    model = J2IsotropicHardening(
        elastic=ElasticModel(hill_params["E"], hill_params["nu"], tdim=3),
        sigma_Y=hill_params["sigma_Y"],
        Q_var=hill_params["Q_var"],
        k=hill_params["k_hardening"]
    )

    try:
        # On tente de lancer la simulation dolfinx
        _, u_sim = run_simulation_V2(hill_params, model=model, write_output=False)
        error = compute_u_sim_raw_h5_diff(f, u_sim)
    except RuntimeError as e:
        # Si le solveur de Newton échoue, on ne crash pas !
        print(f"--> [Newton Divergence] Paramètres instables détectés. Pénalisation de l'erreur.")
        # On renvoie une erreur artificiellement grande pour dire à SciPy de rebrousser chemin
        error = 1e3
    return error

# with h5py.File("femu_files/res.h5", 'r') as f:
#     diffs = compute_hill_raw_h5_error_from_parameters(f)
#     print(f"Différence totale : {diffs}")


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
    (150_000, 250_000),   # E [MPa]
    (0.25, 0.35),         # nu 
    (10.0, 500.0),        # sigma_Y [MPa]
    (20.0, 400.0),         # Q_var [MPa]
    (10.0, 1500.0),          # k_hardening
]



def femu_V1(h5_file, params0 = [200_000.0, 0.3, 100.0, 50.0, 1_000.0, 0.900, 0.600, 0.400, 1.7, 1.3, 1.350]):
    with h5py.File(h5_file, 'r') as f:
        def objective_function(params):
            print(params)
            return compute_hill_raw_h5_error_from_parameters(f, params)
        result = minimize(objective_function, params0, method='BFGS')
        print("Optimized parameters:", result.x)
        print("Minimum error:", result.fun)
    return result



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
            options={'ftol': 1e-8, 'maxiter': 150, 'disp': True, 'eps': 1e-3}
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



def femu_V3(
    h5_file,
    params0=[200_500.0, 0.29, 102.0, 52.0, 1_010.0],
    bounds=bounds_ref
):
    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    
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
            
            print(f"Current params (phys): {params_phys}")
            
            # 2. Calcul de l'erreur avec les valeurs physiques
            error = compute_J2_raw_h5_error_from_parameters(f, params_phys)
            
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
            options={'ftol': 1e-5, 'gtol': 1e-4, 'maxiter': 150, 'disp': True, 'eps': 1e-3}
        )
        
    plt.ioff()
    plt.show()
    
    # --- Post-traitement ---
    # On reconstruit l'objet résultat pour renvoyer les paramètres physiques optimaux
    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds))
    
    return result_phys


if __name__ == "__main__":
    from random import uniform,seed
    seed(42)  # Pour la reproductibilité
    perturbation_percentage = 0.01  # 1% de perturbation aléatoire
    normalized_result = normalize_params([200_500.0, 0.29, 102.0, 52.0, 1_010.0], bounds_ref_J2)
    normalized_disturbed = [i + uniform(-perturbation_percentage, perturbation_percentage) for i in normalized_result]
    parameters_disturbed = denormalize_params(normalized_disturbed, bounds_ref_J2)
    optimizer_result = femu_V3("femu_files/res.h5", parameters_disturbed, bounds_ref_J2)
    print("Optimized parameters (phys):", optimizer_result.x)


# if __name__ == "__main__":
#     optimizer_result = femu("femu_files/res.h5", None, bounds_ref, 35, 70)        