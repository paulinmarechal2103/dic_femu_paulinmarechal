import optuna
import numpy as np
import matplotlib.pyplot as plt
import h5py

import h5py
import numpy as np


from plasticity_simu import *
from hill48_model import Hill48Model,Hill48state


bounds_ref = [
    (150_000, 250_000),   # E [MPa] : acier, OK
    (0.25, 0.35),         # nu : métaux typiques
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

def femu_optuna(
        h5_file,
        params0=[200_500.0, 0.29, 105.0, 52.0, 8.0, 0.52, 0.52, 0.48, 1.52, 1.48, 1.45],
        bounds=bounds_ref,
        n_startup_trials=35,
        n_successful_calls_target=70
    ):

    param_names = ['E', 'nu', 'sigma_Y', 'Q_var', 'k_hardening', 'F', 'G', 'H', 'L', 'M', 'N']

    # --- Callback d'arrêt uniquement ---
    def stop_callback(study, trial):
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_success = len(complete_trials)
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"-> Essai {trial.number + 1} RÉUSSI | Succès : {n_success}/{n_successful_calls_target} | Erreur : {trial.value:.4e}")
        else:
            print(f"-> Essai {trial.number + 1} ÉCHOUÉ/PRUNED (Écarté par l'optimiseur)")

        if n_success >= n_successful_calls_target:
            study.stop()

    # --- Configuration du Stockage SQLite ---
    # C'est ce fichier .db qui va servir de pont avec VS Code
    storage_url = "sqlite:///femu_optimization.db"
    study_name = "femu_hill48_study"

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42, multivariate=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Création ou chargement de l'étude
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True  # Bonus : Si le code coupe, il reprend là où il s'est arrêté !
    )

    # Injection du point initial (uniquement si la base est neuve)
    if params0 is not None and len(study.trials) == 0:
        initial_dict = {name: val for name, val in zip(param_names, params0)}
        study.enqueue_trial(initial_dict)

    # --- Boucle d'optimisation ---
    with h5py.File(h5_file, 'r') as f:
        
        def objective(trial):
            params = [trial.suggest_float(name, bounds[i][0], bounds[i][1]) for i, name in enumerate(param_names)]
            try:
                error = compute_hill_raw_h5_error_from_parameters(f, params)
                return error
            except (RuntimeError, ValueError, Exception):
                raise optuna.exceptions.TrialPruned()

        study.optimize(objective, n_trials=None, callbacks=[stop_callback])

    print("\n--- OPTIMISATION TERMINÉE ---")
    return study

if __name__ == "__main__":
    study_result = femu_optuna("femu_files/res.h5", bounds=bounds_ref)