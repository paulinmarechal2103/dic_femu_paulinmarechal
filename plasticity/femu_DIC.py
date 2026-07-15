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

# def compute_u_residuals_is_imported(f, u_sim, V, dataset_path="/Function/displacement_projected",
#                                      mask_value=0.1, atol=1e-6, tol_max_dist=1e-5):
#     """
#     Calcule les résidus de déplacement entre le champ de référence (H5)
#     et le champ simulé u_sim (liste de uh.x.array.copy(), un par pas de temps),
#     uniquement aux points où is_imported == mask_value.

#     La correspondance géométrique H5 <-> DOFs de `V` se fait via un KDTree
#     découpé par composante (subspace collapse) pour s'affranchir du layout de dolfinx.
#     """
#     gdim = V.mesh.geometry.dim
#     vdim = V.dofmap.bs

#     # --- 1. Extraction et application du masque is_imported ---
#     is_imported_path = "/Function/is_imported"
#     if is_imported_path not in f:
#         raise KeyError(f"La fonction '{is_imported_path}' est introuvable dans le fichier H5.")

#     is_imported_group = f[is_imported_path]
#     if isinstance(is_imported_group, h5py.Dataset):
#         is_imported = is_imported_group[:].flatten()
#     else:
#         is_imported = is_imported_group["0"][:].flatten()

#     mask = np.isclose(is_imported, mask_value, atol=atol)
#     if mask.sum() == 0:
#         raise ValueError(f"Aucun point avec is_imported ≈ {mask_value} trouvé.")

#     # --- 2. Recherche récursive automatique du dataset 'geometry' ---
#     def find_geometry_dataset(group):
#         if "geometry" in group and isinstance(group["geometry"], h5py.Dataset):
#             return group["geometry"][:]
#         for key in group.keys():
#             if isinstance(group[key], h5py.Group):
#                 res = find_geometry_dataset(group[key])
#                 if res is not None:
#                     return res
#         return None

#     points_parent = find_geometry_dataset(f)
#     if points_parent is None:
#         raise KeyError(
#             f"Impossible de localiser un dataset nommé 'geometry' dans l'arborescence du fichier H5. "
#             f"Groupes présents à la racine : {list(f.keys())}"
#         )

#     # Extraction des coordonnées des points d'intérêt
#     masked_points = points_parent[mask, :gdim]

#     # --- 3. Cartographie géométrique des DOFs (Subspace Collapse) ---
#     num_masked_points = len(masked_points)
#     flat_indices = np.zeros((num_masked_points, vdim), dtype=np.int32)

#     for comp in range(vdim):
#         # On isole le sous-espace associé à la composante courante (X, Y ou Z)
#         sub_V, sub_to_parent = V.sub(comp).collapse()
#         sub_coords = sub_V.tabulate_dof_coordinates()[:, :gdim]

#         # Appariement géométrique par composante
#         tree = KDTree(sub_coords)
#         distances, sub_dof_indices = tree.query(masked_points)

#         if np.any(distances > tol_max_dist):
#             max_d = np.max(distances)
#             raise ValueError(
#                 f"Erreur d'appariement géométrique sur la composante {comp} : "
#                 f"la distance max ({max_d:.2e}) dépasse la tolérance tol_max_dist ({tol_max_dist:.2e})."
#             )

#         # Stockage de la table de correspondance inverse fournie par dolfinx
#         flat_indices[:, comp] = sub_to_parent[sub_dof_indices]

#     # --- 4. Boucle sur les pas de temps et calcul des résidus ---
#     if dataset_path not in f:
#         raise KeyError(f"Le chemin '{dataset_path}' est introuvable dans le fichier H5.")

#     displacement_group = f[dataset_path]
#     errors = []
#     step = 0

#     while str(step) in displacement_group:
#         if step >= len(u_sim):
#             raise IndexError(
#                 f"Le pas de temps {step} est requis par le fichier H5 mais absent de la liste u_sim "
#                 f"(taille de u_sim : {len(u_sim)})."
#             )

#         # Référence H5 : forme (N_points_total, vdim) -> Extraction des lignes masquées
#         d1 = displacement_group[str(step)][:][mask, :vdim]
        
#         # Simulation : Extraction via la matrice d'indices (N_masked, vdim)
#         d2 = u_sim[step][flat_indices]

#         # Différence brute aplatie pour least_squares
#         diff = (d1 - d2).flatten()
#         errors.append(diff)
#         step += 1

#     if step == 0:
#         raise ValueError(f"Aucun pas de temps n'a pu être lu sous le chemin {dataset_path}.")

#     return np.concatenate(errors)

def compute_u_residuals_is_imported(f, u_sim, V, dataset_path="/Function/displacement_projected",
                                          mask_value=0.1, atol=1e-6, tol_max_dist=1e-5, atol_z=1e-6):
    """
    Calcule les résidus de déplacement (composantes X et Y uniquement) entre H5 et u_sim,
    pour les points où is_imported == mask_value ET z == 0.
    """
    gdim = V.mesh.geometry.dim
    vdim = V.dofmap.bs

    # --- 1. Extraction du masque is_imported ---
    is_imported_path = "/Function/is_imported"
    if is_imported_path not in f:
        raise KeyError(f"La fonction '{is_imported_path}' est introuvable dans le fichier H5.")

    is_imported_group = f[is_imported_path]
    if isinstance(is_imported_group, h5py.Dataset):
        is_imported = is_imported_group[:].flatten()
    else:
        is_imported = is_imported_group["0"][:].flatten()

    mask_imported = np.isclose(is_imported, mask_value, atol=atol)

    # --- 2. Extraction de la géométrie et création du masque Z=0 ---
    def find_geometry_dataset(group):
        if "geometry" in group and isinstance(group["geometry"], h5py.Dataset):
            return group["geometry"][:]
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                res = find_geometry_dataset(group[key])
                if res is not None:
                    return res
        return None

    points_parent = find_geometry_dataset(f)
    if points_parent is None:
        raise KeyError("Impossible de localiser le dataset 'geometry' dans le H5.")

    # Création du masque combiné : is_imported ET z=0
    if gdim >= 3:
        z_coords = points_parent[:, 2]
        mask_z = np.isclose(z_coords, 0.0, atol=atol_z)
        final_mask = mask_imported & mask_z
    else:
        # Si le maillage est 2D, Z=0 est implicite
        final_mask = mask_imported

    if final_mask.sum() == 0:
        raise ValueError(f"Aucun point avec is_imported ≈ {mask_value} ET Z ≈ 0 trouvé.")

    masked_points = points_parent[final_mask, :gdim]

    # --- 3. Cartographie géométrique des DOFs pour X et Y uniquement ---
    num_masked_points = len(masked_points)
    
    # On définit explicitement les composantes ciblées : 0 (X) et 1 (Y)
    active_comps = [0, 1] if vdim >= 2 else [0]
    flat_indices = np.zeros((num_masked_points, len(active_comps)), dtype=np.int32)

    for i, comp in enumerate(active_comps):
        sub_V, sub_to_parent = V.sub(comp).collapse()
        sub_coords = sub_V.tabulate_dof_coordinates()[:, :gdim]

        tree = KDTree(sub_coords)
        distances, sub_dof_indices = tree.query(masked_points)

        if np.any(distances > tol_max_dist):
            max_d = np.max(distances)
            raise ValueError(
                f"Erreur d'appariement sur la composante {comp} : "
                f"distance max ({max_d:.2e}) > tolérance ({tol_max_dist:.2e})."
            )

        flat_indices[:, i] = sub_to_parent[sub_dof_indices]

    # --- 4. Calcul des résidus sur X et Y ---
    if dataset_path not in f:
        raise KeyError(f"Le chemin '{dataset_path}' est introuvable dans le fichier H5.")

    displacement_group = f[dataset_path]
    errors = []
    step = 0

    while str(step) in displacement_group:
        if step >= len(u_sim):
            raise IndexError(f"Le pas de temps {step} est absent de u_sim.")

        # Référence H5 : application du masque combiné puis sélection des colonnes X et Y
        d1 = displacement_group[str(step)][:][final_mask][:, active_comps]
        
        # Simulation : Extraction via la matrice d'indices (seulement X et Y)
        d2 = u_sim[step][flat_indices]

        diff = (d1 - d2).flatten()
        errors.append(diff)
        step += 1

    if step == 0:
        raise ValueError(f"Aucun pas de temps n'a pu être lu sous le chemin {dataset_path}.")

    return np.concatenate(errors)

def compute_J2_residuals_DIC_BC(domain,V, W, WT,f,h5_path, params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]): 
    """
    Compute the total displacement difference between 
    H5 reference raw file extracted with h5py
    and simulation output array for a given set of Hill48 parameters.

    params should be a list or array containing the following parameters in order:
    (E, nu, sigma_Y, Q_var, k_hardening)
    
    """
    params = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    h5_bc_path = h5_path,
    h5_function_path = "/Function/displacement_projected",
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
        elastic=ElasticModel(params["E"], params["nu"], tdim=3),
        sigma_Y=params["sigma_Y"],
        Q_var=params["Q_var"],
        k=params["k_hardening"]
    )

    try:
        # On tente de lancer la simulation dolfinx
        _, u_sim = run_simulation_bc_h5_fast(domain, V, W, WT, params, model=model)
        error = compute_u_residuals_is_imported(f, u_sim,V, params["h5_function_path"])
    except RuntimeError as e:
        # Si le solveur de Newton échoue, on ne crash pas !
        print(f"--> [Newton Divergence] Paramètres instables détectés. Pénalisation de l'erreur.")
        # On renvoie une erreur artificiellement grande pour dire à SciPy de rebrousser chemin
        error = 1e3
    return error

def compute_hill_residuals_DIC_BC(domain,V, W, WT,f, params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0, 0.900, 0.600, 0.400, 1.7, 1.3, 1.350]):
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

    model_hill48 = Hill48Model(
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

    try:
        # On tente de lancer la simulation dolfinx
        _, u_sim = run_simulation_fast(domain, V, W, WT, hill_params,1, model=model_hill48, write_output=False)
        error = compute_u_residuals(f, u_sim)
    except RuntimeError as e:
        # Si le solveur de Newton échoue, on ne crash pas !
        print(f"--> [Newton Divergence] Paramètres instables détectés. Pénalisation de l'erreur.")
        # On renvoie une erreur artificiellement grande pour dire à SciPy de rebrousser chemin
        error = 1e3
    return error


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




from datetime import datetime



def femu_res_J2_DIC_BC(
        XDMF_FILE,
        params0=[200_500.0, 0.29, 102.0, 52.0, 1_010.0],
        bounds=bounds_ref_J2,
        params_names=["E", "nu", "sigma_Y", "Q_var", "k_hardening"]
    ):
    """
    domain : dolfinx.mesh.Mesh déjà créé à partir du maillage PyVista
    V, W, WT : espaces de fonctions déjà construits pour ce maillage
    h5_file : chemin vers le fichier de référence contenant les déplacements mesurés
    params0 : liste des paramètres initiaux pour l'optimisation (E, nu, sigma_Y, Q_var, k_hardening)
    bounds : liste des tuples (min, max) pour chaque paramètre, utilisée pour la normalisation et les contraintes de l'optimiseur
    """
    with io.XDMFFile(MPI.COMM_WORLD, XDMF_FILE, "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
    V, W, WT = build_function_spaces(domain)

    h5_file = XDMF_FILE.replace(".xdmf", ".h5") 

    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    
    # 11 paramètres + 1 erreur = 12 slots (3 lignes x 4 colonnes)
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0]) # Erreur en haut à gauche
    
    # On crée les axes pour les paramètres
    ax_params = []
    for i in range(1, 6):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []    # Stockera la norme scalaire (somme des carrés) pour le plot
    history_params = [] # Stockera les paramètres PHYSIQUES (dénormalisés) pour le plot

    # --- Préparation de la Normalisation pour SciPy ---
    params0_norm = normalize_params(params0, bounds)
    bounds_norm = Bounds([0.0]*len(params0), [1.0]*len(params0))

    with h5py.File(h5_file, 'r') as f:
        def objective_function(params_norm):
            # 1. Dénormalisation pour retrouver les valeurs physiques
            params_phys = denormalize_params(params_norm, bounds)
            
            print(f"{str(datetime.now())} \nsimu EF n°{len(history_err)}, iteration = {len(history_err)//(len(params_norm)+1)}\nCurrent params (phys): {params_phys}\n")
            
            # 2. Calcul des résidus (doit être un vecteur/array 1D pour least_squares)
            residuals = compute_J2_residuals_DIC_BC(domain, V, W, WT, f, h5_file, params_phys)
            
            # 3. Calcul de la norme (scalaire) pour le suivi graphique
            # least_squares minimise (0.5 * sum(r**2)). On stocke la somme des carrés.
            error_scalar = np.sum(np.square(residuals))
            
            # 4. Stockage pour l'historique
            history_err.append(error_scalar)
            history_params.append(params_phys)
            data_p = np.array(history_params)
            
            # 5. Mise à jour graphique (avec les valeurs physiques)
            # Rafraîchissement estimé à chaque itération de l'optimiseur (Jacobien inclus)
            if True: #len(history_err) % (len(params_norm) + 1) == 0: 
                try:
                    # Plot Erreur (Somme des carrés des résidus)
                    ax_err.clear()
                    ax_err.plot(history_err, color='firebrick', lw=1.5)
                    ax_err.set_yscale('log')
                    ax_err.set_title("Norme Résidus (Log $\sum r^2$)")
                    ax_err.grid(True, which="both", ls="-", alpha=0.2)

                    # Plot Paramètres (Physiques)
                    for i in range(len(params_phys)):
                        ax_params[i].clear()
                        ax_params[i].plot(data_p[:, i], color='royalblue')
                        ax_params[i].set_title(f"{params_names[i]}: {params_phys[i]:.2e}", fontsize=9)
                        ax_params[i].grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    plt.pause(0.001)
                except Exception as e:
                    # Permet de continuer si la fenêtre est fermée ou s'il y a un souci d'affichage
                    pass
                    
            print(f"Residuals norm (sum of squares): {error_scalar}")
            
            # least_squares a STRICTEMENT besoin du vecteur de résidus brut
            return residuals

        # L'optimiseur reçoit les versions normalisées (0 à 1)
        result_norm = least_squares(
            objective_function,
            params0_norm,
            method='trf',
            bounds=bounds_norm,
            ftol=1e-8, gtol=1e-7, max_nfev=150, verbose=2, x_scale=1.0, diff_step=1e-2,
            # loss='soft_l1',     # Filtre les résidus aberrants de la DIC
            # f_scale=0.01,       # Seuil de résidu typique au-delà duquel on atténue (à ajuster)
        )
        
    plt.ioff()
    plt.show()
    
    # --- Post-traitement ---
    # On reconstruit l'objet résultat pour renvoyer les paramètres physiques optimaux
    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds))
    
    return result_phys

def femu_res_hill_DIC_BC(
        XDMF_FILE,
        params0=[200_500.0, 0.29, 102.0, 52.0, 1_010.0],
        bounds=bounds_ref,
        params_names=["E", "nu", "sigma_Y", "Q_var", "k_hardening", "F", "G", "H", "L", "M", "N"]
    ):
    """
    domain : dolfinx.mesh.Mesh déjà créé à partir du maillage PyVista
    V, W, WT : espaces de fonctions déjà construits pour ce maillage
    params0 : liste des paramètres initiaux pour l'optimisation (E, nu, sigma_Y, Q_var, k_hardening)
    bounds : liste des tuples (min, max) pour chaque paramètre, utilisée pour la normalisation et les contraintes de l'optimiseur
    """
    with io.XDMFFile(MPI.COMM_WORLD, XDMF_FILE, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
    V, W, WT = build_function_spaces(domain)
    # --- Configuration du Plot ---
    plt.ion()
    fig = plt.figure(figsize=(16, 10))
    
    # 11 paramètres + 1 erreur = 12 slots (3 lignes x 4 colonnes)
    gs = fig.add_gridspec(3, 4)
    ax_err = fig.add_subplot(gs[0, 0]) # Erreur en haut à gauche
    
    # On crée les axes pour les paramètres
    ax_params = []
    for i in range(1, len(params_names)+1):
        row, col = divmod(i, 4)
        ax_params.append(fig.add_subplot(gs[row, col]))
    
    history_err = []    # Stockera la norme scalaire (somme des carrés) pour le plot
    history_params = [] # Stockera les paramètres PHYSIQUES (dénormalisés) pour le plot

    # --- Préparation de la Normalisation pour SciPy ---
    params0_norm = normalize_params(params0, bounds)
    bounds_norm = Bounds([0.0]*len(params0), [1.0]*len(params0))
    h5_file = XDMF_FILE.replace(".xdmf", ".h5") 


    with h5py.File(h5_file, 'r') as f:
        def objective_function(params_norm):
            # 1. Dénormalisation pour retrouver les valeurs physiques
            params_phys = denormalize_params(params_norm, bounds)
            
            print(f"{str(datetime.now())} \nsimu EF n°{len(history_err)}, iteration = {len(history_err)//(len(params_norm)+1)}\nCurrent params (phys): {[f'{params_names[i]}: {params_phys[i]:.2e}' for i in range(len(params_phys))]}\n")
            # 2. Calcul des résidus (doit être un vecteur/array 1D pour least_squares)
            residuals = compute_hill_residuals(domain, V, W, WT, f, params_phys)
            
            # 3. Calcul de la norme (scalaire) pour le suivi graphique
            # least_squares minimise (0.5 * sum(r**2)). On stocke la somme des carrés.
            error_scalar = np.sum(np.square(residuals))
            
            # 4. Stockage pour l'historique
            history_err.append(error_scalar)
            history_params.append(params_phys)
            data_p = np.array(history_params)
            
            # 5. Mise à jour graphique (avec les valeurs physiques)
            # Rafraîchissement estimé à chaque itération de l'optimiseur (Jacobien inclus)
            if True: #len(history_err) % (len(params_norm) + 1) == 0: 
                try:
                    # Plot Erreur (Somme des carrés des résidus)
                    ax_err.clear()
                    ax_err.plot(history_err, color='firebrick', lw=1.5)
                    ax_err.set_yscale('log')
                    ax_err.set_title("Norme Résidus (Log $\sum r^2$)")
                    ax_err.grid(True, which="both", ls="-", alpha=0.2)

                    # Plot Paramètres (Physiques)
                    for i in range(len(params_phys)):
                        ax_params[i].clear()
                        ax_params[i].plot(data_p[:, i], color='royalblue')
                        ax_params[i].set_title(f"{params_names[i]}: {params_phys[i]:.2e}", fontsize=9)
                        ax_params[i].grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    plt.pause(0.001)
                except Exception as e:
                    # Permet de continuer si la fenêtre est fermée ou s'il y a un souci d'affichage
                    pass
                    
            print(f"Residuals norm (sum of squares): {error_scalar}")
            
            # least_squares a STRICTEMENT besoin du vecteur de résidus brut
            return residuals

        # L'optimiseur reçoit les versions normalisées (0 à 1)
        result_norm = least_squares(
            objective_function,
            params0_norm,
            method='trf',
            bounds=bounds_norm,
            ftol=1e-10, gtol=1e-10,   # empêche l'arrêt prématuré
            max_nfev=150, verbose=2,
            x_scale='jac',
            diff_step=1e-2            # perturbation plus grande que 1e-3 pour sortir du bruit
        )
        
    plt.ioff()
    plt.show()
    
    # --- Post-traitement ---
    # On reconstruit l'objet résultat pour renvoyer les paramètres physiques optimaux
    result_phys = result_norm
    result_phys.x = np.array(denormalize_params(result_norm.x, bounds))
    
    return result_phys


if __name__ == "__main__":
    bounds_ref_J2_centr= [
        (150_000, 250_000),   # E [MPa]
        (0.25, 0.35),         # nu 
        (10.0, 500.0),        # sigma_Y [MPa]
        (20.0, 400.0),         # Q_var [MPa]
        (10.0, 1500.0),          # k_hardening
    ]

    XDMF_FILE = "MAINTEST/projection_cad_temporelle_mask.xdmf"
    #XDMF_FILE = "results/projection_cad_temporelle_mask.xdmf"
    
    real_params = [200_000.0, 0.3, 100.0, 50.0, 1_000.0]
    params_names = ["E", "nu", "sigma_Y", "Q_var", "k_hardening"]
    from random import uniform,seed
    seed(43)  # Pour la reproductibilité
    perturbation_percentage = 0.15  # 5% de perturbation aléatoire
    normalized_result = normalize_params(real_params, bounds_ref_J2_centr)
    normalized_disturbed = [i + uniform(-perturbation_percentage, perturbation_percentage) for i in normalized_result]
    normalized_disturbed = [min(max(i, 0.0), 1.0) for i in normalized_disturbed]  # Clamp entre 0 et 1
    parameters_disturbed = denormalize_params(normalized_disturbed, bounds_ref_J2_centr)
    optimizer_result = femu_res_J2_DIC_BC(XDMF_FILE, bounds=bounds_ref_J2_centr, params0=parameters_disturbed,params_names = params_names)
    
    print("Optimized parameters (phys):", optimizer_result.x)
    print("normalized error:", [f"{params_names[i]} : {round(abs(optimizer_result.x[i] - real_params[i])/abs(real_params[i])*100,5)}%" for i in range(len(real_params))])
