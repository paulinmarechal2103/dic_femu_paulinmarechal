import os
import h5py
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
h5_filename = "carre_trou_ortho.h5"  # Remplacez par le nom réel de votre fichier .h5
output_dir = "carre_trou_ortho_y0_10_csv"
OUTPUTFILENAME = "carre_trou_ortho_y0_10_"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(h5_filename, "r") as f:
    
    # 1. Extraction des coordonnées (Mesh/Grid/geometry)
    geo_path = 'Mesh/Grid/geometry'
    if geo_path not in f:
        raise KeyError(f"Le chemin {geo_path} n'a pas été trouvé dans le fichier HDF5.")
        
    coords = f[geo_path][:]
    X = coords[:, 0]
    Y = coords[:, 1]
    
    # Condition sur Y : entre -7 et 7
    y_mask = (Y >= -0.0) & (Y <= 10.0)
    
    # Condition sur Z : z proche de 0
    if coords.shape[1] > 2:
        Z = coords[:, 2]
        z_mask = np.isclose(Z, 0.0, atol=1e-5)
    else:
        Z = np.zeros_like(X)
        z_mask = np.ones_like(X, dtype=bool)
    
    # Combinaison des deux masques (Z == 0 ET -7 <= Y <= 7)
    surface_mask = z_mask & y_mask
    
    # Application du filtre sur les coordonnées
    X_surf = X[surface_mask]
    Y_surf = Y[surface_mask]
    Z_surf = Z[surface_mask]
    
    if len(X_surf) == 0:
        print("ATTENTION : Aucun point trouvé avec Z = 0 et Y entre -7 et 7. Vérifiez vos coordonnées.")
    else:
        print(f"Nombre de points identifiés (Z = 0 et -7 <= Y <= 7) : {len(X_surf)}")
    
    # 3. Extraction du déplacement (Function/displacement)
    func_path = 'Function/displacement'
    if func_path not in f:
        raise KeyError(f"Le groupe de fonction {func_path} n'existe pas.")
        
    u_group = f[func_path]
    
    # Récupération et tri des pas de temps
    steps = sorted([int(k) for k in u_group.keys() if k.isdigit()])
    
    print(f"Extraction du déplacement pour {len(steps)} pas de temps...")
    
    for step in steps:
        u_data = u_group[str(step)][:]
        
        # Si le vecteur est aplati (N * dimension), on le reforme
        if len(u_data.shape) == 1:
            u_data = u_data.reshape(-1, coords.shape[1])
            
        # Extraction des déplacements uniquement pour les points filtrés
        u_surf = u_data[surface_mask]
        
        # Structuration des données dans le DataFrame
        df_step = pd.DataFrame({
            'x': X_surf,
            'y': Y_surf,
            'z': Z_surf,
            'u': u_surf[:, 0],  # Déplacement selon X
            'v': u_surf[:, 1],  # Déplacement selon Y
        })
        
        # Ajout de la composante hors-plan si le modèle est 3D
        if coords.shape[1] > 2:
            df_step['w_FE'] = u_surf[:, 2]  # Déplacement selon Z
            
        # Ajout de la colonne "sigma" remplie de 1
        df_step['sigma'] = 1
            
        # Sauvegarde du fichier pour le pas de temps actuel
        csv_filename = os.path.join(output_dir, f"{OUTPUTFILENAME}{step:04d}.csv")
        df_step.to_csv(csv_filename, index=False)
        
    print(f"Extraction terminée. Les fichiers filtrés sont dans le dossier : '{output_dir}'.")
