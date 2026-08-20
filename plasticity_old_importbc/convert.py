import h5py
import numpy as np

# 1. Inspection du fichier pour valider la structure
with h5py.File('astar_2D_coarse.h5', 'r') as f:
    print("Structure du fichier H5 :")
    for key in f.keys():
        print(f" - {key}: shape = {f[key].shape}, dtype = {f[key].dtype}")

    data0 = f['data0'][:]
    data1 = f['data1'][:]

# 2. Conversion si data0=Nœuds et data1=Éléments (Cas le plus probable)
if len(data0.shape) == 2 and data0.shape[1] in [2, 3]:
    print("\nStructure détectée : Liste de nœuds et d'éléments.")
    
    # Gmsh requiert des coordonnées en 3D (X, Y, Z)
    if data0.shape[1] == 2:
        nodes = np.hstack([data0, np.zeros((data0.shape[0], 1))])
    else:
        nodes = data0

    # Choix du type d'élément Gmsh (2 = Triangle à 3 nœuds, 3 = Quadrangle à 4 nœuds)
    if data1.shape[1] == 3:
        msh_elm_type = 2 
    elif data1.shape[1] == 4:
        msh_elm_type = 3
    else:
        msh_elm_type = 1 # Ligne / Autre

    # Écriture du fichier au format MSH 2.2 (standard et très compatible)
    with open('astar_2D.msh', 'w') as msh:
        msh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # Nœuds
        msh.write("$Nodes\n")
        msh.write(f"{len(nodes)}\n")
        for i, coord in enumerate(nodes):
            msh.write(f"{i+1} {coord[0]} {coord[1]} {coord[2]}\n")
        msh.write("$EndNodes\n")
        
        # Éléments
        msh.write("$Elements\n")
        msh.write(f"{len(data1)}\n")
        for i, conn in enumerate(data1):
            # Format Gmsh 2.2 : id_élément type_élément nb_tags tag1 tag2 nœud1 nœud2...
            # On ajoute +1 car Gmsh commence l'indexation à 1 (contrairement à Python qui commence à 0)
            nodes_str = " ".join(str(int(n) + 1) for n in conn)
            msh.write(f"{i+1} {msh_elm_type} 2 1 1 {nodes_str}\n")
        msh.write("$EndElements\n")
        
    print("\nFichier 'astar_2D.msh' généré avec succès !")

else:
    print("\nLa structure correspond plutôt à une grille de pixels/images (Grid Map).")
    print("Si c'est le cas, indique-moi ce que révèlent les 'shapes' du premier print afin que je t'ajuste le script pour transformer la grille en maillage.")