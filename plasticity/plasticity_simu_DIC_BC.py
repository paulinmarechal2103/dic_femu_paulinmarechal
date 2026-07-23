# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
from abc import ABC, abstractmethod

import meshio
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import h5py
from plasticity_simu import *
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import dolfinx
import pyvista as pv
import basix.ufl
from time import sleep

# ---------------------------------------------------------------------------
# Default simulation configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    t_start     = 0.0,
    T           = 3.0,
    num_steps   = 50,
    load_amp    = 0.01,       # amplitude of the applied displacement
    length      = 10.0,       # half-length of the specimen
    mesh_file   = "Flat_specimen_refined.msh",
    output_dir  = "results_plasticity",
    file_name    = "res",
    # Elastic constants (used when no model is supplied)
    E           = 200_000.0,
    nu          = 0.3,
    # J2 isotropic hardening parameters (used when no model is supplied)
    sigma_Y     = 100.0,
    Q_var       = 50.0,
    k_hardening = 1000.0,
)

def load_domain_from_vtu(vtu_path, comm=MPI.COMM_WORLD):
    """
    Lit un fichier .vtu avec PyVista et reconstruit un dolfinx.mesh.Mesh.
    Fonctionne pour les maillages 2D et 3D homogènes.
    """
    import pyvista as pv
    import dolfinx
    import ufl
    
    pv_mesh = pv.read(vtu_path)
    points = pv_mesh.points  # Shape: (N, 3)
    
    cells_dict = pv_mesh.cells_dict
    if not cells_dict:
        raise ValueError("Le fichier VTU ne contient pas de maillage ou de cellules valides.")
        
    if len(cells_dict) > 1:
        raise ValueError("Le maillage contient des types de cellules mixtes, non supportés en natif.")
    
    vtk_type, cells = list(cells_dict.items())[0]
    
    if vtk_type == 5:    # VTK_TRIANGLE
        cell_type = dolfinx.mesh.CellType.triangle
    elif vtk_type == 9:  # VTK_QUAD
        cell_type = dolfinx.mesh.CellType.quadrilateral
    elif vtk_type == 10: # VTK_TETRA
        cell_type = dolfinx.mesh.CellType.tetrahedron
    elif vtk_type == 12: # VTK_HEXAHEDRON
        cell_type = dolfinx.mesh.CellType.hexahedron
    else:
        raise NotImplementedError(f"Le type VTK {vtk_type} n'est pas encore mappé.")
    
    gdim = points.shape[1]
    coordinate_element = basix.ufl.element("Lagrange", cell_type.name, 1, shape=(gdim,))
    domain = dolfinx.mesh.create_mesh(comm, cells, ufl.Mesh(coordinate_element), points)
    return domain

def get_vtu_files_from_pvd(pvd_file_path):
    """
    Parse un fichier .pvd pour extraire la liste ordonnée des fichiers .vtu associés.
    """
    tree = ET.parse(pvd_file_path)
    root = tree.getroot()
    vtu_files = []
    
    base_dir = os.path.dirname(pvd_file_path)
    for dataset in root.iter('DataSet'):
        file_rel_path = dataset.get('file')
        full_path = os.path.join(base_dir, file_rel_path)
        vtu_files.append(full_path)
        
    return vtu_files


def dirichlet_bcs(domain, space, disp_value_up, disp_value_down, tol=1e-6):
    """
    Applique les vecteurs de déplacement imposés sur les bords haut et bas.

    Parameters
    ----------
    domain          : dolfinx.mesh.Mesh
    space           : fem.FunctionSpace (espace vectoriel V)
    disp_value_up   : array-like 3D - [ux, uy, uz] pour le bord supérieur
    disp_value_down : array-like 3D - [ux, uy, uz] pour le bord inférieur
    tol             : float - tolérance géométrique
    """
    fdim = domain.topology.dim - 1
    bs = space.dofmap.index_map_bs  # 2 en 2D, 3 en 3D

    # Conversion en numpy array et tronquage selon la dimension du problème (bs)
    vec_up   = np.asarray(disp_value_up, dtype=PETSc.ScalarType)[:bs]
    vec_down = np.asarray(disp_value_down, dtype=PETSc.ScalarType)[:bs]

    # Auto-détection des bornes géométriques selon l'axe Y (index 1)
    y_coords = domain.geometry.x[:, 1]
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Localisation des facettes
    down_facets = dolfinx.mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] <= (y_min + tol)
    )
    up_facets = dolfinx.mesh.locate_entities_boundary(
        domain, fdim, lambda x: x[1] >= (y_max - tol)
    )

    # Création des objets fem.Constant
    c_down = fem.Constant(domain, vec_down)
    c_up   = fem.Constant(domain, vec_up)

    bc_down = fem.dirichletbc(
        c_down,
        fem.locate_dofs_topological(space, fdim, down_facets),
        space
    )
    bc_up = fem.dirichletbc(
        c_up,
        fem.locate_dofs_topological(space, fdim, up_facets),
        space
    )
    return [bc_down, bc_up]


def run_simulation_bc_vtu_fast(domain, V, W, WT, config=None, coord=1, model=None):

    cfg = {
        "t_start": 0.0,
        "T": 3.0,
        "num_steps": 50,
        "bc_tol": 1e-6,
        # Vecteurs 3D de vitesse/déplacement par unité de temps
        "disp_value_up":   [0.0,  0.01, 0.0],
        "disp_value_down": [0.0, -0.01, 0.0],
        **(config or {})
    }

    t = cfg["t_start"]
    num_steps = cfg["num_steps"]
    dt = (cfg["T"] - t) / num_steps
    bc_tol = cfg.get("bc_tol", 1e-6)

    # Conversion des vecteurs 3D de base
    base_up   = np.array(cfg["disp_value_up"], dtype=PETSc.ScalarType)
    base_down = np.array(cfg["disp_value_down"], dtype=PETSc.ScalarType)

    if model is None:
        elastic = ElasticModel(E=cfg["E"], nu=cfg["nu"], tdim=domain.topology.dim)
        model = J2IsotropicHardening(
            elastic, sigma_Y=cfg["sigma_Y"], Q_var=cfg["Q_var"], k=cfg["k_hardening"]
        )

    state = model.create_state(domain, W, WT)

    # ---------------------------------------- BCs initiales (servent à instancier le solver)
    bcs = dirichlet_bcs(
        domain, V, 
        disp_value_up=base_up * t, 
        disp_value_down=base_down * t, 
        tol=bc_tol
    )

    uh, problem, solver = build_solver(domain, V, model, state, bcs)
    ds = build_right_facet_tag(domain, coord)

    # ----------------- initialisation PyVista MultiBlock -----------------
    cells, types, x = dolfinx.plot.vtk_mesh(V)
    displ_multiblock = pv.MultiBlock()

    # -------------------------------------------- Boucle temporelle ------
    force_vec = []

    opts = PETSc.Options()
    opts["ksp_monitor"]  = None
    opts["snes_monitor"] = None
    log.set_log_level(log.LogLevel.ERROR)

    for step in range(num_steps + 1):
        t += dt
        current_disp_up   = base_up * t
        current_disp_down = base_down * t

        # 2. Application directe des CLs mises à jour dès le step 0
        problem.bcs = dirichlet_bcs(
            domain, V, 
            disp_value_up=current_disp_up, 
            disp_value_down=current_disp_down, 
            tol=bc_tol
        )

        solver.solve(uh)

        # ---- Post-traitement ----
        eps = model.elastic.epsilon(uh)
        delta_p, delta_eps_p = model.update(state, eps)

        # ---- Création du bloc PyVista ----
        step_grid = pv.UnstructuredGrid(cells, types, x)
        
        bs = V.dofmap.index_map_bs
        d_values = uh.x.array.copy().reshape((-1, bs))
        
        if bs < 3:
            d_values_3d = np.zeros((d_values.shape[0], 3))
            d_values_3d[:, :bs] = d_values
        else:
            d_values_3d = d_values
            
        step_grid.point_data["displacement"] = d_values_3d
        displ_multiblock.append(step_grid)
        displ_multiblock.set_block_name(step, f"step_{step:02d}")

        # ---- Calcul de la force de réaction ----
        stress = model.elastic.sigma(eps - (delta_eps_p + state.eps_p_old))
        force  = fem.assemble_scalar(fem.form(stress[1, 1] * ds(1)))
        force_vec.append(force)

        model.commit(state, uh)

    print("p_max =", np.max(state.p_old.x.array), " | p_mean =", np.mean(state.p_old.x.array))

    return force_vec, displ_multiblock




def animer_deformee(multiblock, factor=10.0, component=None, fps=20):
    all_scalars = []
    for block in multiblock:
        disp = block.point_data["displacement"]
        if component is None:
            val = np.linalg.norm(disp, axis=1)
        else:
            val = disp[:, component]
        all_scalars.append(val)

    clim = [min(np.min(v) for v in all_scalars), max(np.max(v) for v in all_scalars)]

    current_mesh = multiblock[0].warp_by_vector("displacement", factor=factor)
    current_mesh["disp_plot"] = all_scalars[0]

    plotter = pv.Plotter()
    title_str = "Deplacement Magnitude (mm)" if component is None else f"Deplacement U_{component}"
    
    plotter.add_mesh(
        current_mesh,
        scalars="disp_plot",
        clim=clim,
        cmap="jet",
        scalar_bar_args={"title": title_str}
    )
    plotter.add_axes()
    plotter.show(interactive_update=True)

    print("Lancement de l'animation... (Faites Ctrl+C dans le terminal pour figer l'image)")
    step = 0
    delai = 1.0 / fps
    
    try:
        while True:
            deformed_step = multiblock[step].warp_by_vector("displacement", factor=factor)
            current_mesh.points = deformed_step.points
            current_mesh["disp_plot"] = all_scalars[step]
            plotter.add_text(f"Pas de temps : {step:02d}", name="step_text", position="upper_left", font_size=12)
            plotter.update()
            sleep(delai)
            
            step += 1
            if step >= len(multiblock):
                step = 0
                
    except KeyboardInterrupt:
        print("\nAnimation interrompue.")

    print("Fenêtre finale active pour manipulation statique.")
    plotter.show()


if __name__ == "__main__":
    from time import time

    # ---------------------------------------------------------
    # Configuration générale de la simulation
    # ---------------------------------------------------------
    config = dict(
        t_start           = 0.0,
        T                 = 3.0,
        num_steps         = 50,
        length            = 10.0,
        output_dir        = "results_plasticity",
        file_name         = "import_bcs",
        # Propriétés matériau
        E                 = 200_000.0,
        nu                = 0.3,
        sigma_Y           = 100.0,
        Q_var             = 50.0,
        k_hardening       = 1000.0,
        # Vecteurs 3D [ux, uy, uz] servant de base (multipliés par t)
        disp_value_up     = [0.0,  0.01, 0.0],  # Vitesse bord supérieur [mm/s]
        disp_value_down   = [0.0, -0.01, 0.0],  # Vitesse bord inférieur [mm/s]
        bc_tol            = 1e-6,
        # Fichiers d'entrée
        pvd_file_path     = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected.pvd",
        vtu_function_name = "displacement_projected"
    )
    
    start_time = time()

    # Charger le maillage depuis le premier fichier VTU
    vtu_file = "MAINTEST/pyvista_exports/csv_projection/dic_series_projected_0000.vtu"

    domain = load_domain_from_vtu(vtu_file)
    print("Domaine chargé avec succès !")
    print(f"Nombre de cellules : {domain.topology.index_map(domain.topology.dim).size_global}")

    # Construction des espaces fonctionnels
    V, W, WT = build_function_spaces(domain)

    # Instanciation du modèle élastoplastique J2
    model = J2IsotropicHardening(
        elastic=ElasticModel(config["E"], config["nu"], tdim=domain.topology.dim),
        sigma_Y=config["sigma_Y"],
        Q_var=config["Q_var"],
        k=config["k_hardening"],
    )

    # Lancement de la simulation (récupération de la force et des blocs PyVista)
    force_vec, multiblock = run_simulation_bc_vtu_fast(
        domain, V, W, WT, 
        config=config, 
        model=model
    )

    end_time = time()
    print(f"Simulation terminée en {end_time - start_time:.2f} secondes.")
    print(f"Nombre de pas générés dans le MultiBlock : {len(multiblock)}")