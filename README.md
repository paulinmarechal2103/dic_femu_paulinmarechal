# VIC-2D/3D Digital Image Correlation & Finite Element Model Updating Framework

A Python framework dedicated to VIC-2D/3D Digital Image Correlation (DIC) data exploitation for mechanical parameter identification. The pipeline leverages Finite Element Model Updating (FEMU), combining FEniCSx for PDE solving and Scikit-Optimize for Bayesian optimization.

> 🚧 **Work In Progress (WIP):** Active development is currently focused exclusively on the `plasticity/` directory. Other modules may be unstable or incomplete.

---

## Prerequisites & Installation

> ⚠️ **Platform Note:** This installation setup is explicitly tailored and tested for a Linux environment (some libraries are only working on linux).

Due to complex scientific dependencies, it is highly recommended to use Conda (or Mamba if Conda is crashing) to manage your virtual environment.

### 1. Create the Environment

Execute the following command to create a dedicated environment named `femu_env` containing all required packages from the `conda-forge` channel:

```bash
conda create -n femu_env -c conda-forge python=3.11 fenics-dolfinx=0.10.0 mpich petsc h5py numpy scipy matplotlib meshio pyvista scikit-optimize

