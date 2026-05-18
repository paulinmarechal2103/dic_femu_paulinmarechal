A Python framework for VIC2D-3D Digital Image Correlation (DIC) for mechanical parameters identification using Finite Element Model Updating (FEMU), combining FEniCSx for PDE solving and Scikit-Optimize for Bayesian optimization. 

### WIP

## Prerequisites & Installation

> ⚠️ **Note:** This installation setup is explicitly tailored and tested for a **Linux framework**.

Because of the complex scientific dependencies (MPI, PETSc, FEniCSx), it is highly recommended to use **Conda** (or **Mamba** for faster dependency resolution) to manage the environment.

### 1. Create the Environment

Run the following command to create a dedicated environment named `femu_env` with all the required packages from the `conda-forge` channel:

```bash
conda create -n femu_env -c conda-forge python=3.11 \
  fenics-dolfinx=0.10.0 mpich petsc h5py numpy scipy matplotlib meshio pyvista scikit-optimize
