# A VIC-2D/3D Digital Image Correlation & Finite Element Model Updating Framework

This framework bridges experimental full-field measurements and numerical simulations to solve inverse problems in solid mechanics. It processes displacement from VIC-2D/3D and forces values and uses an optimization loop to identify material properties.

The current PDE solver backend relies on **FEniCSx**, but the core pipeline is built with a modular abstraction layer to support any Finite Element (FE) solver that can be implemented via python.

You will find everything you need in the [Wiki](https://github.com/paulinmarechal2103/dic_femu_paulinmarechal/wiki) !

## Prerequisites & Installation

> **Platform Note:** This installation setup is explicitly tailored for a Linux environment.

Use Conda (or Mamba if Conda is crashing) to manage your virtual environment.

### Environment setup
Using the `environement.yml` file, execute the following command to create a dedicated conda environment named `femu_env` containing all required packages :

```bash
conda env create -f environnement.yml
```
