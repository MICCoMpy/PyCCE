![image](docs/source/logo.png)

# PyCCE Code Repository

Welcome to the repository, containing the source code of **PyCCE** - a Python library for computing
qubit dynamics in the central spin model with the cluster-correlation expansion (CCE) method.

### Installation

Run `pip install .` in the main folder.

### Base Units

* Gyromagnetic ratios are given in rad / ms / G.
* Magnetic field in G.
* Timesteps in ms.
* Distances in A.
* All coupling constants are given in kHz.

### Usage

1. Prepare spin bath `BathArray` from `BathCell`.
2. Run calculations with `Simulator`.

See the `examples` folder for tutorials and scripts of calculations.

### Documentation

Full documentation is available online at [Read the Docs](https://pycce.readthedocs.io/en/latest/).

See also [aiida-pycce](https://github.com/MICCoMpy/aiida-pycce), a plugin for running PyCCE in a high-throughput manner using the AiiDA framework.
