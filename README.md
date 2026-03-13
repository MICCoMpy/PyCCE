![image](docs/source/logo.png)

## **PyCCE** code repository

Welcome to the repository, containing the source code of **PyCCE** - a Python library for computing
qubit dynamics in the central spin model with the cluster-correlation expansion (CCE) method.

Installation
------------

Run `pip install .` in the main folder.

Base Units
----------

* Gyromagnetic ratios are given in rad / ms / G.
* Magnetic field in G.
* Timesteps in ms.
* Distances in A.
* All coupling constants are given in kHz.

Usage
-----

1. Prepare spin bath `BathArray` from `BathCell`.
2. Run calculations with `Simulator`.

See the `examples` folder for tutorials and scripts of calculations.

Documentation
-------------

The tutorial and documentation are hosted on [GitHub Pages](https://MICCoMpy.github.io/PyCCE/).

See also [aiida-pycce](https://github.com/MICCoMpy/aiida-pycce), a plugin for running PyCCE in a high-throughput manner using the AiiDA framework.

Contact
-------

Please use the GitHub [issue tracker](https://github.com/MICCoMpy/PyCCE/issues/) for bug reports. Contributions to new features are welcome.
