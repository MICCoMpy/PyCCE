import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pycce as pc

pc.utilities.config_torch()

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()
except:
    comm = None
    global_rank = 0

local_rank = pc.utilities.get_local_rank()

# Bath spin array
bath = pc.random_bath("e", [5e3, 5e3, 5e3], density=1e16, density_units="cm-3", seed=2)  # Density in 1/cm^3

# Center array
center = pc.CenterArray(spin=1 / 2, alpha=[1, 0], beta=[0, 1])

# Define simulator object with linbladian map
calc = pc.Simulator(
    center,
    bath=bath,
    order=2,
    r_bath=1.4e3,
    r_dipole=0.7e3,
    pulses=1,
    magnetic_field=300,
    n_clusters=None,
)


et1 = 0.5  # in ms
decay_rate = 1 / et1 / 2  # in rad / ms
calc.bath.add_single_jump("p", rate=decay_rate)  # p for creation operator S+
calc.bath.add_single_jump("m", rate=decay_rate)  # m for annihilation operator S-

ts = np.linspace(0, 2, 51)

# Compute coherence
calc.order = 2

if global_rank == 0:
    print(calc)

test_mecce = calc.compute(ts, method="mecce", parallel=True, device=f"cuda:{local_rank}", verbose=True)

if global_rank == 0:
    np.savetxt(f"test_mecce_{calc.order}.txt", test_mecce)
    print("completed")
