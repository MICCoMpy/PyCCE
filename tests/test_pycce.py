import pytest
import numpy as np
import pycce as pc


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("pulses", [1, 2])
@pytest.mark.parametrize("method", ["cce", "gcce"])
@pytest.mark.parametrize("nbstates", [None, 2])
def test_cce_gcce(order, pulses, method, nbstates):
    # Bath spin array
    bath = pc.random_bath("13C", 100, 1000, seed=1)

    # Define simulator object
    calc = pc.Simulator(
        1,
        bath=bath,
        order=order,
        r_bath=50,
        r_dipole=6,
        pulses=pulses,
        magnetic_field=0,
        alpha=[0, 1, 0],
        beta=[1, 0, 0],
        D=1e7,
    )

    ts = np.linspace(0, 0.2, 11)

    # Compute coherence
    test = np.abs(calc.compute(ts, method=method, nbstates=nbstates))
    np.savetxt(f"ref_order{order}_pulses{pulses}_{method}_nbstates{nbstates}.txt", test)
    ref = np.loadtxt(f"ref_order{order}_pulses{pulses}_{method}_nbstates{nbstates}.txt", dtype=np.complex128)

    assert np.allclose(test, ref)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("pulses", [1, 2])
@pytest.mark.parametrize("method", ["mecce"])
@pytest.mark.parametrize("nbstates", [None, 2])
def test_mecce(order, pulses, method, nbstates):
    # Bath spin array
    bath = pc.random_bath("e", [5e3, 5e3, 5e3], density=1e16, density_units="cm-3", seed=2)  # Density in 1/cm^3

    # Center array
    center = pc.CenterArray(spin=1 / 2, alpha=[1, 0], beta=[0, 1])

    # Define simulator object with linbladian map
    calc = pc.Simulator(
        center, bath=bath, order=order, r_bath=1.4e3, r_dipole=0.7e3, pulses=pulses, magnetic_field=300, n_clusters=None
    )
    calc.bath.linmap = pc.LindbladianMap()

    # Add one dissipative channel for each bath spin pair
    et1 = 0.5  # in ms
    decay_rate = 1 / et1 / 2  # in rad / ms
    calc.bath[0].so["p"] = np.sqrt(decay_rate)
    calc.bath[0].so["m"] = np.sqrt(decay_rate)

    ts = np.linspace(0, 2, 11)

    # Compute coherence
    test = np.abs(calc.compute(ts, method=method, nbstates=nbstates))
    np.savetxt(f"ref_order{order}_pulses{pulses}_{method}_nbstates{nbstates}.txt", test)
    ref = np.loadtxt(f"ref_order{order}_pulses{pulses}_{method}_nbstates{nbstates}.txt", dtype=np.complex128)

    assert np.allclose(test, ref)
