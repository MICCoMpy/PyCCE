import numpy as np
import pycce as pc


def test_cce_gcce():
    # Bath spin array
    bath = pc.random_bath("13C", 100, 1000, seed=1)

    # Define simulator object
    calc = pc.Simulator(
        1, bath=bath, order=2, r_bath=50, r_dipole=6, pulses=1, magnetic_field=0, alpha=[0, 1, 0], beta=[1, 0, 0], D=1e7
    )

    ts = np.linspace(0, 0.2, 11)

    # Compute coherence
    test_cce = np.abs(calc.compute(ts, method="cce"))
    ref_cce = np.loadtxt("ref_cce.txt", dtype=np.float64)

    assert np.allclose(test_cce, ref_cce)

    # Compute coherence
    test_gcce = np.abs(calc.compute(ts, method="gcce"))
    ref_gcce = np.loadtxt("ref_gcce.txt", dtype=np.float64)

    assert np.allclose(test_gcce, ref_gcce)


def test_mecce():
    # Bath spin array
    bath = pc.random_bath("e", [5e3, 5e3, 5e3], density=1e16, density_units="cm-3", seed=2)  # Density in 1/cm^3
    bath.types.add_type("e", 1.0, pc.ci["e"].gyro / 2)

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
    calc.bath.linmap = pc.LindbladianMap()

    # Define spin states
    plus_1 = np.array([1, 0, 0], dtype=np.complex128)
    zero = np.array([0, 1, 0], dtype=np.complex128)
    minus_1 = np.array([0, 0, 1], dtype=np.complex128)

    # Spin matrices
    p0_1 = np.outer(zero, plus_1)
    m1_0 = np.outer(plus_1, zero)
    p0__1 = np.outer(zero, minus_1)
    m0__1 = np.outer(minus_1, zero)

    # Strength of pair interaction
    def pos(coord_1, coord_2, r_h=200):
        pos = np.abs(coord_1) - np.abs(coord_2)
        r = np.linalg.norm(pos)
        return r / r_h

    # Define dissipator rates with four different dissipators
    et1 = 0.5  # in ms
    decay_rate = 1 / et1 / 2  # in rad / ms
    for k in range(len(calc.clusters[1])):
        i, j = calc.clusters[2][k]
        n_1 = calc.bath[i]
        n_2 = calc.bath[j]
        hop_rate = decay_rate * np.exp(-np.abs(pos(n_1["xyz"], n_2["xyz"])))
        # dis = pc.Dissipator(size=2,  left=p0_1, right=m1_0, symmetric=False)
        dis1 = pc.Dissipator(size=2, symmetric=True)
        dis2 = pc.Dissipator(size=2, symmetric=True)
        dis3 = pc.Dissipator(size=2, symmetric=True)
        dis4 = pc.Dissipator(size=2, symmetric=True)
        dis1.add_jump_operator(p0_1, 0, rate=hop_rate)
        dis1.add_jump_operator(m1_0, 1, rate=hop_rate)
        dis2.add_jump_operator(m1_0, 0, rate=hop_rate)
        dis2.add_jump_operator(p0_1, 1, rate=hop_rate)
        dis3.add_jump_operator(p0__1, 0, rate=hop_rate)
        dis3.add_jump_operator(m0__1, 1, rate=hop_rate)
        dis4.add_jump_operator(m0__1, 0, rate=hop_rate)
        dis4.add_jump_operator(p0__1, 1, rate=hop_rate)
        calc.bath.linmap.add_dissipator(k, dis1)
        calc.bath.linmap.add_dissipator(k, dis2)
        calc.bath.linmap.add_dissipator(k, dis3)
        calc.bath.linmap.add_dissipator(k, dis4)

    ts = np.linspace(0, 2, 51)

    # Compute coherence
    calc.order = 1
    test_mecce = calc.compute(ts, method="mecce")
    ref_mecce = np.loadtxt("ref_mecce.txt", dtype=np.complex128)

    assert np.allclose(test_mecce, ref_mecce)
