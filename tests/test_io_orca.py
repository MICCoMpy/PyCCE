import os

import numpy as np
import pytest

import pycce as pc
from pycce.constants import MHZ_TO_KHZ

HERE = os.path.dirname(os.path.abspath(__file__))

# Trace/3 of the reference matrix, i.e. ORCA's printed A(iso) = 466.6282 MHz.
EXPECTED_AISO_MHZ = (444.4082 + 532.9815 + 422.4948) / 3


@pytest.mark.parametrize("fixture", ["orca5_hfc.out", "orca6_hfc.out"])
def test_read_orca_parses_hyperfine(fixture):
    """Hyperfine is read from both the ORCA 5 and ORCA 6 spellings."""
    atoms = pc.read_orca(os.path.join(HERE, fixture), find_isotopes=False)

    assert np.any(atoms.A), f"no hyperfine tensor parsed from {fixture}"

    aiso = np.trace(atoms.A[0]) / 3
    assert aiso == pytest.approx(EXPECTED_AISO_MHZ * MHZ_TO_KHZ, rel=1e-9)


def test_orca5_and_orca6_agree():
    """The same data under either header must parse identically."""
    v5 = pc.read_orca(os.path.join(HERE, "orca5_hfc.out"), find_isotopes=False)
    v6 = pc.read_orca(os.path.join(HERE, "orca6_hfc.out"), find_isotopes=False)

    np.testing.assert_allclose(v5.A, v6.A)
    np.testing.assert_allclose(v5.xyz, v6.xyz)
