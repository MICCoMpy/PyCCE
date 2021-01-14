import warnings

import numpy as np
from numba import jit

from ..units import BOHR_TO_ANGSTROM, HBAR, ELECTRON_GYRO

# Copied from ASE
chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'spin_matrix', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


class Cube:
    """
    Class for the processing of .cube datafiles with polarization

    Parameters
    ------------
        @param filename: str
            name of the .cube file

    """
    _dt = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])

    def __init__(self, filename):

        with open(filename, "r") as content:
            # first two lines are comments
            self.comments = next(content).strip() + "\n" + next(content).strip()

            # total number of atoms | xyz of the cube origin
            tot = next(content).split()
            natoms = int(tot[0])

            self.origin = np.array([float(x) for x in tot[1:]])
            self.voxel = np.empty([3, 3], dtype=np.float64)
            self.size = np.empty(3, dtype=np.int32)

            self.atoms = np.empty(natoms, dtype=self._dt)


            for i in range(3):
                tot = next(content).split()
                self.size[i] = int(tot[0])

                if self.size[i] < 0:
                    self.voxel[i] = [float(x) for x in tot[1:]]

                else:
                    self.voxel[i] = [float(x) * BOHR_TO_ANGSTROM for x in tot[1:]]

            for j in range(natoms):
                tot = next(content).split()

                self.atoms[j]['N'] = chemical_symbols[int(tot[0])]
                self.atoms[j]['xyz'] = [float(x) for x in tot[2:]]

            if self.size[0] > 0:
                self.atoms['xyz'] *= BOHR_TO_ANGSTROM

            data = [float(x) for line in content for x in line.split()]

        if self.size[0] < 0:
            self.data = np.array(data).reshape(np.abs(self.size))
        else:
            self.data = np.array(data).reshape(np.abs(self.size)) / (BOHR_TO_ANGSTROM ** 3)

        # check if diagonal
        # (see https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python)
        if np.any(self.voxel.reshape(-1)[:-1].reshape(2, 4)[:, 1:]):
            warnings.warn('Voxel might be non-orthogonal. Correctness of the results is not guaranteed')

        na = np.newaxis
        a = np.arange(self.size[0])[:, na] * self.voxel[0][na, :]
        b = np.arange(self.size[1])[:, na] * self.voxel[1][na, :]
        c = np.arange(self.size[2])[:, na] * self.voxel[2][na, :]
        mesh = a[:, na, na, :] + b[na, :, na, :] + c[na, na, :, :]

        self.grid = mesh + self.origin
        self.integral = np.trapz(np.trapz(np.trapz(self.data))) * np.linalg.det(self.voxel)
        self.spin = round(self.integral) * 0.5

    def transform(self, R=None, shift=None, inplace=True):
        """
        Changes coordinates of the grid. DOES NOT ASSUME PERIODICITY.
        @param R: ndarray with shape (3, 3)

            Rotation matrix
                R =  [n_1^(1) n_1^(2) n_1^(3)]
                     [n_2^(1) n_2^(2) n_2^(3)]
                     [n_3^(1) n_3^(2) n_3^(3)]

            n_i^(j) corresponds to coeff of initial basis vector i
            for j new basis vector:
            e'_j = n_1^(j)*e_1 + n_2^(j)*e_2 + n_3^(j)*e_3

            in other words, columns of R are coordinates of the new
            basis in the old basis.

            Given vector in initial basis v = [v1, v2, v3],
            vector in new basis is given as v' = R.T @ vector

        @param shift: ndarray with shape (3,)
            shift in the origin of coordinates (in the rotated basis)

        @param inplace: bool
            If False rotates a copy of grid

        @return: ndarray
            grid
        """
        grid = self.grid

        if R is not None:
            assert (np.isclose(np.linalg.det(R), 1.)), 'Determinant of R is not equal to 1'
            grid = np.einsum('ij,abcj->abci', R.T, grid)

        if shift is not None:
            grid = grid + np.asarray(shift)

            if inplace:
                self.origin += shift

        if inplace:
            self.grid = grid

        return grid

    def integrate(self, position, gyro_n, gyro_e=ELECTRON_GYRO, spin=None):
        """
        Integrate over polarization data, stored in Cube object,
        to obtain hyperfine dipolar-dipolar tensor
        @param position: ndarray with shape (3,)
            position of the nuclei at which to compute A in final coordinate system
        @param gyro_n: float
            gyromagnetic ratio of nucleus
        @param gyro_e: float
            gyromagnetic ratio of central spin
        @param spin: float
            total spin of the central spin
        @return: ndarray with shape (3, 3)
            A tensor
        """
        if spin is None:
            spin = self.spin

        if np.around(spin * 2) != np.around(self.integral):
            warnings.warn(f'provided spin: {spin:.2f} is not equal to one from spin density: {self.integral / 2:.2f}')

        position = np.asarray(position)

        if len(position.shape) > 1:
            A = cube_integrate_array(self.data, self.grid, self.voxel, spin,
                                     position, gyro_n, gyro_e)
        else:
            A = cube_integrate(self.data, self.grid, self.voxel, spin,
                               position, gyro_n, gyro_e)
        return A

        # d['A'] = -(3 * np.outer(pos, pos) - identity * r ** 2) / (r ** 5) * pre


@jit(nopython=True)
def cube_integrate(data, grid, voxel, spin, position, gyro_n, gyro_e=ELECTRON_GYRO):
    pos = grid - position

    dist = np.sqrt(np.sum(pos ** 2))

    pre = gyro_e * gyro_n * HBAR / (2 * spin) * np.linalg.det(voxel)

    A = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):

            if i == j:
                integrand = - pre * data * (3 * pos[:, :, :, i] * pos[:, :, :, j] - dist ** 2) / dist ** 5
            else:
                integrand = - pre * data * (3 * pos[:, :, :, i] * pos[:, :, :, j]) / dist ** 5

            A[i, j] = np.trapz(np.trapz(np.trapz(integrand)))

    return A


@jit(nopython=True)
def cube_integrate_array(data, grid, voxel, spin, coordinates, gyros, gyro_e=ELECTRON_GYRO):
    As = np.zeros((coordinates.shape[0], 3, 3), dtype=np.float64)
    for i, position in enumerate(coordinates):
        As[i] = cube_integrate(data, grid, voxel, spin, position, gyros[i], gyro_e)
    return As