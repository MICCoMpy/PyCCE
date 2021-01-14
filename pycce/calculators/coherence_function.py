import numpy as np

from pycce.cluster_expansion import cluster_expansion_decorator
from pycce.hamiltonian import projected_hamiltonian


def propagators(timespace, H0, H1, N):
    """
    Function to compute propagators U0 and U1 in conventional CCE
    @param timespace: ndarray
        Time delay values at which to compute propagators
    @param H0: ndarray
        Hamiltonian projected on alpha qubit state
    @param H1: ndarray
        Hamiltonian projected on beta qubit state
    @param N: int
        number of pulses in CPMG
    @return: U0, U1
    """
    eval0, evec0 = np.linalg.eigh(H0)
    eval1, evec1 = np.linalg.eigh(H1)

    eigen_exp0 = np.exp(-1j * np.tensordot(timespace,
                                           eval0, axes=0), dtype=np.complex128)
    eigen_exp1 = np.exp(-1j * np.tensordot(timespace,
                                           eval1, axes=0), dtype=np.complex128)

    v0 = np.matmul(np.einsum('ij,kj->kij', evec0, eigen_exp0,
                             dtype=np.complex128),
                   evec0.conj().T, dtype=np.complex128)

    v1 = np.matmul(np.einsum('ij,kj->kij', evec1, eigen_exp1,
                             dtype=np.complex128),
                   evec1.conj().T, dtype=np.complex128)

    if N == 0:
        return v0, v1

    V0_HE = np.matmul(v0, v1, dtype=np.complex128)
    V1_HE = np.matmul(v1, v0, dtype=np.complex128)

    if N == 1:
        return V0_HE, V1_HE

    V0 = np.matmul(V0_HE, V1_HE, dtype=np.complex128)  # v0 @ v1 @ v1 @ v0
    V1 = np.matmul(V1_HE, V0_HE, dtype=np.complex128)  # v1 @ v0 @ v0 @ v1

    U0 = np.linalg.matrix_power(V0, N // 2)
    U1 = np.linalg.matrix_power(V1, N // 2)

    if N % 2 == 1:
        U0 = np.linalg.matmul(U0, V0_HE)
        U1 = np.linalg.matmul(U1, V1_HE)

    return U0, U1


def computeL(H0, H1, timespace, N, as_delay=False):
    """
    Function to compute cluster L in conventional CCE
    @param H0: ndarray
        Hamiltonian projected on alpha qubit state
    @param H1: ndarray
        Hamiltonian projected on beta qubit state
    @param timespace: ndarray
        Time points at which to compute L
    @param N: int
        number of pulses in CPMG
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @return: L
    """
    # if timespace was given not as delay between pulses,
    # divide to obtain the delay
    if not as_delay and N > 0:
        timespace = timespace / (2 * N)

    U0, U1 = propagators(timespace, H0, H1, N)

    # L = np.trace(np.matmul(U0, np.transpose(
    #     U1.conj(), axes=(0, 2, 1))), axis1=1, axis2=2) / U0.shape[1]
    # L is computed as Tr[rho U0 U1dagger]; rho = Identity / dim
    L = np.einsum('zij,zij->z', U0, U1.conj()) / U0.shape[1]
    return L


@cluster_expansion_decorator
def decorated_coherence_function(cluster, allspin, projections_alpha, projections_beta, B, timespace, N,
                                 as_delay=False):
    """
        Overarching decorated function to compute L in conventional CCE. The call of the function includes:
    @param subclusters: dict
        dict of subclusters included in different CCE order
        of structure {int order: np.array([[i,j],[i,j]])}
    @param allnspin: ndarray
        array of all atoms
    @param projections_alpha: ndarray
        ndarray containing projections of alpha state [<Sx>, <Sy>, <Sz>]
    @param projections_beta: ndarray
        ndarray containing projections of beta state [<Sx>, <Sy>, <Sz>]
    @param B: ndarray
        Magnetic field of B = np.array([Bx, By, Bz])
    @param timespace: ndarray
        Time points at which to compute L
    @param N: int
        number of pulses in CPMG
    @param as_delay: bool
        True if time points are delay between pulses,
        False if time points are total time
    @return: L
        L computed with conventional CCE
    """
    nspin = allspin[cluster]
    H0, H1 = projected_hamiltonian(nspin, projections_alpha, projections_beta, B)
    L = computeL(H0, H1, timespace, N, as_delay=as_delay)
    return L