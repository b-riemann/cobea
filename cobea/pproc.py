"""
Small postprocessing functions for COBEA results.
"""

from numpy import conj, empty, zeros, eye, dot

def symplectic_form(D=2):
    """
    Compute the symplectic form.
    
    Parameters
    ----------
    D : int
        number of spatial dimensions of the phase space vectors considered.

    Returns
    -------
    Omega : array
        a matrix that can be used to compute invariants I from phase space eigenvectors Q via (* matrix product)
        I = Q.T * Omega * Q
    """
    Omega = zeros([2 * D, 2 * D])
    Omega[:D, D:] = eye(D)
    Omega[D:, :D] = -eye(D)
    return Omega


def phasor_eigenvectors(R_drift, length):
    """
    Compute phase space vector from the spatial vectors around a drif t space

    Parameters
    ----------
    R_drift : array
        An array of two spatial vectors R_drift[0] and R_drift[1]
    length : float
        length of the drift space

    Returns
    -------
    Q : array
        Phase space vector
    """
    D = R_drift.shape[2]
    Q = empty([R_drift.shape[1], 2 * D], dtype=R_drift.dtype)
    Q[:, :D] = R_drift[0]
    Q[:, D:] = (R_drift[1] - R_drift[0]) / length
    return Q


def invariants_from_eigenvectors(Q):
    """
    Compute invariant of motion from a phase space eigenvector

    Parameters
    ----------
    Q : array
        phase space eigenvector with shape (1,2*M)

    Returns
    -------
    invariant : float
        invariants of motion
    """
    Om = symplectic_form(Q.shape[1] / 2)  # (R_drift.shape[0])
    M = Q.shape[0]  # R_drift.shape[1]
    invariant = empty(M, dtype=complex)
    for m in xrange(M):
        invariant[m] = -1.j * dot(conj(Q[m].T), dot(Om, Q[m])) / 2
    print('     invariants: ' + str(invariant))
    return invariant.real


def invariants_of_motion(R_drift, length):
    """
    Compute invariant of motion from the eigenorbits around a known drift space

    Parameters
    ----------
    R_drift : array
        An array of two spatial vectors R_drift[0] and R_drift[1]
    length : float
        length of the drift space

    Returns
    -------
    invariant : float
        invariants of motion
    """
    Q = phasor_eigenvectors(R_drift, length)
    return invariants_from_eigenvectors(Q)


def invariant_tunes(rslt, di, driftlen=0):
    """
    Invariant postprocessing algorithm.
    The Result object is modified by information from a drift space. monitor vectors, corrector parameters and the sign of mu_m is changed accordingly.
    
    Parameters
    ----------
    rslt : object
        A valid :py:class:`cobea.model.Result` object. The object is modified.
    di : list
        j indices of the used drift space
    driftlen : float
        length of the use drift space (usually in m)
    """
    if driftlen != 0:
        Im = invariants_of_motion(rslt.R_jmw[di], driftlen)
        rslt.normalize(Im)
        rslt.additional['invariants'] = Im
    else:
        print('guessing quadrant')
        inprod = invariants_of_motion(rslt.R_jmw[di], 1)
        rslt.normalize(inprod)
