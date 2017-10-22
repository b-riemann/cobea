"""
Small postprocessing and helper functions for COBEA results.

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""

from numpy import asarray, conj, dot, empty, eye, sign, sum, zeros
from .mcs import find_indices


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
    Om = symplectic_form(int(Q.shape[1] / 2))  # symplectic_form(M)
    M = Q.shape[0]  # R_drift.shape[1]
    invariant = empty(M, dtype=complex)
    for m in range(M):
        invariant[m] = -1.j * dot(conj(Q[m].T), dot(Om, Q[m])) / 2
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


def normalize_using_drift(model, di, drift_length):
    """
    Invariant postprocessing algorithm.
    The Result object is modified by information from a drift space. monitor vectors, corrector parameters and the sign of mu_m is changed accordingly.
    
    Parameters
    ----------
    model : object
        A valid :py:class:`cobea.model.BEModel` object or descendant. The object is modified.
    di : list
        j indices of the used drift space
    drift_length : float
        length of the use drift space
    """
    invariants = invariants_of_motion(model.R_jmw[asarray(di)], drift_length)
    model.normalize(invariants)

    try: # if the object is a Result object (subclassing BEModel), this works
        model.additional['invariants'] = invariants
    except AttributeError:
        pass


def guess_mu_sign(rslt):
    """
    for weakly coupled setups, guess the sign of mu (quadrant) based on monitor phase advance
    """
    x = sign(rslt.delphi_jmw)
    for m in range(rslt.M):
        if sum(x[:,m,m]) < 0:
            rslt.flip_mu(m)


def l_bfgs_iterate(alloc_items=10000):
    """
    convert the iterate.dat file produced by L-BFGS-B

    Parameters
    ----------
    alloc_items : int
        the number of maximum iterations for which memory is allocated.

    Returns
    -------
    iter : dict
        a dictionary with the following fields. The field names and descriptions have been copied from a demo output.
          'it' : array
              iteration number
          'nf' : array
              number of function evaluations
          'nseg' : array
              number of segments explored during the Cauchy search
          'nact' : array
              number of active bounds at the generalized Cauchy point
          'sub'  : str
              manner in which the subspace minimization terminated
                  con = converged,
                  bnd = a bound was reached
          'itls' : int
              number of iterations performed in the line search
          'stepl' : float
              step length used
          'tstep' : float
              norm of the displacement (total step)
          'projg' : float
              norm of the projected gradient
          'f'    : float
              function value
    """
    narr = empty((5, alloc_items), dtype=int)
    farr = empty((5, alloc_items))
    with open('iterate.dat') as f:
        num = -1
        for line in f:
            if line[:5] == '   it':
                num = 0
            else:
                if num > -1:
                    lisp = line.split()
                    if len(lisp) != 10:
                        break
                    for c in range(4):
                        if lisp[c][0] != '-':
                            narr[c, num] = int(lisp[c])
                    if lisp[c][0] != '-':
                        narr[4, num] = int(lisp[5])
                    for c in range(4):
                        if lisp[c + 6][0] != '-':
                            farr[c, num] = float(lisp[c + 6].replace('D', 'E'))
                    num += 1

    return {
        'it': narr[0, :num],
        'nf': narr[1, :num],
        'nseg': narr[2, :num],
        'nact': narr[3, :num],
        'itls': narr[4, :num],
        'stepl': farr[0, :num],
        'tstep': farr[1, :num],
        'projg': farr[2, :num],
        'f': farr[3, :num]}


def layer(result, drift_space=None, convergence_info=False):
    """
    Postprocessing layer

    Parameters
    ----------
    result : object
        A :py:class:`cobea.model.Result` object. The object is modified during processing.
    drift_space : iterable
        if not None, a tuple or list with 3 elements (monitor name 1, monitor name 2, drift space length / m)
    convergence_info : bool
        if True, convergence information from L-BFGS is added to the result dictionary (before saving).
    """
    if convergence_info:
        # read in convergence information
        result.additional['conv'] = l_bfgs_iterate()

    try:  # assume that drift_space information was given
        di = find_indices(drift_space[:2], result.topology.mon_names)
        print(("PPr> normalizing using drift\n"
               "       %s -- %s with length ~ %.4f m.") % tuple(drift_space))
        normalize_using_drift(result, di, drift_space[2])
        print("       invariants: %s" % result.additional['invariants'])
    except TypeError: # drift_space is None
        print('PPr> no drift space given, guessing tune quadrant by phase advance sign')
        guess_mu_sign(result)

    print('PPr> computing fit errors')
    result.update_errors()
