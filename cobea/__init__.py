"""
Closed-Orbit Bilinear-Exponential Analysis (COBEA)

This is a Python implementation of the COBEA algorithm [1] to be used for studying betatron oscillations in particle
accelerators by closed-orbit information.

[1] B. Riemann. ''The Bilinear-Exponential Model and its Application to Storage Ring Beam Diagnostics'',
    PhD Dissertation (TU Dortmund University, 2016),
    DOI Link: (https://dx.doi.org/10.17877/DE290R-17221)

Bernard Riemann, TU Dortmund University,
bernard.riemann@tu-dortmund.de
"""
from numpy import asarray, sum, empty, sign, sqrt, NaN

from time import time  # for benchmarks
from pickle import load

from .mcs import find_indices, topo_indices
from .mcs import layer as startvalue_layer

# from scipy.optimize import minimize
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

from .pproc import invariant_tunes
from model import Response, Result


def read_elemnames(finame):
    """
    A helper function to read element names from text files into a list of strings. Standard input is a text file with linebreaks between elements.
    """
    fi = open(finame)
    elemnames = [line.split()[0]
                 for line in fi.readlines() if len(line.split()) != 0]
    fi.close()
    return elemnames


def optimization_layer(rslt, iprint=-1):
    """
    Implementation of the Optimization layer.
    It uses L-BFGS [1] as special case of L-BFGS-B [2] in scipy.optimize

    [1] D.C.~Liu and J.~Nocedal, ``On the Limited Memory Method for Large Scale Optimization'', Math. Prog. B \textbf{45} (3), pp.~503--528, 1989. DOI 10.1007/BF01589116

	[2] C.~Zhu, R.H.~Byrd and J.~Nocedal, ```Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization'', ACM Trans. Math. Software \textbf{23} (4), pp.~550--560, 1997. DOI 10.1145/279232.279236

    Parameters
    ----------
    rslt : object
        A valid :py:class:`cobea.model.Result` object. The object is modified during processing; the model variables are set to their optimal values.

    Returns
    -------
    additional_rslt : dict
        Additional information from the obtimization procedure. The function value and optimum are contained directly in the aforementioned rslt object that acted as input.
    """
    x = rslt._opt_wrap()
    print('Opt> search space dimensions: %i. Running...' % rslt.ndim)
    xopt, fval, additional_rslt = fmin_l_bfgs_b(
        rslt._gradient, x, args=(rslt.matrix,), iprint=iprint, maxiter=2e4, factr=100)
    print('Opt> Finished with %i gradient (L-BFGS) iterations' % additional_rslt['nit'])
    print('        chi^2 = %.3e (final)' % fval)
    rslt._opt_unwrap(xopt)
    return additional_rslt


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


def cobea(response, drift_space=NaN, convergence_info=False):
    """
    Main COBEA function with pre- and postprocessing.

    Parameters
    ----------
    response :  object
        A valid :py:class:`cobea.model.Response` object representing the input.
    drift_space : tuple
        if not-NaN, a tuple with 3 elements (monitor name 1, monitor name 2, drift space length / m)
    convergence_info : bool
        if True, convergence information from L-BFGS is added to the result dictionary (before saving).

    Returns
    -------
    result : object
        A :py:class:`cobea.model.Result` object.
    """

    # for benchmarking purposes, time for core computations is measured, displayed, saved in rslt.additional['coretime']
    tim = -time()
    # pre-allocate the Result object (no true results, just arrays initialized by numpy.empty),
    # and already set its input matrix and topology
    rslt = Result(response)
    # run the start value layer. rslt is modified
    monidx, corridx, splitidx, pcaDevs, Sg = startvalue_layer(rslt, locruns=-1)

    # run the optimization layer. rslt is modified. additional_rslt is the optimizer output (not the Result object)
    additional_rslt = optimization_layer(rslt, -1 + 2 * convergence_info)
    # measure core computation time
    tim += time()
    print('elapsed time (MCS+Opt): %.2f s' % tim)

    # add optimizer output to rslt.additional dictionary
    rslt.additional.update( additional_rslt )
    # normalize dispersion functions and coefficients by a somewhat arbitrary measure
    #bskal = sqrt(sum(rslt.b_k ** 2) / rslt.b_k.size) * sign(sum(rslt.b_k))
    #rslt.b_k /= bskal
    #rslt.d_jw *= bskal

    rslt.additional.update( {'coretime': tim, 'pca_orbits': pcaDevs, 'pca_singvals': Sg} )

    # read in convergence information if it exists
    if convergence_info:
        rslt.additional['conv'] = l_bfgs_iterate()


    chisq, grad = rslt._gradient_unwrapped(response.matrix, 0)
    #print("error check 1: %e" % chisq)

    try:  # assume that drift_space information was given
        print('PPr> normalizing using drift ')
        print('       %s -- %s' % drift_space[:2])
        print('       with length %s m.' % drift_space[2])
        driftidx = asarray(topo_indices(drift_space[:2], response.topology.line))
        di = find_indices(driftidx, monidx)
        invariant_tunes(rslt, di, drift_space[2])
    except TypeError:
        # no drift_space info, get the optimum splitidx value from MCS startvalue_layer and check invariant sign with it
        si = find_indices(splitidx[0], monidx)
        invariant_tunes(rslt, si)

    print('PPr> computing fit errors')
    rslt.update_errors()
    return rslt


def load_result(savefile):
    """
    Load (un-pickle) a Result object (or any other object)
    """
    with open(savefile) as f:
        rslt = load(f)
    return rslt
