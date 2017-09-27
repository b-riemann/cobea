"""
Closed-Orbit Bilinear-Exponential Analysis (COBEA)

This is a Python implementation of the COBEA algorithm [1] to be used for studying betatron oscillations in particle
accelerators by closed-orbit information.

[1] B. Riemann. ''The Bilinear-Exponential Model and its Application to Storage Ring Beam Diagnostics'',
    PhD Dissertation (TU Dortmund University, 2016),
    DOI Link: (https://dx.doi.org/10.17877/DE290R-17221)

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from numpy import asarray, sum, empty, sign, sqrt, NaN

from time import time  # for benchmarks
from pickle import load

from .mcs import find_indices
from .mcs import layer as startvalue_layer

from scipy.optimize.lbfgsb import fmin_l_bfgs_b  # minimize

from .pproc import invariant_tunes
from .model import Response, Result


def read_elemnames(finame):
    """
    A helper function to read element names from text files into a list of strings. Standard input is a text file with linebreaks between elements.
    """
    with open(finame) as fi:
        elemnames = [line.split()[0] for line in fi.readlines()
                     if len(line.split()) != 0]
    return elemnames


def optimization_layer(result, iprint=-1):
    """
    Implementation of the Optimization layer. It uses L-BFGS [1] as special case of L-BFGS-B [2] in scipy.optimize.
    The result object is modified to yield the optimal BEModel.
    A sub-dictionary with additional information is added under the key result.additional['Opt'].

    [1] D.C. Liu and J. Nocedal, ``On the Limited Memory Method for Large Scale Optimization'', Math. Prog. B 45 (3), pp.~503--528, 1989. DOI 10.1007/BF01589116

    [2] C. Zhu, R.H. Byrd and J. Nocedal, ``Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization'', ACM Trans. Math. Software 23 (4), pp.~550--560, 1997. DOI 10.1145/279232.279236

    Parameters
    ----------
    result : object
        A valid :py:class:`cobea.model.Result` object. The object is modified during processing; the model variables are set to their optimal values.

    Returns
    -------
    result : object
        Identical to input object.
    """
    x = result._opt_wrap()
    print('Opt> search space dimensions: %i. Running...' % result.ndim)
    xopt, fval, optimizer_dict = fmin_l_bfgs_b(
        result._gradient, x, args=(result.matrix,), iprint=iprint, maxiter=2e4, factr=100)
    print('Opt> Finished with %i gradient (L-BFGS) iterations' % optimizer_dict['nit'])
    print('       chi^2 = %.3e (final)' % fval)
    result._opt_unwrap(xopt)
    result.additional['Opt'] = optimizer_dict
    return result


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

    # for benchmarking purposes, time for core computations is measured,
    coretime = -time() # start measuring ('tic')

    # run the start value layer, return result object:
    result = startvalue_layer(response)

    # run the optimization layer, result is modified:
    optimization_layer(result, -1 + 2 * convergence_info)

    coretime += time() # stop measuring ('toc')
    print('elapsed time (MCS+Opt): %.2f s' % coretime)
    result.additional['coretime'] = coretime

    # read in convergence information if it exists
    if convergence_info:
        result.additional['conv'] = l_bfgs_iterate()

    #chisq, grad = result._gradient_unwrapped(response.matrix, 0)
    #print("error check 1: %e" % chisq)

    try:  # assume that drift_space information was given
        print('PPr> normalizing using drift ')
        print('       %s -- %s' % drift_space[:2])
        print('       with length %s m.' % drift_space[2])
        di = find_indices(drift_space[:2],result.topology.mon_names)
        invariant_tunes(result, di, drift_space[2])
    except TypeError:
        # no drift_space info, get the optimum splitidx value
        # from MCS startvalue_layer and check invariant sign with it
        inv_monitors = result.topology.mon_names[result.additional['MCS']['splitidx'][0]]
        print('PPr> no drift space given,\n     using %s -- %s to check tune quadrant' % tuple(inv_monitors))
        invariant_tunes(result,
            find_indices(inv_monitors, result.topology.mon_names))

    print('PPr> computing fit errors')
    result.update_errors()
    print(result)
    return result


def load_result(savefile):
    """
    Load (un-pickle) a Result object (or any other object)
    """
    with open(savefile,'rb') as f:
        result = load(f)
    return result
