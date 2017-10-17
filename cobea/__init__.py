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

from .mcs import layer as startvalue_layer

from scipy.optimize.lbfgsb import fmin_l_bfgs_b  # minimize

from .pproc import layer as pproc_layer
from .model import Response, Result


def read_elemnames(finame):
    """
    A helper function to read element names from text files into a list of strings.
    Standard input is a text file with linebreaks between elements.
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
    x = result._to_statevec()
    print('Opt> search space dimensions: %i. Running...' % result.ndim)
    xopt, fval, optimizer_dict = fmin_l_bfgs_b(
        result._gradient, x, args=(result.matrix,), iprint=iprint, maxiter=2e4, factr=100)
    print('Opt> Finished with %i gradient (L-BFGS) iterations' % optimizer_dict['nit'])
    print('       chi^2 = %.3e (final)' % fval)
    result._from_statevec(xopt)
    result.additional['Opt'] = optimizer_dict
    return result





def cobea(response, drift_space=None, convergence_info=False):
    """
    Main COBEA function with pre- and postprocessing.

    Parameters
    ----------
    response :  object
        A valid :py:class:`cobea.model.Response` object representing the input.
    drift_space : iterable
        if not None, a tuple or list with 3 elements (monitor name 1, monitor name 2, drift space length / m)
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

    pproc_layer(result, drift_space, convergence_info)
    print(result)
    return result


def load_result(savefile):
    """
    Load (un-pickle) a Result object (or any other object)
    """
    with open(savefile,'rb') as f:
        result = load(f)
    return result