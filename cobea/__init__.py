"""
Closed-Orbit Bilinear-Exponential Analysis (COBEA)

This is a Python implementation of the COBEA algorithm [1] to be used for studying betatron oscillations in particle
accelerators by closed-orbit information.

[1] B. Riemann. ''The Bilinear-Exponential Model and its Application to Storage Ring Beam Diagnostics'',
    PhD Dissertation (TU Dortmund University, 2016),
    DOI Link: (https://dx.doi.org/10.17877/DE290R-17221)

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from pickle import load as pickle_load
from scipy.optimize.lbfgsb import fmin_l_bfgs_b  # minimize

from .mcs import layer as startvalue_layer
from .model import Response, Result, version
from .pproc import layer as pproc_layer


def read_elemnames(filename, delimiter='\n'):
    """
    A helper function to read element names from text files into a list of strings.
    Standard input is a text file with linebreaks between elements.

    Parameters
    ----------
    filename : str
        input file name
    delimiter : str
        (Optional) which separator to use for elements,
        default is the (unix) linefeed (which also works for windows linefeeds)
    """
    with open(filename) as fi:
        contents = fi.read()
    element_names = list()
    for elem in contents.split(delimiter):
        elem = elem.strip()  # this also removes \r in case of \r\n linefeeds
        if len(elem) > 0:
            element_names.append(elem)
    return element_names


def optimization_layer(result, iprint=-1):
    """
    Implementation of the Optimization layer. It uses L-BFGS [1] as special case of L-BFGS-B [2] in scipy.optimize.
    The result object is modified to yield the optimal BEModel.
    A sub-dictionary with additional information is added under the key result.additional['Opt'].

    [1] D.C. Liu and J. Nocedal. ``On the Limited Memory Method for Large Scale Optimization'',
        Math. Prog. B 45 (3), pp.~503--528, 1989. DOI 10.1007/BF01589116

    [2] C. Zhu, R.H. Byrd and J. Nocedal, ``Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale
        bound-constrained optimization'', ACM Trans. Math. Software 23 (4), pp.~550--560, 1997.
        DOI 10.1145/279232.279236

    Parameters
    ----------
    result : object
        A valid :py:class:`cobea.model.Result` object.
        The object is modified during processing; the model variables are set to their optimal values.
    iprint : int
        (Optional) verbosity of fmin_l_bfgs_b. Default: -1

    Returns
    -------
    result : object
        Identical to input object.
    """
    x = result._to_statevec()
    print('Optimization layer: running with %i model parameters...' % result.ndim)
    xopt, fval, optimizer_dict = fmin_l_bfgs_b(
        result._gradient, x, args=(result.input_matrix,), iprint=iprint, maxiter=int(2e4), factr=100)
    print('    ...finished with %i gradient (L-BFGS) iterations.' % optimizer_dict['nit'])
    print('    chi^2 = %.3e (%s)^2' % (fval, result.unit))
    result._from_statevec(xopt)
    result.additional['Opt'] = optimizer_dict
    return result


def cobea(response, convergence_info=False):
    """
    Main COBEA function with pre- and postprocessing.

    Parameters
    ----------
    response :  object
        A valid :py:class:`cobea.model.Response` object representing the input.

    convergence_info : bool
        if True, convergence information from L-BFGS is added to the result dictionary (before saving).

    Returns
    -------
    result : object
        A :py:class:`cobea.model.Result` object.
    """
    # run the start value layer, return result object:
    result = startvalue_layer(response)

    # run the optimization layer, result is modified:
    optimization_layer(result, -1 + 2 * convergence_info)

    # run postprocessing layer, result is modified:
    pproc_layer(result, convergence_info)

    print(result)
    return result


def load_result(filename):
    """
    Load (un-pickle) a Result object (or any other object)
    """
    # if npz:
    #     npd = numpy_load(filename)
    #     result = Result(Response(npd['input_matrix'], npd['corr_names'], npd['mon_names'], list(npd['line']),
    #                       'd_jw' in result, assume_sorted=True))
    # else:
    with open(filename, 'rb') as f:
        result = pickle_load(f)
    return result
