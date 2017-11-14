"""
A command-line tool to compare results from previous runs with computed results.
Takes two command-line arguments: pickled response and pickled result.
Note: Does not work well for M=2 from examples/model_generator yet, but does so for real responses
(probably not enough constraints in model_generator for this case).

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from pickle import load
from numpy import any, abs, max

from cobea.model import DriftSpace


def compare_attribute(old_result, new_result, attribute):
    arr = [getattr(result,attribute) for result in (new_result, old_result)]
    arr_error = [getattr(result.error,attribute) for result in (new_result, old_result)]
    aba = abs(arr[1]-arr[0])
    print('  max abs %s: %.3e' % (attribute, max(aba)))
    if any(aba > 0.1*arr_error[1]):
        raise AssertionError('deviation larger than 0.1*reference error margin')
    aerr = abs(arr_error[0]/arr_error[1] - 1)
    print('  max err rel %s: %.3e' % (attribute, max(aerr)))
    if any(aerr > 0.05):
        raise AssertionError('error margin deviates more than 5% from reference')


def compare_results(new_result, old_result):
    for attribute in ('R_jmw', 'A_km', 'mu_m', 'd_jw', 'b_k'):
        # nulp*spacing(1) ~ 2.22e-16
        compare_attribute(old_result, new_result, attribute)
        # the deviation for error is higher as its exact values are very sensitive,
        # but a relative error of errors of ~ 1e-8 is still OK.
        # compare_attribute(old_result.error, new_result.error, attribute, decimal=error_decimal)

    print('present result is approx. equivalent to reference.')

    for key in ('coretime',):
        try:
            print('%s ratio: %.2f' % (key, new_result.additional[key]/old_result.additional[key]))
        except KeyError:
            pass


def load_response(response_filename, drift_space=None):
    with open(response_filename, 'rb') as f:
        response = load(f)
        try:
            ke = response.known_element
        except AttributeError:
            response.known_element = None if drift_space is None else DriftSpace(drift_space[:2], drift_space[2])
    return response


def compare_computation_to_reference(response_filename, result_filename, make_reference=False):
    response = load_response(response_filename)

    result = cobea(response)

    if make_reference:
        result.save(result_filename)
    else:
        with open(result_filename,'rb') as f:
            reference_result = load(f)
        compare_results(result, reference_result)


if __name__=='__main__':
    from cobea import cobea
    from sys import argv

    if len(argv) > 2:
        try:
            compare_computation_to_reference(argv[1], argv[2], argv[3] == 'ref')
        except IndexError:
            compare_computation_to_reference(argv[1], argv[2])

    else:
        raise AttributeError('not enough input arguments')
