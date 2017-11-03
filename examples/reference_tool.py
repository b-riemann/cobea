"""
A tool to compare randomly generated COBEA results of different runs and versions.
Note: Does not work well for M=2 from model_generator yet, but does so for real responses
(possibly not enough contraints in model_generator for this case).

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from os import makedirs
from model_generator import random_response_drift
from pickle import load
from numpy.testing import assert_array_almost_equal_nulp

from cobea.model import DriftSpace

reference_dict = {'set1': (30,32,1,0.02)} # (K, J, M, relative_noise)


def compare_attribute(old_result, new_result, attribute, nulp=1):
    arr = [getattr(result,attribute) for result in (new_result, old_result)]
    try:
        assert_array_almost_equal_nulp(arr[0], arr[1], nulp=nulp)
    except AssertionError:
        raise AssertionError('%s is different: computation %s vs reference %s' % (attribute,
                                                                                  arr[0].repr(), arr[1].repr))


def compare_results(new_result, old_result):
    for attribute in ('R_jmw', 'A_km', 'mu_m', 'd_jw', 'b_k'):
        # nulp*spacing(1) ~ 2.22e-16
        compare_attribute(old_result, new_result, attribute, nulp=1)
        # the deviation for error is higher as its exact values are very sensitive,
        # but a relative error of errors of ~ 1e-8 is still OK.
        compare_attribute(old_result.error, new_result.error, attribute, nulp=8)

    print('present result is approx. equivalent to reference.')

    for key in ('coretime',):
        try:
            print('%s ratio: %.2f' % (key, new_result.additional[key]/old_result.additional[key]))
        except KeyError:
            pass


def load_response(response_filename, drift_space=None):
    with open(response_filename,'rb') as f:
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
        overwrite = False

        for reference_name in reference_dict:
            print('-------- %s --------' % reference_name)

            try:
                makedirs(reference_name)
            except OSError:
                pass # directory already exists


            response_filename = reference_name+'/response_input.pickle'
            result_filename = reference_name+'/cobea_result.pickle'

            if overwrite:
                K, J, M, relative_noise = reference_dict[reference_name]
                question = 'x'
                while question not in ('Y', 'N'):
                    question = input('really overwrite %s? (please type Y or N)' % reference_name)
                if question == 'Y':
                    response, drift_space = random_response_drift(K, J, M, make_drift=False,
                                                                  hidden_filename=reference_name+'/hidden_model.pickle')
                    response.save(response_filename)
                else:
                    continue

                result = cobea(response)

                result.save(result_filename)
                print('%s: new reference set saved / overwritten' % reference_name)

            else:
                compare_computation_to_reference(response_filename, result_filename)
