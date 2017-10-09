"""
A tool to compare randomly generated COBEA results of different runs and versions.
Note: Does not work well for M=2 from model_generator yet, but does so for real responses
(possibly not enough contraints in model_generator for this case).

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from os import makedirs
from model_generator import *
from pickle import load
from numpy.testing import assert_array_almost_equal_nulp

reference_dict = {'set1': (30,32,1,0.02)} # (K, J, M, relative_noise)


def compare_results(new_result, old_result):
    for attribute in ('R_jmw', 'A_km', 'mu_m', 'd_jw', 'b_k'):
        arr = [getattr(result,attribute) for result in (new_result, old_result)]
        # try:
        assert_array_almost_equal_nulp(arr[0], arr[1])
        # except AssertionError:
        #     print('%s is different: computation %s vs reference %s' % (attribute, arr[0].repr(), arr[1].repr))
    print('present result is approx. equivalent to reference.')

    for key in ('coretime',):
        try:
            print('%s ratio: %.2f' % (key, new_result.additional[key]/old_result.additional[key]))
        except KeyError:
            pass


def compare_computation_to_reference(response_filename, result_filename, make_reference=False):

    with open(response_filename,'rb') as f:
         response = load(f)

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
