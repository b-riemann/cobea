"""
An example script for cobea using randomly generated Response objects.

Bernard Riemann (bernard.riemann@tu-dortmund.de
"""
from numpy import abs
from numpy.random import randint, randn, uniform
from cobea.model import BEModel, Topology, Response
from cobea.pproc import normalize_using_drift
from pickle import dump, load
from os import makedirs


def random_topology(K, J):
    corr_names = ['cor%i' % k for k in range(K)]
    mon_names = ['bpm%i' % j for j in range(J)]

    # create a 'scrambled' combined line list
    tmp_line = list(corr_names)
    tmp_line.extend(mon_names)
    line = list()
    while len(tmp_line) > 0:
        idx = randint(0, len(tmp_line))
        line.append(tmp_line.pop(idx))

    return Topology(corr_names, mon_names, line)


def random_model(K, J, M, include_dispersion, max_180deg=True):
    topology = random_topology(K, J)

    if max_180deg:
        def init_fun(shape, dtype):
            if dtype == complex:
                c = abs(randn(*shape)) + 1.j * abs(randn(*shape))
                c[1::4].real = -c[1::4].real
                c[2::4] = -c[2::4]
                c[3::4].imag = -c[3::4].imag
                return c
            else:  # dtype==float
                return randn(*shape)
    else:
        def init_fun(shape, dtype):
            if dtype == complex:
                return randn(*shape) + 1.j * randn(*shape)
            else:  # dtype==float
                return randn(*shape)

    return BEModel(K, J, M, topology, include_dispersion, init_fun)


def random_response_drift(K, J, M=1, include_dispersion=True, relative_noise=0.02,
                          make_drift=True, hidden_filename=None):
    """generate a random, valid Response object.

    Parameters
    ----------
    K : int
        number of correctors
    J : int
        number of monitors
    M : int
        number of modes respectively directions
    include_dispersion : bool
        whether the response includes dispersive effects or not
    relative_noise : float
        the relative magnitude of noise added to the generated response matrix elements
    make_drift : bool
        whether to generate a valid drift space or not.
        If not, second return argument is just an empty tuple.
    hidden_filename : str
       The hidden model used for generating the Response
        is saved as a file with name hidden_filename and not
        available outside until being reloaded from it.

    Returns
    -------
    response : Response
       the response object that can be used as input for the cobea function
    drift : tuple or None
       additional drift space information if make_drift=True, empty if not.
    """
    hidden_model = random_model(K, J, M, include_dispersion)
    topology = hidden_model.topology
    if make_drift:
        drift_start = randint(0, J-1)
        drift_idx = (drift_start, drift_start+1)
        drift_length = uniform(0.1, 2.0)
        normalize_using_drift(hidden_model, drift_idx, drift_length)

        drift = [topology.mon_names[j] for j in drift_idx]
        drift.append(drift_length)
    else:
        drift = None

    measured_matrix = hidden_model.response_matrix()
    # now just make some noise...
    measured_matrix += relative_noise * randn(K, J, M)

    if hidden_filename is not None:
        with open(hidden_filename, 'wb') as f:
            dump(hidden_model, f, protocol=2)
            print('hidden model saved to ' + hidden_filename)

    return Response(measured_matrix, topology.corr_names, topology.mon_names,
                    topology.line, include_dispersion, drift_space=drift)


if __name__ == '__main__':
    # reference_dict = {'set1': (30,32,1,0.02)} # (K, J, M, relative_noise)
    from cobea import cobea
    import cobea.plotting as plt

    hidden_filename = 'hidden_model.pickle'
    result_filename = None # 'cobea_result.pickle'

    response = random_response_drift(30, 32, hidden_filename=hidden_filename)

    result = cobea(response)

    if result_filename is not None:
        result.save(result_filename)


    # plot results in comparison with hidden model
    with open(hidden_filename, 'rb') as f:
        hidden_model = load(f)
    hidden_data = {'name': 'hidden', 'beta_jmw': hidden_model.cbeta_jmw,
                   'delphi_jmw': hidden_model.delphi_jmw, 'd_jw': hidden_model.d_jw,
                   'b_k': hidden_model.b_k, 'beta_km': hidden_model.cbeta_km,
                   'delphi_km': hidden_model.delphi_km}

    pdf_path = 'model_generator_output/'
    try:
        makedirs(pdf_path)
    except OSError:
        pass  # directory already exists

    plt.plot_result(result, prefix=pdf_path, comparison_data=hidden_data)