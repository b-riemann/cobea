from numpy import abs
from numpy.random import randint, randn, uniform
from cobea.model import BEModel, Topology, Response
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
                          make_drift=True, hidden_file='hidden_model.pickle'):
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

    with open(hidden_file, 'wb') as f:
        dump(hidden_model, f, protocol=2)
        print('hidden model saved to ' + hidden_file)

    return Response(measured_matrix, topology.corr_names, topology.mon_names,
                    topology.line, include_dispersion, unit='a.u.'), drift


if __name__ == '__main__':
    from cobea import cobea, normalize_using_drift
    import cobea.plotting as plt

    response, drift_space = random_response_drift(30, 32)

    result = cobea(response, drift_space)

    # plot results in comparison with hidden model
    with open('hidden_model.pickle', 'rb') as f:
        hidden_model = load(f)
    hidden_data = {'name': 'hidden', 'beta_jmw': hidden_model.cbeta_jmw,
                   'delphi_jmw': hidden_model.delphi_jmw, 'd_jw': hidden_model.d_jw,
                   'b_k': hidden_model.b_k, 'beta_km': hidden_model.cbeta_km,
                   'delphi_km': hidden_model.delphi_km}

    save_path = 'model_generator_output/'
    try:
        makedirs(save_path)
    except OSError:
        pass  # directory already exists

    plt.plot_result(result, prefix=save_path, comparison_data=hidden_data)