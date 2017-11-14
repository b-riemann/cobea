from timeit import default_timer as timer
from cobea import cobea, startvalue_layer, optimization_layer, pproc_layer
from reference_tool import load_response
from numpy import zeros, sum


def cobea_timing(response, trials):
    start = timer()
    for n in range(trials):
        result = cobea(response)
    end = timer()
    return end - start


def layer_timing(response, trials):
    time_vec = zeros(3, dtype=float)
    for n in range(trials):
        start = timer()
        result = startvalue_layer(response)
        end = timer()
        time_vec[0] += end - start
        optimization_layer(result)
        end = timer()
        time_vec[1] += end - start
        start = timer()
        pproc_layer(result)
        end = timer()
        time_vec[2] += end - start
    return time_vec


if __name__=='__main__':
    from sys import argv
    response = load_response(argv[1])
    trials = 5 if len(argv) < 3 else int(argv[2])
    print('>> starting %s trials' % trials)
    # exec_time = cobea_timing(response, trials)
    # print('elapsed time\n  %d trials: %.1f s\n  average: %.2f s' % (trials, exec_time, exec_time/trials))
    time_vec = layer_timing(response, trials)
    print('>> finished %s trials' % trials)
    print('total (sec): start> %02.2f optim> %02.2f pproc> %02.2f' % tuple(time_vec))
    print('average (sec): start> %02.2f optim> %02.2f pproc> %02.2f' % tuple(time_vec/trials))
    print('fraction (percent): start> %.1f optim> %.1f pproc> %.1f' % tuple(100*time_vec/sum(time_vec)))
