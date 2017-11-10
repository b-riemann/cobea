from timeit import default_timer as timer
from cobea import cobea
from reference_tool import load_response


def cobea_timer(response, trials=5):
    start = timer()
    for n in range(trials):
        result = cobea(response)
    end = timer()
    return end - start


if __name__=='__main__':
    from sys import argv
    response = load_response(argv[1])
    trials = 5 if len(argv) < 3 else argv[2]
    exec_time = cobea_timer(response, trials)
    print('elapsed time\n  %d trials: %.1f s\n  average: %.2f s' % (trials, exec_time, exec_time/trials))
