"""
A wrapper for DELTA Fast Orbit Feedback responses (non-standard tool, work-in-progress)
Note: For this to work, you need to have access to special response matrix files of the DELTA storage ring.
If you are interested in general input/output for another accelerator, check model_generator.py and the manual.
"""
from numpy import loadtxt, asarray, empty
from os import makedirs

from cobea.model import Response
from cobea import cobea, read_elemnames
from delta import drift_info
import cobea.plotting as plt

input_prefix = 'delta_fofb_input/'


def import_response(filename='20_10_2017/fofb_rpm.txt'):
    corr_labels = loadtxt(input_prefix + 'mag_list.csv', delimiter=',', usecols=(3, 4), dtype=str)

    channel = asarray(corr_labels[:, 1], dtype=int)-1
    corr_labels = [c.lower() for c in corr_labels[:, 0]]

    rsp_mat_raw = loadtxt(input_prefix+filename).T

    J = int(rsp_mat_raw.shape[1]/2)
    response_matrix = empty((len(corr_labels), J, 2), dtype=float)
    response_matrix[:,:,0] = rsp_mat_raw[channel, :J]
    response_matrix[:,:,1] = rsp_mat_raw[channel, J:]

    response = Response(response_matrix, corr_labels, ['BPM%02i' % j for j in range(1, 55)],
                        read_elemnames(input_prefix+'line.text'), corr_filters=('shk*', 'svk*'))
    response.pop_monitor('BPM12')
    return response


if __name__=='__main__':
    save_path = 'delta_fofb_output/'
    try:
        makedirs(save_path)
    except FileExistsError:
        pass

    response = import_response()

    result = cobea(response, drift_space=drift_info('u250'))
    result.save(save_path+'result.pickle')

    plt.plot_result(result, prefix=save_path)
