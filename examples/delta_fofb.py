"""
A wrapper for DELTA Fast Orbit Feedback responses (non-standard tool, work-in-progress)
Note: For this to work, you need to have access to special response matrix files of the DELTA storage ring.
If you are interested in general input/output for another accelerator, check model_generator.py and the manual.
"""
from numpy import loadtxt, asarray, empty
from os import makedirs

from cobea.model import Response
from cobea import cobea, read_elemnames
from delta import drift_info, makedir_if_nonexistent
import cobea.plotting as plt

input_prefix = 'delta_fofb_input/'


def generate_corr_labels(mag_list='delta_fofb_input/mag_list.csv', usecols=(3, 4)):
    corr_labels = loadtxt(mag_list, delimiter=',', usecols=usecols, dtype=str)
    channel = asarray(corr_labels[:, 1], dtype=int)-1
    corr_labels = [c.lower() for c in corr_labels[:, 0]]
    return channel, corr_labels


def import_response(filename='20_10_2017/fofb_rpm.txt'):
    channel, corr_labels = generate_corr_labels()

    rsp_mat_raw = loadtxt(input_prefix+filename).T

    J = int(rsp_mat_raw.shape[1]/2)
    response_matrix = empty((len(corr_labels), J, 2), dtype=float)
    response_matrix[:,:,0] = rsp_mat_raw[channel, :J]
    response_matrix[:,:,1] = rsp_mat_raw[channel, J:]

    response = Response(response_matrix, corr_labels, ['BPM%02i' % j for j in range(1, 55)],
                        read_elemnames(input_prefix+'line.text'), unit='mm/A',
                        corr_filters=('shk*', 'svk*'), drift_space=drift_info('u250'),
                        name=filename)
    response.pop_monitor('BPM12')
    return response


if __name__=='__main__':
    save_path = 'delta_fofb_output/'
    makedir_if_nonexistent(save_path)

    response = import_response()

    result = cobea(response)
    result.save(save_path+'result.pickle')

    plt.plot_result(result, prefix=save_path)
