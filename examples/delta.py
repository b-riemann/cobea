"""
a wrapper to run standard DELTA response matrix files through cobea.
Note: For this to work, you need to have access to response matrix files of the DELTA storage ring,
as well as its topology file HK_VK_BPM.Dat
If you are interested in general input/output for another accelerator, check model_generator.py and the manual.

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from cobea import read_elemnames, cobea, Response, load_result
import cobea.plotting as plt
from numpy import max, arange, asarray, nanmean, zeros, NaN, pi
from os import makedirs
import sys


def _read_orbitbrace(st, magfactor=-1, normalized=True):
    """read a line of orbit deviations from a DELTA response file"""
    if magfactor == -1:
        startl = st.find('}') + 1
        endl = st.find('}', startl)
        magfactor = asarray(st[startl:endl].split(), dtype=float)
        #print('magfactor %s' % magfactor)
        startl = endl + 1
    else:
        startl = 20
    startl = st.find('{{', startl) + 2
    endl = st.find('}', startl)
    orbit = asarray(st[startl:endl].split(), dtype=float)
    if not normalized:
        return orbit / magfactor, magfactor
    else:
        return orbit, magfactor


def _responsefile_data(finame, normalized):
    """get 'unprocessed' output from a DELTA response file"""
    with open(finame) as f:
        st = f.readline()
        bpm_numbers = arange(1, 55)  # can be overwritten by the file settings
        orb_one = list()
        orb_two = list()
        corr_names = list()
        while not st == '':
            if st[:2] == 'vk' or st[:2] == 'hk':
                corr_names.append(st[:4].upper())
                st = f.readline()
                orb_one.append(list())
                orb_two.append(list())
                while st[:3] == '   ':
                    orb, mag_factor = _read_orbitbrace(st, normalized=normalized)
                    orb_one[-1].append(orb)
                    st = f.readline()
                    orb, mag_factor = _read_orbitbrace(st, mag_factor, normalized)
                    orb_two[-1].append(orb)
                    st = f.readline()
            elif st[:4] == 'tune':
                tbt_tunes = asarray(st[6:-2].split(), dtype=float)
                st = f.readline()
            elif st[:4] == 'bpms':
                startl = st.find('{{') + 2
                endl = st.find('}', startl)
                bpm_numbers = asarray(st[startl:endl].split(), dtype=int)
                st = f.readline()
            else:
                st = f.readline()
    return orb_one, orb_two, corr_names, bpm_numbers, tbt_tunes


def _reorder(orb_one, orb_two, corrnames):
    Al = max([len(orb_one[n]) for n in range(len(orb_one))])  # 4
    Orb = zeros((Al, len(corrnames), len(orb_one[0][0]), 2))
    Orb[:] = NaN  # Orb.shape = (currents, K, J, M)
    for k in range(len(corrnames)):
        if corrnames[k][:2] == 'HK':
            for cur in range(len(orb_one[k])):
                Orb[cur, k, :, 0] = orb_one[k][cur]
                Orb[cur, k, :, 1] = orb_two[k][cur]
        elif corrnames[k][:2] == 'VK':
            for cur in range(len(orb_one[k])):
                Orb[cur, k, :, 1] = orb_one[k][cur]
                Orb[cur, k, :, 0] = orb_two[k][cur]
        else:
            print(
                'response file> corrector name %s cannot be recognized.' %
                corrnames[k])
        if not len(orb_one[k]) == Al:
            print(
                'response file> corrector %s (index %i) uses less perturbations.' %
                (corrnames[k], k))
            # unused elements stay filled with NaNs
    return Orb


def import_response(filename, normalized=True, remove_monitors=('BPM12',), line_file='delta_input/line.text'):
    """
    Convert a DELTA response file into a cobea Result object.
    """
    orb_one, orb_two, corr_names, bpms, pulser_tunes = _responsefile_data(filename, normalized)
    orb = _reorder(orb_one, orb_two, corr_names)
    r_kjw = nanmean(orb, axis=0)
    response = Response(r_kjw, corr_names, ['BPM%02i' % j for j in bpms],
                        read_elemnames(line_file), include_dispersion=True, unit='m/rad')
    for monitor in remove_monitors:
        response.pop_monitor(monitor)
    return response, pulser_tunes


def export_response_matrix(infile, r_kjw, outfile='output/polished.response'):
    with open(infile, 'r') as f:
        with open(outfile, 'w') as g:
            k = 0
            line = f.readline()
            while line != '':
                g.write(line)
                if line[:2] == 'hk' or line[:2] == 'vk':
                    if line[0] == 'h':
                        dr = [0, 1]
                    else:
                        dr = [1, 0]
                    line = f.readline()
                    while line[0] == ' ' or line[0] == '\n':
                        if line[0] == ' ':
                            startl = line.find('{{', 20) + 2
                            endl = line.find('}', startl)
                            star = line[:startl]
                            for d in dr:
                                g.write(star)
                                for j in range(r_kjw.shape[1]):
                                    if r_kjw[k, j, d] < 0:
                                        g.write('  %.3e' % r_kjw[k, j, d])
                                    else:
                                        g.write('   %.3e' % r_kjw[k, j, d])
                                g.write(line[endl:])
                                star = ' ' * (len(star) - 2) + '{{'
                            for d in range(len(dr) - 1):
                                f.readline()
                        else:
                            g.write(line)
                        line = f.readline()
                    k += 1
                else:
                    line = f.readline()


def drift_info(drift):
    """return DELTA-specific drift space information"""
    if drift == 'u250':
        # = D25 + DS in simulation/del14_02.lte
        return 'BPM14', 'BPM15', 5.12947125731 + 0.088
    elif drift == 'cope':
        # = DS + D71 in simulation/del14_02.lte
        return 'BPM38', 'BPM39', 0.088 + 0.779798407729


if __name__ == '__main__':
    recompute = True
    printresults = True
    record_convergence = False
    if len(sys.argv) > 1:
        filestr = sys.argv[1]
        if len(sys.argv) > 2:
            if sys.argv[2] == 're':
                recompute = False
            elif sys.argv[2] == 'noprint':
                printresults = False
    else:
        filestr = 'delta_input/response.100708-1'

    machine = 'delta'
    drift = 'u250'
    save_path = 'delta_output/%s/' % filestr.split('/')[-1]

    try:
        makedirs(save_path)
    except FileExistsError:
        pass

    response, tbt_tunes = import_response(filestr)
    response.save(save_path + 'response_input.pickle')
    tbt_data_in_file = {'name': 'TbT', 'tunes': tbt_tunes}

    result_filename = save_path + 'result.pickle'
    if recompute:
        result = cobea(response,
                       drift_space=drift_info(drift),
                       convergence_info=record_convergence)
        result.save(result_filename)
    else:
        result = load_result(result_filename)

    if printresults:
        fig = plt.prepare_figure(plot_type=3)
        plt.plot_result(result, prefix=save_path, comparison_data=tbt_data_in_file)
        print('Tunes: ')
        for m, modtune in enumerate(tbt_data_in_file['tunes']):
            print('TbT: %.5f, cobea: %.5f $\pm$ %.2e' %
                  (modtune, result.tune(m), result.error.mu_m[m] / (2 * pi)))
