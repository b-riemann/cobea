"""
routines for plotting cobea results

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from matplotlib.pyplot import *
from numpy import sqrt, pi, nanmax, abs, unwrap, NaN, where, any, squeeze, arange, ones, angle, exp

## Axes and Colors ##

# Colors:
    # http://colorbrewer2.org/, number of classes: 3 (or 4), nature: qualitative, only show: colorblind safe, print friendly
    # result: "4-class paired", colors: a6cee3, 1f78b4, b2df8a, (33a02c)
Re_clr = '#33a02c'  # 'mediumturquoise'
Re_clr2 = '#b2df8a'  # 'mediumseagreen'
Im_clr = '#a0332c'  # 'mediumturquoise', interchanged red and green
Im_clr2 = '#dfb28a'  # 'mediumseagreen', interchanged red and green
modelc = '#1f78b4'  # 'chocolate'


def plot_size(plot_type=0):
    """
    plot sizes for all plot types
    """
    lw = 6.6  # minimum linewidth 6.5 inches
    hw = 3.3  # half linewidth
    h2w = 1. / sqrt(2)  # DIN-like scaling of width and height
    if plot_type == 0:
        fs = (lw, h2w * lw)  # full-width plot
    elif plot_type == 1:
        fs = (hw, h2w * hw)  # half-width plot
    elif plot_type == 2:
        fs = (hw, hw)  # half-width square
    elif plot_type == 3:
        fs = (lw / h2w, lw)  # landscape full-width-plot
    elif plot_type == 4:
        fs = (lw / h2w, lw * h2w)  # landscape 0.5 height
    elif plot_type == 5:
        fs = (lw, lw / 2)  # portrait 0.5 height
        # fs=(lw,lw/2) #widescreen (1 per line)
    elif plot_type == 6:
        fs = (lw, lw / h2w)  # full-portrait
    elif plot_type == 7:
        fs = (hw / h2w, hw / h2w)  # 1/sqrt(2) width square
    else:  # plot_type==8:
        fs = (lw, lw)
    return fs


def prepare_figure(plot_type=0):
    """set fonts, tex packages, and figure size. returns figure"""
    #rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    #rcParams["text.usetex"] = True
    #rcParams["font.size"] = 11
    #rcParams["legend.fontsize"] = "medium"
    rcParams['contour.negative_linestyle'] = 'solid'
    return figure(figsize=plot_size(plot_type))


def change_figsize(fig, plot_type=0):
    fig.set_size_inches(plot_size(plot_type))


def printshow(savecond, savename='', fig=gcf()):
    fig.tight_layout()
    if not savecond:
        show()
    else:
        fig.savefig(savename, transparent=True)
        print('figure printed to ' + savename + '.')


def coleur(n=-1):
    """
    a colorset compiled of:
    - 0-5: colorbrewer2 2-class paired
    - 6-11: inverse of 0-5
    """
    clr = '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', \
          '#59311c', '#e0874b', '#4d2075', '#cc5fd3', '#046566', '#1ce5e3'
    if n > 11:
        return 'k'
    elif n > -1:
        return clr[n]
    else:
        return clr


def _make_ticklabels(labels, dir='x', spacing=0, ax=gca()):
    bpmj = arange(len(labels), dtype=int)
    if dir == 'x':
        if spacing == 0:
            if len(labels) > 60:
                spacing = 3
            elif len(labels) > 40:
                spacing = 2
            else:
                spacing = 1
        ax.set_xticks(bpmj[::spacing])
        ax.set_xticklabels(labels[::spacing], rotation='vertical', size='small', family='monospace')
    else:
        if spacing == 0:
            if len(labels) > 60:
                spacing = 5
            elif len(labels) > 40:
                spacing = 3
            else:
                spacing = 1
        ax.set_yticks(bpmj[::spacing])
        ax.set_yticklabels(labels[::spacing], size='small', family='monospace')


def monitor_label(mon_labels=0, spacing=0, ax=gca()):
    """apply monitor labels to an axis"""
    ax.set_xlim((-0.5, len(mon_labels) - 0.5))
    if isinstance(mon_labels, int):
        ax.set_xlabel('monitor index $j$')
    else:
        _make_ticklabels(mon_labels, 'x', spacing, ax)


def corrector_label(corr_labels=[], spacing=0, dir='y', ax=gca()):
    """apply corrector labels to an axis"""
    if len(corr_labels) == 0:
        if dir == 'x':
            ax.set_xlabel('corrector index $k$')
        else:
            ax.set_ylabel('corrector index $k$')
    else:
        _make_ticklabels(corr_labels, dir, spacing, ax)


def plot_matrix(Devdr, devlbl, cmap=('PRGn','Greens'), ax=gca()):
    """plot an arbitrary matrix with divergent or sequential colormap (helper function)"""
    # alternative cmap = ('bwr','Blues')
    maxdev = nanmax(abs(Devdr))
    if any(Devdr < 0):
        cm = cmap[0] if isinstance(cmap, tuple) else cmap
        lims=(-maxdev, maxdev)
    else:
        cm=cmap[1] if isinstance(cmap, tuple) else cmap
        lims=(0, maxdev)
    quad = ax.pcolormesh(arange(Devdr.shape[1]+1)-0.5, arange(Devdr.shape[0]+1)-0.5,
               squeeze(Devdr), vmin=lims[0], vmax=lims[1], cmap=cm)
    cb = colorbar(quad, ax=ax)
    cb.set_label(devlbl)


def matrix_axes(topology, ax=gca(), corr_filter=None):
    if corr_filter is None:
        k_mask = ones(len(topology.corr_names), dtype=bool)
    else:
        k_mask = topology.corr_masks[corr_filter]
    corrector_label(topology.corr_names[k_mask], ax=ax)
    monitor_label(topology.mon_names, ax=ax)
    return k_mask


def plot_topology(topology):
    """create a figure that shows the accelerator topology. Input: Topology object"""
    fig, ax = subplots()
    plot_matrix((topology.S_jk > 0).T, devlbl=r'$S_{kj}$ > 0', ax=ax)
    matrix_axes(topology, ax=ax, corr_filter=None)
    return fig


def plot_response(response, w=0, label='deviation', ax=gca(), corr_filter='all'):
    """Plot response matrix into an axis for a given direction w (0: x, 1: y)"""
    k_mask = matrix_axes(response.topology, ax=ax, corr_filter=corr_filter)
    plot_matrix(response.matrix[k_mask, :, w], devlbl=label+' / '+response.unit, ax=ax)


def plot_residual(result, w=0, label='residual', ax=gca(), corr_filter='all'):
    """plot fit residual into an axis for a given direction w (0: x, 1: y)"""
    k_mask = matrix_axes(result.topology, ax=ax, corr_filter=corr_filter)
    chi_kjw = result.matrix[k_mask, :, w] - result.response_matrix()[k_mask, :, w]
    plot_matrix(chi_kjw, devlbl=label+' / ' + result.unit, ax=ax)
    return chi_kjw


def plot_Dev_err(result, w=0, corr_filter='all'):
    """
    create a figure that shows response matrix and residual error
    for a given direction w (0: x, 1: y)
    """

    mat_fig, mat_ax = subplots(2, 1, sharex=True, sharey=True)
    change_figsize(mat_fig, 0)
    plot_response(result, w, ax=mat_ax[0], corr_filter=corr_filter)
    setp(mat_ax[0].get_xticklabels(), visible=False)
    chi_kjw = plot_residual(result, w, ax=mat_ax[1], corr_filter=corr_filter)

    hist_fig, hist_ax = subplots()
    change_figsize(hist_fig, 2)
    hist_ax.hist(chi_kjw.flatten(), 40)
    hist_ax.set_xlabel('deviations / ' + result.unit)
    hist_ax.set_ylabel('counts')

    return mat_fig, hist_fig


def _plot_Re(ax, val, err, label='', color=Re_clr, xval=[]):
    if len(xval) < 1:
        xval=arange(len(val))
    ax.fill_between(xval, val - err, val + err, label=label,
                    facecolor=color, linewidth=0, alpha=0.4)
    ax.plot(xval, val, marker='+', color='black', linewidth=0.3)


def _plot_ReIm(ax, val, err, estr='', xval=[]):
    if len(xval) < 1:
        xval=arange(len(val))
    ax.fill_between(xval, val.real - err.real, val.real + err.real,
                    facecolor=Re_clr, linewidth=0, alpha=0.4, label='Re'+estr)
    ax.fill_between(xval, val.imag - err.imag, val.imag + err.imag,
                    facecolor=Im_clr, linewidth=0, alpha=0.4, label='Im'+estr)
    ax.plot(xval, val.real, marker='+', color='black', linewidth=0.3)
    ax.plot(xval, val.imag, marker='x', color='black', linewidth=0.3)


def _draw_zero(ax):
    xl = ax.get_xlim()
    ax.plot(xl,(0,0),'k--',linewidth=0.5)


def _ground_ylim(ax):
    yl = ax.get_ylim()
    ax.set_ylim((0,yl[1]))


## monitor quantities ##


def R_jmw(result, m, w=None, direction='xy', ax=gca()):
    """plot real and imaginary parts of monitor vectors (incl. errors) into an axis for a given mode m"""
    if w is None:
        w = m
    val = result.R_jmw[:, m, w]
    err = result.error.R_jmw[:, m, w]
    _plot_ReIm(ax, val, err)

    if 'invariants' in result.additional:  # normalized data
        ax.set_ylabel(r'$\hat R_{j%i%c}$ / m' % (m, direction[w]))
    else:
        ax.set_ylabel(r'$R_{j%i%c}$ / m' % (m, direction[w]))
    _draw_zero(ax)


def comp_minimal(comparison_data):
    d = {'name': 'comparison'}
    d.update(comparison_data)
    return d


def cbeta_jmw(result, m, w=None, comparison_data={}, direction='xy', ax=gca()):
    """plot beta resp. const*beta (incl. errors) into an axis for a given mode m"""
    # plot of beta or const*beta
    if w is None:
        w = m

    if 'beta_jmw' in comparison_data:
        cm = comp_minimal(comparison_data)

    _plot_Re(ax, result.cbeta_jmw[:, m, w],
             result.error.cbeta_jmw[:, m, w], 'cobea')
    if 'beta_jmw' in comparison_data:
        comp_beta = cm['beta_jmw'][:, m, w] if result.M > 1 else cm['beta_jmw'][:, m]
        ax.plot(comp_beta, marker='.', color=modelc,
                label=cm['name'], linewidth=0.5)

    _ground_ylim(ax)

    if 'invariants' in result.additional:  # normalized data
        ax.set_ylabel(r'$\beta_{j%i%c}$ / m' % (m, direction[w]))
    else:
        ax.set_ylabel(r'$\beta_{j%i%c}$ / m' % (m, direction[w]))


def _w_expand(comparison_key, m, w):
    try:
        return comparison_key[:, m, w]
    except IndexError:
        return comparison_key[:, m]


def delphi_jmw(result, m, w=None, comparison_data={}, yl=-1, direction='xy', ax=gca()):
    """
    plot phase-advance per monitor (incl. errors) into an axis for a given mode m
    """
    if w is None:
        w = m
    bpmj = arange(result.J) - 0.5
    if yl > 0 and w == m:  # background colors counting half integer advances
        plus = yl * ones(result.J)
        for j in range(result.J):
            if result.phi_jmw[j, m, w] < 0:
                plus[j] = NaN
        ax.fill_between(bpmj, 0 * plus, plus, facecolor=Re_clr2,
                        alpha=0.4, linewidth=0)

    _plot_Re(ax, result.delphi_jmw[:, m, w], result.error.delphi_jmw[:, m, w],
             'cobea', xval=bpmj[1:])

    if 'delphi_jmw' in comparison_data:
        delphi_m = _w_expand(comparison_data['delphi_jmw'], m, w)
        comparison_available = True
    elif 'phi_jmw' in comparison_data:
        phi_mw = _w_expand(comparison_data['phi_jmw'], m, w) * pi / 180
        delphi_m = angle(exp(1.j * (phi_mw[1:] - phi_mw[:-1])), deg=True)
        comparison_available = True
    else:
        comparison_available = False

    if comparison_available:
        cm = comp_minimal(comparison_data)
        ax.plot(bpmj[1:], delphi_m, marker='.', color=modelc, linewidth=0.5,
                label=cm['name'] + ' ' + direction[w])

    ax.set_ylabel(r'$\Delta \phi_' + direction[w] + '$ / deg')
    yl = ax.get_ylim()
    if yl[1] > 180:
        ax.set_ylim((0, 180))
    else:
        ax.set_ylim((0, yl[1]))


def d_jw(result, w, comparison_data, direction='xy', ax=gca()):
    """
    plot const*dispersion (incl. errors) into an axis for a given direction w (0: x, 1: y)
    """
    _plot_Re(ax, result.d_jw[:, w],
             result.error.d_jw[:, w],
             'cobea $d_{j%c}$' % direction[w])

    if 'd_jw' in comparison_data:
        cm = comp_minimal(comparison_data)
        ax.plot(arange(result.J), cm['d_jw'][:, w] if cm['d_jw'].ndim > 1 else cm['d_jw'],
                marker='.', color=modelc,
                label=cm['name'] + ' $D_%c$' % direction[w], linewidth=0.5)
    # legend(ncol=2, loc=4, bbox_to_anchor=(1, 0.93))
    _draw_zero(ax)
    ax.set_ylabel('m')


def _xstrlabel_subplots(n_fig, x_elems):
    fig, ax = subplots(n_fig, 1, sharex=True)
    [axi.grid(True, linestyle='--') for axi in ax]
    siz = plot_size(plot_type=3 if x_elems > 35 else 6)
    if n_fig < 4:
        fig.set_size_inches((siz[0],siz[1]*0.75))
    else:
        fig.set_size_inches(siz)
    return fig, ax


def monitor_results(result, m=0, w=None, comparison_data={}, direction='xy'):
    """
    plot monitor results for mode m,
    optionally in comparison with comparison_data.

    Parameters
    ----------
    result : object
        A :py:class:`cobea.model.Result` object.
    m : int
        mode index to plot results for
    w : int
        direction index to plot results for
    comparison_data : dict
        a dictionary containing optional data from alternative decoupled storage ring models, which may contain the following keys:
        'name': name of the algorithm or model used
        'beta': an array of shape (result.M,result.J) that contains Courant-Snyder beta values for each direction and monitor
        'phi': an array of the same shape as 'beta', containing Courant-Snyder betatron phases
        'dispersion': an array of the same shape, containing dispersion values
    """
    if w is None:
        w = m
    show_dispersion = result.include_dispersion and m == w
    fig, ax = _xstrlabel_subplots(4 if show_dispersion else 3, result.J)

    #stri = 'cobea' + ' tune: %.4f' % result.tune(m)
    #stri += ' $\pm$ %.4f' % abs(result.error.mu_m[m] / (2 * pi))
    #ax1.text(0, 1.1, stri, horizontalalignment='left', transform=ax1.transAxes,
    #        color=Re_clr, fontweight='bold', size='large')
    #if 'tunes' in comparison_data:
    #    ax1.text(1, 1.1, comparison_data['name'] + ' tune: %.4f' % comparison_data['tunes'][m],
    #         horizontalalignment='right', transform=ax1.transAxes,
    #         color=modelc, fontweight='bold', size='large')

    R_jmw(result,m,w,direction=direction,ax=ax[0])
    ax[0].legend(ncol=2)

    cbeta_jmw(result,m,w,comparison_data,direction=direction,ax=ax[1])
    ax[1].legend(ncol=2)

    delphi_jmw(result,m,w,comparison_data,direction=direction,ax=ax[2])

    if show_dispersion:
        d_jw(result,w,comparison_data,direction=direction,ax=ax[3])
        ax[3].legend(ncol=2)

    monitor_label(result.topology.mon_names, ax=ax[-1])
    return fig


## corrector quantities


def A_km(result, m, ax=gca(), filter='all'):
    """plot real and imaginary parts of corrector parameters (incl. errors) into an axis for a given mode m"""
    k_mask = result.topology.corr_masks[filter]
    _plot_ReIm(ax, result.A_km[k_mask, m], result.error.A_km[k_mask, m])
    ax.set_ylabel('$A_{km}$ / a.u.')


def cbeta_km(result, m, comparison_data={}, ax=gca(), filter='all'):
    """
    plot const*beta at correctors assuming
    decoupled optics and thin correctors
    ToDo: errors for this quantity
    """
    k_mask = result.topology.corr_masks[filter]
    _plot_Re(ax, result.cbeta_km[k_mask, m], 0, label='cobea')
    if 'beta_km' in comparison_data:
        cm = comp_minimal(comparison_data)
        ax.plot(cm['beta_km'][k_mask, m], marker='.',
                color=modelc, label=cm['name'])
        ax.legend()
    ax.set_ylabel(r'$|A_{km}| \approx $ const $\beta_km$ / a.u.')


#def delphi_km(result, m, comparison_data={}, ax=gca(), filter=''):
#    bpmk = arange(result.K - 1) + 0.5
#    _plot_Re(ax, result.delphi_km[:, m], 0, xval=bpmk, label='cobea')
#    if 'delphi_km' in comparison_data:
#        cm = comp_minimal(comparison_data)
#        ax.plot(bpmk, cm['delphi_km'][:,m], marker='.',
#                color=modelc, label=cm['name'])
#        ax.legend()
#    ax.set_ylabel('$\Delta$ arg($A_{km}$) / deg')


def b_k(result, w=0, comparison_data={}, direction='xy', ax=gca(), filter='all'):
    k_mask = result.topology.corr_masks[filter]
    cm = comp_minimal(comparison_data)
    if result.b_k.ndim > 1:
        for w in range(result.b_k.shape[1]):
            _plot_Re(ax, result.b_k[k_mask, w], result.error.b_k[k_mask, w], label='cobea '+direction[w])
            if 'b_k' in cm:
                ax.plot(cm['b_k'][k_mask, w], marker='.', color=modelc, label=cm['name']+' '+direction[w])
                ax.legend()
        ax.set_ylabel(r'$b_{kw}$ / a.u.')
    else:
        _plot_Re(ax, result.b_k[k_mask], result.error.b_k[k_mask], label='cobea')
        if 'b_k' in cm:
            ax.plot(cm['b_k'][k_mask], marker='.', color=modelc, label=cm['name'])
            ax.legend()
        ax.set_ylabel(r'$b_k$ / a.u.')
    _draw_zero(ax)


def corrector_results(result, m=0, comparison_data={}, direction='xy', filter='all'):
    """create a figure with corrector results for a given mode m"""
    fig, ax = _xstrlabel_subplots(4 if result.include_dispersion else 3, result.K)

    A_km(result, m, ax=ax[0], filter=filter)
    # ax[0].legend(ncol=2)

    cbeta_km(result, m, comparison_data, ax=ax[1], filter=filter)

    # delphi_km(result, m, comparison_data, ax=ax[2], filter=filter)

    if result.include_dispersion:
        b_k(result, m, comparison_data, direction, ax=ax[3], filter=filter)

    corrector_label(result.topology.corr_names[result.topology.corr_masks[filter]], dir='x', ax=ax[-1])
    return fig


def plot_result(result, print_figures=True, prefix='', comparison_data={}, direction='xy', plot_flags='mcdtv'):
    """
    plot cobea results.

    Parameters
    ----------
    result : object
        A :py:class:`cobea.model.Result` object.
    print_figures : bool
        whether to print figures into separate pdf files instead of showing them. Default: True
    prefix : str
        if print_figures=True, prefix contains the relative path to the current folder where results are printed.
    comparison_data : dict
        a dictionary containing optional data from alternative decoupled storage ring models, which may contain the following keys:
        'name': name of the algorithm or model used
        'beta': an array of shape (result.M,result.J) that contains Courant-Snyder beta values for each direction and monitor
        'phi': an array of the same shape as 'beta', containing Courant-Snyder betatron phases
        'dispersion': an array of the same shape, containing dispersion values
    direction : str
        direction characters for the result object. can be 'x','y', or 'xy'.
    plot_flags : str
        which plots are to be created. Each character represents a different result plot:
        'm': monitor_results -> monitor_m*.pdf
        'c': corrector_results -> corrector_m*.pdf
        'd': plot_Dev_err -> Dev_err_w*.pdf, hist_w*.pdf
        't': plot_topology -> topology.pdf
        'v': convergence information -> convergence.pdf. Only works if convergence information is available.
    """
    fig = prepare_figure(plot_type=0)

    if 'm' in plot_flags:
        for m in range(result.M):
            for w in range(result.M):
                fig = monitor_results(result, m, w, comparison_data, direction)
                printshow(print_figures, prefix + 'monitor_m%i_%s.pdf' % (m, direction[w]), fig)

    if 'c' in plot_flags:
        for m in range(result.M):
            for filter in result.topology.corr_masks:
                fig = corrector_results(result, m, comparison_data, direction, filter)
                printshow(print_figures, prefix + 'corrector_m%i_%s.pdf' % (m, filter.replace('*', '')), fig)

    if 'd' in plot_flags:
        for w in range(result.M):
            for filter in result.topology.corr_masks:
                mat_fig, hist_fig = plot_Dev_err(result, w, corr_filter=filter)
                filter_str = filter.replace('*', '')
                printshow(print_figures, prefix + 'Dev_err_%s_%s.pdf' % (direction[w], filter_str), mat_fig)
                printshow(print_figures, prefix + 'hist_%s_%s.pdf' % (direction[w], filter_str), hist_fig)

    if 't' in plot_flags:
        fig = plot_topology(result.topology)
        printshow(print_figures, prefix + 'topology.pdf', fig)

    if 'v' in plot_flags and 'conv' in result.additional:  # convergence information is available
        change_figsize(fig, plot_type=2)  # 5
        # ax1=subplot(1,2,1)
        # xlabel('start-value iterations')
        # ax2=subplot(1,2,2,sharey=ax1)
        # setp( ax2.get_yticklabels(), visible=False)
        fig, ax = subplots()
        ax.semilogy(result.additional['conv']['it'], result.additional['conv']['f'])
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel('residual squared error $\chi^2$')
        ax.set_xlim((0, result.additional['conv']['it'][-1]))
        printshow(print_figures, prefix + 'convergence.pdf', fig)

    # if 's' in plot_flags and result.drift_space is not None:
        # def beta_interp(s_pos, beta, alfa, sint):
        #     # find unique s values first
        #     msk = ones(s_pos.shape[0], dtype=bool)
        #     msk[1:] = diff(s_pos) != 0
        #     xi = s_pos[msk]
        #     outvals = empty((2, len(sint)), dtype=beta.dtype)

        #     yi = empty((xi.shape[0], 2), dtype=beta.dtype)
        #     for n in range(2):
        #         yi[:, 0] = beta[n, msk]
        #         yi[:, 1] = -2 * alfa[n, msk]
        #         bp = BPoly.from_derivatives(xi, yi, orders=3)
        #         outvals[n] = bp(sint)
        #     return outvals

        # di = find_indices(result.drift_space[:2], result.topology.mon_names)
        # sint = arange(0,256.0)/256 * drift_space[2]
        # for m in range(result.M):
        #     beta = [result.beta_jmw[j,m,m] for j in di]
        #     Rpc = (result.R_jmw[di[1],m,m] - result.R_jmw[di[0],m,m]).conj() / result.drift_space[2]
        #     alfa = [-(result.R_jmw[j,m,m] * Rpc).real for j in di]
        #    out
