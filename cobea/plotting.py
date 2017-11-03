"""
routines for plotting cobea results

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from matplotlib.pyplot import *
from numpy import sqrt, pi, nanmax, abs, NaN, any, squeeze, arange, ones, angle, exp, linspace
from matplotlib.patches import Rectangle

# colors from standard matplotlib color cycle
Re_clr = '#1f77b4'
Im_clr = '#ff7f0e'
model_clr = '#2ca02c'

# din short and long sizes in inches - never use inches...
din_a4_l = 11.69
din_a4_s = din_a5_l = 8.27
din_a5_s = 5.83

# size, (left, bottom, right, top, wspace, hspace)
figure_adjust = {'A4': ((din_a4_s, din_a4_l), (0.1,  0.07, 0.95, 0.95, None, 0.05)),
                 'A5': ((din_a5_s, din_a5_l), (0.15, 0.10, 0.95, 0.95, None, 0.05))}


def print_close(savename='', fig=gcf()):
    fig.savefig(savename, transparent=True)
    close(fig)
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
    fig, ax = _xstrlabel_subplots(1, len(topology.mon_names))
    plot_matrix((topology.S_jk > 0).T, devlbl=r'$S_{kj}$ > 0', ax=ax)
    matrix_axes(topology, ax=ax, corr_filter=None)
    return fig


def plot_response(response, w=0, label='deviation', ax=gca(), corr_filter='all'):
    """Plot response matrix into an axis for a given direction w (0: x, 1: y)"""
    k_mask = matrix_axes(response.topology, ax=ax, corr_filter=corr_filter)
    plot_matrix(response.input_matrix[k_mask, :, w], devlbl=label+' / '+response.unit, ax=ax)


def plot_residual(result, w=0, label='residual', ax=gca(), corr_filter='all'):
    """plot fit residual into an axis for a given direction w (0: x, 1: y)"""
    k_mask = matrix_axes(result.topology, ax=ax, corr_filter=corr_filter)
    chi_kjw = result.input_matrix[k_mask, :, w] - result.response_matrix()[k_mask, :, w]
    plot_matrix(chi_kjw, devlbl=label+' / ' + result.unit, ax=ax)
    return chi_kjw


def plot_Dev_err(result, w=0, corr_filter='all'):
    """
    create a figure that shows response matrix and residual error
    for a given direction w (0: x, 1: y)
    """

    mat_fig, mat_ax = _xstrlabel_subplots(2, result.J, sharex=True, sharey=True)
    plot_response(result, w, ax=mat_ax[0], corr_filter=corr_filter)
    setp(mat_ax[0].get_xticklabels(), visible=False)
    chi_kjw = plot_residual(result, w, ax=mat_ax[1], corr_filter=corr_filter)

    hist_fig, hist_ax = _subplots(1, 1, 'A5')
    hist_ax.hist(chi_kjw.flatten(), 40)
    hist_ax.set_xlabel('deviations / ' + result.unit)
    hist_ax.set_ylabel('counts')

    return mat_fig, hist_fig


def _plot_boxes(ax, val, err=None, label='', color=Re_clr, xval=None, marker='.', sigma_scale=3):
    if xval is None:
        xval=arange(len(val))
    if err is not None:
        #ax.fill_between(xval, val - sigma_scale*err, val + sigma_scale*err, label=label,
        #                facecolor=color, linewidth=0, alpha=0.4)
        for n, xv in enumerate(xval):
            er = sigma_scale*err[n]
            ax.add_patch(Rectangle((xv-0.5, val[n] - er), 1, 2*er, facecolor=color, alpha=0.4))
    ax.plot(xval, val, marker=marker, color=color, linewidth=0.3, label=label+r' (%d$\sigma$)' % sigma_scale)


def _plot_boxes_complex(ax, val, err=None, estr='', xval=None, markers=('.', '.'), sigma_scale=3):
    if xval is None:
        xval=arange(len(val))
    _plot_boxes(ax, val.real, err.real if err is not None else None, xval=xval,
                color=Re_clr, label='Re'+estr, marker=markers[0], sigma_scale=sigma_scale)
    _plot_boxes(ax, val.imag, err.imag if err is not None else None, xval=xval,
                color=Im_clr, label='Im'+estr, marker=markers[1], sigma_scale=sigma_scale)


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
    _plot_boxes_complex(ax, val, err)

    if 'invariants' in result.additional:  # normalized data
        ax.set_ylabel(r'$\hat R_{j%i%c}$ / $\sqrt{\mathrm{m}}$' % (m, direction[w]))
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

    _plot_boxes(ax, result.cbeta_jmw[:, m, w], result.error.cbeta_jmw[:, m, w], label='cobea')
    if 'beta_jmw' in comparison_data:
        comp_beta = cm['beta_jmw'][:, m, w] if result.M > 1 else cm['beta_jmw'][:, m]
        ax.plot(comp_beta, marker='.', color=model_clr,
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


def delphi_jmw(result, m, w=None, comparison_data={}, direction='xy', ax=gca()):
    """
    plot phase-advance per monitor (incl. errors) into an axis for a given mode m
    """
    if w is None:
        w = m
    inter_monitor = arange(result.J) - 0.5

    _plot_boxes(ax, result.delphi_jmw[:, m, w], result.error.delphi_jmw[:, m, w], label='cobea', xval=inter_monitor[1:])

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
        ax.plot(inter_monitor[1:], delphi_m, marker='.', color=model_clr, linewidth=0.5,
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
    _plot_boxes(ax, result.d_jw[:, w], result.error.d_jw[:, w], label='cobea $d_{j%c}$' % direction[w])

    if 'd_jw' in comparison_data:
        cm = comp_minimal(comparison_data)
        ax.plot(arange(result.J), cm['d_jw'][:, w] if cm['d_jw'].ndim > 1 else cm['d_jw'],
                marker='.', color=model_clr,
                label=cm['name'] + ' $D_%c$' % direction[w], linewidth=0.5)
    # legend(ncol=2, loc=4, bbox_to_anchor=(1, 0.93))
    _draw_zero(ax)
    ax.set_ylabel('m')


def _subplots(n_vert, n_horz, key, **kwargs):
    fig, ax = subplots(n_vert, n_horz, **kwargs)
    try:
        [axi.grid(True, linestyle='--') for axi in ax]
    except TypeError:
        ax.grid(True)
    fig_size, adjust_params = figure_adjust[key]
    fig.set_size_inches(fig_size)
    fig.subplots_adjust(*adjust_params)
    return fig, ax


def _xstrlabel_subplots(n_fig, x_elems, sharex=True, **kwargs):
    return _subplots(n_fig, 1, 'A4' if x_elems > 35 else 'A5', sharex=sharex, **kwargs)


def info_table(result, m, direction=None, desc=None, loc='top', ax=gca(), **kwargs):
    cell_text = (('cobea v%s' % result.version, result.name),
                 ('%smode %d%s' % (desc+', ' if desc is not None else '', m,
                                   '' if direction is None else ', direction %s' % direction),
                  r'tune $Q_%d$ = %.4f $\pm$ %.4f (3$\sigma$)' % (m, result.tune(m), 3*result.error.tune(m))))
    ax.table(cellText=cell_text, loc=loc, **kwargs)


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

    info_table(result, m, direction[w], desc='monitors', ax=ax[0])

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
    _plot_boxes_complex(ax, result.A_km[k_mask, m], result.error.A_km[k_mask, m])
    ax.set_ylabel('$A_{km}$ / a.u.')


def cbeta_km(result, m, comparison_data={}, ax=gca(), filter='all'):
    """
    plot const*beta at correctors assuming
    decoupled optics and thin correctors
    ToDo: errors for this quantity
    """
    k_mask = result.topology.corr_masks[filter]
    _plot_boxes(ax, result.cbeta_km[k_mask, m], None, label='cobea')
    if 'beta_km' in comparison_data:
        cm = comp_minimal(comparison_data)
        ax.plot(cm['beta_km'][k_mask, m], marker='.',
                color=model_clr, label=cm['name'])
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
            _plot_boxes(ax, result.b_k[k_mask, w], result.error.b_k[k_mask, w], label='cobea ' + direction[w])
            if 'b_k' in cm:
                ax.plot(cm['b_k'][k_mask, w], marker='.', color=model_clr, label=cm['name'] + ' ' + direction[w])
                ax.legend()
        ax.set_ylabel(r'$b_{kw}$ / a.u.')
    else:
        _plot_boxes(ax, result.b_k[k_mask], result.error.b_k[k_mask], label='cobea')
        if 'b_k' in cm:
            ax.plot(cm['b_k'][k_mask], marker='.', color=model_clr, label=cm['name'])
            ax.legend()
        ax.set_ylabel(r'$b_k$ / a.u.')
    _draw_zero(ax)


def corrector_results(result, m=0, comparison_data={}, direction='xy', filter='all'):
    """create a figure with corrector results for a given mode m"""
    fig, ax = _xstrlabel_subplots(4 if result.include_dispersion else 3, result.K)

    info_table(result, m, desc='correctors', ax=ax[0])

    A_km(result, m, ax=ax[0], filter=filter)
    ax[0].legend(ncol=2)

    cbeta_km(result, m, comparison_data, ax=ax[1], filter=filter)

    # delphi_km(result, m, comparison_data, ax=ax[2], filter=filter)

    if result.include_dispersion:
        b_k(result, m, comparison_data, direction, ax=ax[3], filter=filter)

    corrector_label(result.topology.corr_names[result.topology.corr_masks[filter]], dir='x', ax=ax[-1])
    return fig


def plot_result(result, prefix='', comparison_data={}, direction='xy', plot_flags='mcdtvs'):
    """
    plot cobea results.

    Parameters
    ----------
    result : object
        A :py:class:`cobea.model.Result` object.
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
    if 'm' in plot_flags:
        for m in range(result.M):
            for w in range(result.M):
                fig = monitor_results(result, m, w, comparison_data, direction)
                print_close(prefix + 'monitor_m%i_%s.pdf' % (m, direction[w]), fig)

    if 'c' in plot_flags:
        for m in range(result.M):
            for filter in result.topology.corr_masks:
                fig = corrector_results(result, m, comparison_data, direction, filter)
                print_close(prefix + 'corrector_m%i_%s.pdf' % (m, filter.replace('*', '')), fig)

    if 'd' in plot_flags:
        for w in range(result.M):
            for filter in result.topology.corr_masks:
                mat_fig, hist_fig = plot_Dev_err(result, w, corr_filter=filter)
                filter_str = filter.replace('*', '')
                print_close(prefix + 'Dev_err_%s_%s.pdf' % (direction[w], filter_str), mat_fig)
                print_close(prefix + 'hist_%s_%s.pdf' % (direction[w], filter_str), hist_fig)

    if 't' in plot_flags:
        fig = plot_topology(result.topology)
        print_close(prefix + 'topology.pdf', fig)

    if 'v' in plot_flags and 'conv' in result.additional:  # convergence information is available
        fig, ax = _subplots(1, 1, 'A5')
        ax.semilogy(result.additional['conv']['it'], result.additional['conv']['f'])
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel('residual squared error $\chi^2$')
        ax.set_xlim((0, result.additional['conv']['it'][-1]))
        print_close(prefix + 'convergence.pdf', fig)

    if 's' in plot_flags and result.known_element is not None:
        pass
        # delta_s = linspace(0, result.known_element.length, 256)
        # drift_js = result.topology.monitor_index(result.known_element.mon_names)
        # R_ends = result.R_jmw[ drift_js ]
        # R_ends_err = result.error.R_jmw[ drift_js ]
        # for m in range(result.M):
        #     for w in range(result.M):
        #         fig, ax = _subplots(3, 1, 'A5')
        #         R_s, R_s_err = result.known_element.inside_tracking(R_ends[:, m, w], delta_s,
        #                                                             rj_drift_err=R_ends_err[:, m, w])
        #         beta_s = R_s.real**2 + R_s.imag**2
        #         beta_s_err = 2*(R_s.real*R_s_err.real + R_s.imag*R_s_err.imag)
        #         phi_deg_s = angle( R_s*R_ends[0, m, w].conj() ) * 180 / pi
        #         _plot_boxes_complex(ax[0], R_s, err=R_s_err, xval=delta_s, markers=(None, None))
        #         ax[0].set_ylabel(r'$\hat R_{%i%c}(s)$ / $\sqrt{\mathrm{m}}$' % (m, direction[w]))
        #         _plot_boxes(ax[1], beta_s, err=beta_s_err, xval=delta_s, marker=None)
        #         ax[1].set_ylabel(r'$\beta_{%i%c}(s)$ / m' % (m, direction[w]))
        #         ax[-1].set_xlabel('distance from upstream monitor / m')
        #         _plot_boxes(ax[2], phi_deg_s, err=None, xval=delta_s, marker=None)
        #         ax[2].set_ylabel(r'$\phi_{%i%c}(s)$ / deg' % (m, direction[w]))
        #         print_close(prefix + 'known_element_m%i_%s.pdf' % (m, direction[w]), fig)
