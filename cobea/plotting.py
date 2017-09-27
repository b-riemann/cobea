"""
routines for plotting COBEA results

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from matplotlib.pyplot import *
from numpy import sqrt, pi, nanmax, abs, unwrap, NaN, where, any, squeeze, arange, ones

## Axes and Colors ##

# Colors:
    # http://colorbrewer2.org/, number of classes: 3 (or 4), nature: qualitative, only show: colorblind safe, print friendly
    # rslt: "4-class paired", colors: a6cee3, 1f78b4, b2df8a, (33a02c)
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
        if labeldir == 'x':
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


def matrix_axes(topology, ax=gca()):
    corrector_label(topology.corr_names, ax=ax)
    monitor_label(topology.mon_names, ax=ax)


def plot_topology(topology):
    """create a figure that shows the accelerator topology. Input: Topology object"""
    fig, ax = subplots()
    plot_matrix((topology.S_jk > 0).T, devlbl=r'$S_{kj}$ > 0', ax=ax)
    matrix_axes(topology, ax=ax)
    return fig


def plot_response(response, w=0, label='deviation', ax=gca()):
    """Plot response matrix into an axis for a given direction w (0: x, 1: y)"""
    plot_matrix(response.matrix[:,:,w], devlbl=label+' / '+response.unit, ax=ax)
    matrix_axes(response.topology, ax=ax)


def plot_residual(result, w=0, label='residual', ax=gca()):
    """plot fit residual into an axis for a given direction w (0: x, 1: y)"""
    plot_matrix(result.matrix[:,:,w] - result.response_matrix()[:,:,w], devlbl=label+' / '+result.unit, ax=ax)
    matrix_axes(result.topology, ax=ax)


def plot_Dev_err(result, w=0):
    """create a figure that shows response matrix and residual error for a given direction w (0: x, 1: y)"""
    fig, ax = subplots(2, 1, sharex=True, sharey=True)
    change_figsize(fig, 0)
    plot_response(result, w, ax=ax[0])
    setp(ax[0].get_xticklabels(), visible=False)
    plot_residual(result, w, ax=ax[1])
    return fig


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


def R_jmw(rslt, ax, m, direction='xy'):
    """plot real and imaginary parts of monitor vectors (incl. errors) into an axis for a given mode m"""
    ds = [0, 1]
    md = [str(m) + str(ds[0]), str(m) + str(ds[1])]

    for w in range(rslt.M):
        val = rslt.R_jmw[:, m, w]
        err = rslt.error.R_jmw[:, m, w]
        _plot_ReIm(ax, val, err)

    if 'invariants' in rslt.additional:  # normalized data
        ax.set_ylabel('$R_{jmw}$ / m')
    else:
        ax.set_ylabel('$R_{jmw}$ / m')
    _draw_zero(ax)


def cbeta_jmw(rslt, ax, m, comparison_data={}, direction='xy'):
    """plot beta resp. const*beta (incl. errors) into an axis for a given mode m"""
    # plot of beta or const*beta
    if 'invariants' in rslt.additional:  # normalized data
        betastr = r'$\beta'
    else:
        betastr = r'const. $\cdot \beta'

    for w in range(rslt.M):
        _plot_Re(ax, rslt.cbeta_jmw[:, m, w],
                 rslt.error.cbeta_jmw[:, m, w], 'COBEA %s' % direction[w])

    if 'beta' in comparison_data:
        ax.plot(arange(rslt.J), comparison_data['beta'][m], marker='.', color=modelc,
                label=comparison_data['name'] + ' ' + direction[m], linewidth=0.5)

    _ground_ylim(ax)
    ax.set_ylabel(betastr + '$ / m')


def delphi_jmw(rslt,ax,m,comparison_data={},yl=-1,direction='xy'):
    """
    plot phase-advance per monitor (incl. errors) into an axis for a given mode m
    """
    phi_jmw = rslt.phi_jmw
    bpmj=arange(rslt.J)-0.5
    if yl > 0: # background colors counting half integer advances
        plus = yl*ones(rslt.J)
        for j in xrange(rslt.J):
            if phi_jmw[j,m,m] < 0:
                plus[j] = NaN
        ax.fill_between(bpmj,0*plus,plus,facecolor=Re_clr2,
                alpha=0.4,linewidth=0)

    for w in range(rslt.M):
        _plot_Re(ax, rslt.delphi_jmw[:, m, w], rslt.error.delphi_jmw[:, m, w],
                'COBEA %s' % direction[w], xval=bpmj[1:])

    if 'phi' in comparison_data:
        phideg = 180 * unwrap(comparison_data['phi']
                              [m, 1:] - comparison_data['phi'][m, :-1]) / pi
        ax.plot(bpmj[1:], phideg, marker='.', color=modelc, linewidth=0.5,
            label=comparison_data['name']+' '+direction[m])

    #legend(ncol=3, loc=4, bbox_to_anchor=(1, 0.93))
    ax.set_ylabel(r'$\Delta \phi_' + direction[m] + '$ / deg')
    yl = ax.get_ylim()
    if yl[1] > 180:
        ax.set_ylim((0,180))
    else:
        ax.set_ylim((0,yl[1]))



def d_jw(rslt,ax,w,comparison_data,direction='xy'):
    """
    plot const*dispersion (incl. errors) into an axis for a given direction w (0: x, 1: y)
    """
    _plot_Re(ax, rslt.d_jw[:, w],
        rslt.error.d_jw[:, w],
             'COBEA $d_{j%c}$' % direction[w])
    if 'dispersion' in comparison_data:
        ax.plot(arange(rslt.J), comparison_data['dispersion'][w],
            marker='.', color=modelc,
            label=comparison_data['name'] + ' $D_%c$' % direction[w], linewidth=0.5)
    #legend(ncol=2, loc=4, bbox_to_anchor=(1, 0.93))
    _draw_zero(ax)
    ax.set_ylabel('m')


def monitor_results(rslt, m=0, comparison_data={}, direction='xy'):
    """
    plot monitor results for mode m,
    optionally in comparison with comparison_data.

    Parameters
    ----------
    result : object
        A :py:class:`cobea.model.Result` object.
    m : int
        mode index to plot results for
    comparison_data : dict
        a dictionary containing optional data from alternative decoupled storage ring models, which may contain the following keys:
        'name': name of the algorithm or model used
        'beta': an array of shape (rslt.M,rslt.J) that contains Courant-Snyder beta values for each direction and monitor
        'phi': an array of the same shape as 'beta', containing Courant-Snyder betatron phases
        'dispersion': an array of the same shape, containing dispersion values
    """

    n_fig=3
    if rslt.include_dispersion:
        n_fig+=1
    fig, ax = subplots(n_fig,1,sharex=True)
    for n in range(n_fig-1):
        setp(ax[n].get_xticklabels(), visible=False)
    [axi.grid(True, linestyle='--') for axi in ax]
    if rslt.J > 35:
        plot_type=3
    else:
        plot_type=6
    siz = plot_size(plot_type)
    if not rslt.include_dispersion:
        fig.set_size_inches((siz[0],siz[1]*0.75))
    else:
        fig.set_size_inches(siz)


    if not 'name' in comparison_data:
        comparison_data['name'] = 'comparison'

    #stri = 'COBEA' + ' tune: %.4f' % rslt.tune(m)
    #stri += ' $\pm$ %.4f' % abs(rslt.error.mu_m[m] / (2 * pi))
    #ax1.text(0, 1.1, stri, horizontalalignment='left', transform=ax1.transAxes,
    #        color=Re_clr, fontweight='bold', size='large')
    #if 'tunes' in comparison_data:
    #    ax1.text(1, 1.1, comparison_data['name'] + ' tune: %.4f' % comparison_data['tunes'][m],
    #         horizontalalignment='right', transform=ax1.transAxes,
    #         color=modelc, fontweight='bold', size='large')

    R_jmw(rslt,ax[0],m,direction=direction)
    ax[0].legend(ncol=2)

    cbeta_jmw(rslt,ax[1],m,comparison_data,direction=direction)
    ax[1].legend(ncol=2)

    delphi_jmw(rslt,ax[2],m,comparison_data,direction=direction)

    if rslt.include_dispersion:
        d_jw(rslt,ax[3],m,comparison_data,direction=direction)
        ax[3].legend(ncol=2)

    monitor_label(rslt.topology.mon_names, ax=ax[-1])
    return fig


## corrector quantities


def A_km(rslt, m, ax=gca()):
    """plot real and imaginary parts of corrector parameters (incl. errors) into an axis for a given mode m"""
    _plot_ReIm(ax, rslt.A_km[:, m], rslt.error.A_km[:, m])
    ax.set_ylabel('$A_{km}$ / a.u.')


def cbeta_km(rslt, m, ax=gca()):
    """
    plot const*beta at correctors assuming
    decoupled optics and thin correctors
    ToDo: errors for this quantity
    """
    _plot_Re(ax, rslt.cbeta_km[:, m], 0)
    ax.set_ylabel('$|A_{km}| \approx $ const $\beta_km$ / a.u.')


def delphi_km(rslt, m, ax=gca()):
    bpmk = arange(rslt.K - 1) + 0.5
    _plot_Re(ax, rslt.delphi_km[:, m], 0, xval=bpmk)
    ax.set_ylabel('$\Delta$ arg($A_{km}$) / deg')


def corrector_results(rslt, m=0):
    """create a figure with corrector results for a given mode m"""
    n_fig=3
    fig, ax = subplots(n_fig,1,sharex=True)
    for n in range(n_fig-1):
        setp(ax[n].get_xticklabels(), visible=False)
    [axi.grid(True, linestyle='--') for axi in ax]
    if rslt.K > 35:
        plot_type=3
    else:
        plot_type=6
    change_figsize(fig,plot_type)

    A_km(rslt,m,ax[0])
    #ax[0].legend(ncol=2)

    cbeta_km(rslt,m,ax[1])
    #ax[1].legend(ncol=2)

    delphi_km(rslt,m,ax[2])
    corrector_label(rslt.topology.corr_names, dir='x')
    return fig


def plot_result(result, print_figures=True, prefix='.', comparison_data={}, direction='xy', plot_flags='mcdt'):
    """
    plot COBEA results.

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
        'beta': an array of shape (rslt.M,rslt.J) that contains Courant-Snyder beta values for each direction and monitor
        'phi': an array of the same shape as 'beta', containing Courant-Snyder betatron phases
        'dispersion': an array of the same shape, containing dispersion values
    direction : str
        direction characters for the result object. can be 'x','y', or 'xy'.
    plot_flags : str
        which plots are to be created. Each character represents a different result plot:
        'm': monitor_results -> monitor_m*.pdf
        'c': corrector_results -> corrector_m*.pdf
        'd': plot_Dev_err, hist -> Dev_err_w*.pdf, hist_w*.pdf
        't': plot_topology -> topology.pdf
        'c': convergence information -> convergence.pdf. Only works if convergence information is available.
    """
    fig = prepare_figure(plot_type=0)

    erm = result.additional['err']

    if 'd' in plot_flags:
        for w in range(result.M):
            fig = plot_Dev_err(result, w)
            printshow(print_figures, prefix + 'Dev_err_w%i.pdf' % w, fig)

            fig = prepare_figure(2)
            hist(erm['chi_kjw'][:, :, w].flatten(), 40)
            xlabel('deviations / ' + result.unit)
            printshow(print_figures, prefix + 'hist_w%i.pdf' % w, fig)


    if 'c' in plot_flags and 'conv' in result.additional:  # convergence information is available
        change_figsize(fig, plot_type=2)  # 5
        # ax1=subplot(1,2,1)
        # xlabel('start-value iterations')
        # ax2=subplot(1,2,2,sharey=ax1)
        # setp( ax2.get_yticklabels(), visible=False)
        semilogy(result.additional['conv']['it'], result.additional['conv']['f'])
        xlabel('L-BFGS iterations')
        ylabel('residual squared error $\chi^2$')
        xlim((0, result.additional['conv']['it'][-1]))
        printshow(print_figures, prefix + 'convergence.pdf')

    if 'm' in plot_flags:
        for m in range(result.M):
            fig = monitor_results(result, m, comparison_data, direction=direction)
            printshow(print_figures, prefix + 'monitor_m%i.pdf' % m, fig)
            fig = corrector_results(result, m)
            printshow(print_figures, prefix + 'corrector_m%i.pdf' % m, fig)

    if 't' in plot_flags:
        fig = plot_topology(result.topology)
        printshow(print_figures, prefix + 'topology.pdf', fig)

    #for part in range(2):
    #    subplot(1, 2, part + 1)
    #    bar(range(result.additional['pca_singvals'][part].shape[0]),
    #        result.additional['pca_singvals'][part])
    #printshow(printfigs, prefix + 'PCA_singvals.pdf')
