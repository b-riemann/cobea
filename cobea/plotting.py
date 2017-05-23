"""
routines for plotting COBEA results
Bernard Riemann, April 2016
"""
from matplotlib.pyplot import *
#xlabel, ylabel, plot, semilogy, bar, imshow, subplot, xlim, ylim, axes, legend, \
#    colorbar, figure, rcParams, tight_layout, show, savefig, clf, setp, xticks, yticks, title, text, hist
from numpy import sqrt, pi, nanmax, abs, unwrap, NaN, where, any, squeeze, arange, ones

# ***************
# * Plot basics *
# ***************

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


def printshow(savecond, savename='', tight=True, equal=False):
    if tight:
        tight_layout()
    if equal:
        axes().set_aspect('equal', 'datalim')
    if not savecond:
        show()
    else:
        savefig(savename, transparent=True)
        print('figure printed to ' + savename + '.')
        clf()


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


# ***************
# * Orbit stuff *
# ***************


def monitor_label(mon_labels=0, spacing=0):
    xlim((-0.5,len(mon_labels)-0.5))
    if isinstance(mon_labels, int):
        xlabel('monitor index $j$')
    else:
        bpmj = arange(len(mon_labels))
        if spacing == 0:
            if len(mon_labels) > 60:
                spacing = 3
            elif len(mon_labels) > 40:
                spacing = 2
            else:
                spacing = 1
        xticks(bpmj[::spacing], mon_labels[::spacing],
               rotation='vertical', size='small', family='monospace')

def corrector_label(corr_labels=[], spacing=0):
    if len(corr_labels) == 0:
        ylabel('corrector index $k$')
    else:
        corj = arange(len(corr_labels))
        if spacing == 0:
            if len(corr_labels) > 60:
                spacing = 5
            elif len(corr_labels) > 40:
                spacing = 3
            else:
                spacing = 1
        yticks(corj[::spacing], corr_labels[::spacing],
               size='small', family='monospace')


def plot_matrix(Devdr, devlbl, maxdev=-1):
    if maxdev == -1:
        maxdev = nanmax(abs(Devdr))
    ext = [-0.5, Devdr.shape[1]-0.5, Devdr.shape[0]-0.5, -0.5]
    if any(Devdr < 0):
        imshow(squeeze(Devdr), extent=ext, vmin=-maxdev, vmax=maxdev,
               interpolation='nearest', aspect='auto', cmap='PRGn')
    else:
        imshow(Devdr, extent=ext, vmin=0, vmax=maxdev,
               interpolation='nearest', aspect='auto', cmap='Greens')
    cb = colorbar()
    # drc=['horiz.','vert.']
    cb.set_label(devlbl)


def matrix_axes(topology):
    corrector_label(topology.corr_names)
    monitor_label(topology.mon_names)


def plot_topology(topology):
    plot_matrix(topology.S_jk.T, devlbl='sign(S_kj)')
    matrix_axes(topology)


def plot_response(response, w=0, label='deviation'):
    plot_matrix(response.matrix[:,:,w], devlbl=label+' / '+response.unit)
    matrix_axes(response.topology)


def plot_difference(response1, response2matrix, w=0, label='difference'):
    plot_matrix(response1.matrix[:,:,w] - response2matrix[:,:,w], devlbl=label+' / '+response1.unit)
    matrix_axes(response1.topology)


def plot_result_residual(result, w=0, label='residual', dispersion=True):
    plot_difference(result, result.response_matrix(dispersion), w, label)


def plot_Dev_err(result, w=0):
    ax1 = subplot(2, 1, 1)
    plot_response(result, w)
    setp(ax1.get_xticklabels(), visible=False)
    subplot(2, 1, 2, sharex=ax1)
    plot_result_residual(result, w)


def plot_corrpars(Y):
    coridx = arange(Y.shape[0]) + 1
    w = 0.25
    bar(coridx - w, abs(Y[:, 0]), w, color='b', label='$m=1$')
    bar(coridx, abs(Y[:, 1]), w, color='g', label='$m=2$')
    # bar(coridx+0.5*w,rmsYerr,     w, color='k', label='rms error')
    # plot(abs(Y[:,0]),color='b',label='$m=1$')
    # plot(abs(Y[:,1]),color='g',label='$m=2$')
    ylabel('$|D_{km}|$')
    xlabel('corrector index $k$')
    xlim([0, Y.shape[0] + 1])
    legend()


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

def R_jmw(rslt,ax,m,direction='xy'):
    # cartesian coupled plot
    ds = [0, 1]
    md = [str(m) + str(ds[0]), str(m) + str(ds[1])]

    for w in range(rslt.M):
        val = rslt.R_jmw[:, m, w]
        err = rslt.error.R_jmw[:, m, w]
        _plot_ReIm(ax, val, err)

    #ax.legend(ncol=4, loc=1, bbox_to_anchor=(1, 1),
    #       prop={'size': 8})  # ,labelspacing=0.2
    if 'invariants' in rslt.additional: # normalized data
        ax.set_ylabel('$R_{jmw}$ / m')
        ### betastr = r'$\beta'
    else:
        betastr = r'const. $\cdot \beta'
        ax.set_ylabel('$R_{jmw}$ / m')
    _draw_zero(ax)

def beta_jmw(rslt,ax,m,comparison_data={},direction='xy'):
    # plot of beta or const*beta
    if 'invariants' in rslt.additional: # normalized data
        betastr = r'$\beta'
    else:
        betastr = r'const. $\cdot \beta'

    for w in range(rslt.M):
        _plot_Re(ax, rslt.cbeta_jmw[:, m, w],
                 rslt.error.cbeta_jmw[:, m, w], 'COBEA %s' % direction[w])

    if 'beta' in comparison_data:
        ax.plot(arange(rslt.J), comparison_data['beta'][m], marker='.', color=modelc,
             label=comparison_data['name']+' '+direction[m], linewidth=0.5)

    #ax[1].legend(ncol=4, loc=4, bbox_to_anchor=(1, 0.93))
    _ground_ylim(ax)
    ax.set_ylabel(betastr + '$ / m')


def delphi_jmw(rslt,ax,m,comparison_data={},yl=-1,direction='xy'):
    """
    phase-advance plot
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
    plot monitor (and mu) results for mode m,
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

    beta_jmw(rslt,ax[1],m,comparison_data,direction=direction)
    ax[1].legend(ncol=2)

    delphi_jmw(rslt,ax[2],m,comparison_data,direction=direction)

    if rslt.include_dispersion:
        d_jw(rslt,ax[3],m,comparison_data,direction=direction)
        ax[3].legend(ncol=2)

    monitor_label(rslt.topology.mon_names)


def plot_result(rslt, print_figures=False, prefix='.', comparison_data={},
                direction='xy'):
    """
    plot/printshow results of a factory.
    """
    fig = prepare_figure(plot_type=0)

    wlbl = ['horiz. ', 'vert. ']

    erm = rslt.additional['err']

    for w in range(rslt.M):
        change_figsize(fig, 0)
        plot_Dev_err(rslt, w)
        printshow(print_figures, prefix + 'Dev_err_w%i.pdf' % w)

        change_figsize(fig, 2)
        hist(erm['chi_kjw'][:, :, w].flatten(), 40)
        xlabel('deviations / ' + rslt.unit)
        printshow(print_figures, prefix + 'hist_w%i.pdf' % w)

    #for part in range(2):
    #    subplot(1, 2, part + 1)
    #    bar(range(rslt.additional['pca_singvals'][part].shape[0]),
    #        rslt.additional['pca_singvals'][part])
    #printshow(printfigs, prefix + 'PCA_singvals.pdf')

    if 'conv' in rslt.additional:  # convergence information is available
        change_figsize(fig, plot_type=2)  # 5
        # ax1=subplot(1,2,1)
        # xlabel('start-value iterations')
        # ax2=subplot(1,2,2,sharey=ax1)
        # setp( ax2.get_yticklabels(), visible=False)
        semilogy(rslt.additional['conv']['it'], rslt.additional['conv']['f'])
        xlabel('L-BFGS iterations')
        ylabel('residual squared error $\chi^2$')
        # axis('tight')
        xlim((0, rslt.additional['conv']['it'][-1]))
        printshow(print_figures, prefix + 'convergence.pdf')


    for m in range(rslt.M):
        monitor_results(rslt, m, comparison_data, direction=direction)
        printshow(print_figures, prefix + 'monitor_m%i.pdf' % m)


#def plot(obj):
#    """
#    Plot an arbitrary COBEA object
#    :param obj: A cobea object of the class Response, be, or Result
#    """
#    #if isinstance(obj, be)
