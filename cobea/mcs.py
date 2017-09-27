"""
Monitor-Corrector Subset (MCS) algorithm submodule

MCS can be used as start-value layer of COBEA.
"""

from numpy import arange, angle, conj, copy, reshape, empty, dot, asarray, exp, sign, sqrt, NaN, \
    nanargmin, isnan, mod, nonzero, sum, abs
from numpy.linalg import pinv
from scipy.linalg import lstsq, svd, eig

from .model import Result


### Basic Index functions ###
# Note: topo_indices and find_indices do very similar things.
# One of them might be removed in a future release

def topo_indices(strilist, elto):
    """
    construct indices from stringlists. holds up to level.2 lists.
    strilist: list of elements
    elto: larger list of elements in which strilist elements are looked for.
    """
    if isinstance(strilist[0], str):
        re = [elto.index(stri) for stri in strilist]
    else:
        re = [[elto.index(stri) for stri in ll] for ll in strilist]
    return re  # asarray(re)

def find_indices(x, y):
    """
    find all indices i for which x[n] = y[i[n]] (j arbitrary).
    len(x) < len(y).
    (This function could be re-moved to __init__ later on)
    """
    # this naive method can be improved, see
    # http://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    indices = list()
    for el_x in x:
        indices.append(nonzero(el_x == y)[0][0])
    return asarray(indices)


### Equations and residuals ###

def unbrace(complex_matrix):
    w = complex_matrix.shape[1]
    real_matrix = empty((complex_matrix.shape[0], 2 * complex_matrix.shape[1]))
    real_matrix[:, arange(w)] = complex_matrix.real
    real_matrix[:, w + arange(w)] = complex_matrix.imag
    return real_matrix


def brace(vector):
    shap = list(vector.shape)
    shap[0] = int(shap[0] / 2)
    complex_vector = empty(shap, dtype=complex)
    complex_vector.real = vector[:shap[0]]
    complex_vector.imag = vector[shap[0]:]
    return complex_vector


def complexsolv(realvec, mat):
    """solve the half-complex equation system
    realvec = real(compmat*conj(compsol))
    for compsol.
    returns:
        compsol, res (from lstsq),
        realvec_rc: reconstructed realvec from compsol,
        s: singular values of compmat (from lstsq)"""
    mat = unbrace(mat)
    sol, res, rank, s = lstsq(mat, realvec)
    realvec_rc = dot(mat, sol).real
    sol = brace(sol)
    return sol, res, realvec_rc, s


### Corrector-Monitor (CM) mapping ###

def phasejump_coeffs(bpm_s, corr_s, mus):
    J = len(bpm_s)
    K = len(corr_s)
    M = len(mus)
    E = empty([J, K, M], dtype=complex)
    for j in range(J):
        for k in range(K):
            for m in range(M):
                E[j, k, m] = exp(1.j * sign(bpm_s[j] - corr_s[k]) * mus[m] / 2)
    return E


def corrector_matrix_k(R, cE):
    """output the complex corrector equation system matrix corrmat for a given corrector.
    corrmat.shape = [input_bpm*Directions+direction,mode]
    R:  full input monitor array, R.shape = [input_bpm,mode,direction]
    cE: conj(E[:,k,:]) of Ejkm
    """
    D = R.shape[2]
    corrmat = empty([R.shape[0] * D, R.shape[1]], dtype=cE.dtype)
    for f in range(R.shape[0]):
        for m in range(R.shape[1]):
            for d in range(D):
                corrmat[f * D + d, m] = R[f, m, d] * cE[f, m]
    return corrmat


def corrector_system_k(Dev_fd, R_fmd, cE_fm):
    covec = reshape(Dev_fd, -1)
    corrmat = corrector_matrix_k(R_fmd, cE_fm)
    compsol, res, covec_rc, s = complexsolv(covec, corrmat)
    return compsol, res, reshape(covec_rc, Dev_fd.shape), s


def corrector_systems(Dev, monvec, bpm_s , corr_s, mus, printmsg=True, E=[]):
    """set up and solve the corrector equation systems.
    Dev[k,f,d]: Deviations at all correctors for fast BPMs.
    monvec: all input monitor vectors.
    returns:
    D[k,m]: corrector parameters
    complexsolv parameters as arrays
    """
    if len(E) == 0:
        cE = conj(phasejump_coeffs(bpm_s , corr_s, mus))
    else:
        cE = E.conj()

    # create arrays (preallocate)
    D  = empty((cE.shape[1], cE.shape[2]), dtype=monvec.dtype)
    SV = empty((cE.shape[1], 2 * cE.shape[2]))
    Dev_rc = empty(Dev.shape, dtype=Dev.dtype)

    Res = list()
    for k in range(cE.shape[1]):  # range(len(corr_s)):
        D[k], x, Dev_rc[k], SV[k] = corrector_system_k(
            Dev[k], monvec, cE[:, k, :])
        Res.append(x)
    Res = asarray(Res)
    if printmsg:
        # this should be the error of the complete block-diagonal system:
        if Res.size != 0:
            print('CES> squared error  ' + str(sum(Res)))
        else:
            print('CES> finished. (not overdetermined)')
    return D, Res, Dev_rc, SV


def monitor_matrix_j(Y, E):
    """output the complex monitor equation system matrix monmat for a given monitor AND direction.
    monmat.shape = [corrector,mode]
    Y: corrector parameters, Y.shape = [corrector,mode]
    E: E[j,:,:] of Ejkm
    """
    return Y * E


def monitor_systems(Dev, D, all_bpm_s, corr_s, mus, printmsg=True, E=[]):
    """set up and solve the monitor equation systems, return R[j,m,d], the full monitor vector set for all monitors.
    Dev[k,j,d]: Deviations at all correctors for all BPMs.
    D[k,m]: all corrector parameters."""
    if len(E) == 0:
        E = phasejump_coeffs(all_bpm_s, corr_s, mus)  # E[j,k,m]

    # create arrays (preallocate)
    R = empty([Dev.shape[1], D.shape[1], Dev.shape[2]], dtype=D.dtype)
    Dev_rc = empty(Dev.shape, dtype=Dev.dtype)
    SV = empty([Dev.shape[1], Dev.shape[2], 2 * D.shape[1]])

    Res = list()  # Res[j,d]
    for j in range(Dev.shape[1]):
        momat = monitor_matrix_j(D, E[j])
        Res.append([])
        for d in range(Dev.shape[2]):
            R[j, :, d], x, Dev_rc[:, j, d], SV[
                j, d] = complexsolv(Dev[:, j, d], momat)
            Res[-1].append(x)  # Res[j,d] = x
    rmsResidual = 0
    return R, rmsResidual, Dev_rc, SV


### 'Two-Linac' part of MCS (including PCA) ###

def flatten_Dev(Dev):
    """
    Index transform of Deviation matrix (k,j,w)
    to PCA processing matrix (k,j*w)
    """
    pcaproc = empty((Dev.shape[0],
                     Dev.shape[2] * Dev.shape[1]), dtype=Dev.dtype)
    pcaproc[:, :Dev.shape[1]] = Dev[:, :, 0]
    if Dev.shape[2] == 2:
        pcaproc[:, Dev.shape[1]:] = Dev[:, :, 1]
    return pcaproc


def unflatten_Dev(pcaproc, Devshp):
    """
    Index transform of PCA processing matrix (k,j*w)
    to Deviation matrix (k,j,w)
    """
    #nmon = pcaproc.shape[1]/M
    Dev = empty((pcaproc.shape[0], Devshp[1], Devshp[2]),
                dtype=pcaproc.dtype)
    if Devshp[2] == 2:
        for orb in range(pcaproc.shape[0]):
            Dev[orb, :, 0] = pcaproc[orb, :Devshp[1]]
            Dev[orb, :, 1] = pcaproc[orb, Devshp[1]:]
    else:
        Dev[:, :, 0] = pcaproc
    return Dev


def pca_core(Dev, principal_orbits=True):
    """
    Principal Component Analysis of a Deviation matrix.
    """
    # Todo: implement also SVD-cleaned orbits
    pcaproc = flatten_Dev(Dev)
    u, s, v = svd(pcaproc)
    if principal_orbits:
        pcaDev = unflatten_Dev(v[:4], Dev.shape)
        pcaCoeff = u[:, :4]
    return s, pcaDev, pcaCoeff


def composite_vectors(pcaDev):
    """
    make two-orbit vectors (similar to phase space vectors)
    at beginning and end of partial orbits
    """
    M = pcaDev.shape[2]
    compvecs = empty((2, 2 * M, pcaDev.shape[0]),
                     dtype=pcaDev.dtype)
    for m in range(M):
        compvecs[0, m, :] = pcaDev[:, 0, m]
        compvecs[0, M + m, :] = pcaDev[:, 1, m]
        compvecs[1, m, :] = pcaDev[:, -2, m]
        compvecs[1, M + m, :] = pcaDev[:, -1, m]
    return compvecs


def compvecs_to_sectionmap(ps, pe, order):
    if order == -1:
        psi = pinv(ps)
        return dot(pe, psi), psi
    else:  # non-linear, it seems to be broken.. ps[:,i] -> ps[i] ?
        raise Exception('nonlinear mapping not supported yet')
    #    multiv_s, idxer = polyspace(ps[:, 0], ps[:, 1], ps[
    #                                :, 2], ps[:, 3], order)
    #    coeffs, res, rank, sg = lstsq(multiv_s.T, pe)
    #    return coeffs, idxer, sg


def oneturn_eigen(T_zero, T_one):
    mum, Z = eig(dot(T_one, T_zero))
    #print('abs eigenvalues: %s' % abs(mum[[0,2]]) )

    if len(mum) == 4:
        # M=2
        return angle(mum[[0, 2]]), Z[:, [0, 2]]
    else:
        # M=1
        return [angle(mum[0])], Z[:, 0]


def shiftring_eigen(Tpart):
    Tbar = empty((8, 8), dtype=Tpart[0].dtype)
    Tbar[4:, :4] = Tpart[0]
    Tbar[:4, 4:] = Tpart[1]
    lamb, Z = eig(Tbar)
    for s in range(8):
        Z[4:, s] /= lamb[s]
    lamb *= lamb  # lamb**2
    #print('abs eigenvalues: '+str(abs(lamb[[0,4]])))
    return abs(angle(lamb[[0, 4]])), Z[:4, [0, 4]]


def decomposite_eigenvec(Z):
    # monvecs_f = zeros((Z.shape[0]/2,2,2),dtype=Z.dtype)  #[f,m,d]
    M = int( Z.shape[0] / 2 )
    monvecs_fmw = empty((2, M, M), dtype=Z.dtype)
    for w in range(M):
        monvecs_fmw[0, :, w] = Z[w]
        monvecs_fmw[1, :, w] = Z[M + w]
    return monvecs_fmw


def part_mons_corrs(monidx, corridx, splitidx, L):
    # partmons[half], partcorrs[half] for each half:
    # L: should be J+K, len(line)
    partmons = [[], []]
    partcorrs = [[], []]

    # first part
    for elnum in range(splitidx[0, 0], splitidx[1, 1] + 1):
        if elnum in monidx:
            partmons[0].append(monidx.index(elnum))
        elif elnum in corridx:
            # remember: correctors must be outside, not inside of resp. part
            ci = corridx.index(elnum)
            if (elnum > splitidx[0, 1]) & (elnum < splitidx[1, 0]):
                partcorrs[1].append(ci)
        else:
            pass

    # second part. runs over s=0 point, hence mod is required
    for elnum in mod(range(splitidx[1, 0], splitidx[0, 1] + 1 + L), L):
        if elnum in monidx:
            partmons[1].append(monidx.index(elnum))
        elif elnum in corridx:
            # remember: correctors must be outside, not inside of resp. part
            ci = corridx.index(elnum)
            if (elnum > splitidx[1, 1]) | (elnum < splitidx[0, 0]):
                partcorrs[0].append(ci)
        else:
            pass
    return partmons, partcorrs


def pca_tracking(Dev, partmons, partcorrs):
    pcaDevs = []
    pcaCoeffs = []
    Sg = []
    for part in range(2):
        x = Dev[partcorrs[part]][:, partmons[part]]
        s, pcad, pcac = pca_core(x)
        pcaDevs.append(pcad)
        pcaCoeffs.append(pcac)
        Sg.append(s)

    Tpart = []
    for part in range(2):
        # make two-orbit vectors (similar to phase space vectors)
        # at beginning and end of partial orbits
        compvecs = composite_vectors(pcaDevs[part])
        # use only order=-1 for now (order=1 broken)?
        Tp, psi = compvecs_to_sectionmap(compvecs[0], compvecs[1], order=-1)
        Tpart.append(Tp)
    return Tpart, pcaDevs, Sg, pcaCoeffs


def mcs_core(Dev, monidx, corridx, splitidx, L, include_dispersion):

    # PCA for mum, Z, local systems elsewhere
    partmons, partcorrs = part_mons_corrs(monidx, corridx, splitidx, L)
    if any([len(partcorrs[m]) < 2 * Dev.shape[2] for m in range(2)]):
        #print('     not enough correctors. next.')
        return NaN, 0, 0, 0, 0, 0, 0, 0, 0

    Tpart, pcaDevs, Sg, pcaCoeffs = pca_tracking(Dev, partmons, partcorrs)
    mum, Z = oneturn_eigen(Tpart[0], Tpart[1])
    #mum, Z = shiftring_eigen(Tpart)
    #print('shiftring ' + repr(mum/(2*pi)))
    monvecs_f = decomposite_eigenvec(Z)
    if monvecs_f.dtype==float:
        #print('     real eigenvectors / defective matrix. next.')
        return NaN, 0, 0, 0, 0, 0, 0, 0, 0
    fi = find_indices(splitidx[0], monidx)
    A_km, Res, Dev_fast_rc, SV = corrector_systems(
        Dev[:, fi, :], monvecs_f, splitidx[0], corridx, mum, printmsg=False)
    monvec_j, rmsResidual, Dev_rc, SvM = monitor_systems(
        Dev, A_km, monidx, corridx, mum, printmsg=False)

    if include_dispersion:
        Dev_res, dsp, b = dispersion_process(Dev, Dev_rc)
    else:
        Dev_res = Dev - Dev_rc
        dsp = 0
        b = 0 # dsp, b are casted into zeros at final run in function 'layer' 
    rmsResidual = sum(Dev_res**2)
    #print('local method> squared error %s' % rmsResidual)
    print('       mons: %i+%i, corrs: %i+%i, chi^2 = %.3e' %
          (len(partmons[0]), len(partmons[1]),
           len(partcorrs[0]), len(partcorrs[1]), rmsResidual))
    return rmsResidual, Dev_res, monvec_j, A_km, mum, dsp, b, pcaDevs, Sg


def local_step(Dev, monidx, corridx, splitidx, L, include_dispersion):
    rmsResidual, Dev_rc, monvec_j, A_km, mum, dsp, b, pcaDevs, Sg = mcs_core(
        Dev, monidx, corridx, splitidx, L, include_dispersion)
    return rmsResidual


def dice_splitpoints(n, monidx, splitidx):
    """numpy arrays are passed by reference, so splitidx can be overwritten without return"""
    if n < 0:
        shift = 0
        m = -n
    else:
        elmo = len(monidx) - 1
        shift = int(n / elmo)
        m = n - shift * elmo
        shift += 1
        if m >= len(monidx) / 2 - shift:
            m -= int( len(monidx) / 2 ) - shift
            shift = -(shift - 1)
    shift += int( len(monidx) / 2 )
    splitidx[0, 0] = monidx[m]
    splitidx[0, 1] = monidx[m + 1]
    splitidx[1, 0] = monidx[m + shift - 1]
    splitidx[1, 1] = monidx[m + shift]


def dice_size(Lmonidx, maxstrain):
    # spl = Lmonidx # 1 was not substracted before, but fits better!
    return maxstrain * (Lmonidx - 1)


def local_optimization(Dev, monidx, corridx, Nelems, include_dispersion, runs):
    """solve CES and MES systems
    compute residual Res error
    and find optimal splitidx"""
    splitidx = empty([2, 2], dtype='int')


    rms = empty(runs)
    for n in range(runs):  # range(len(monidx)-spl-1):
        dice_splitpoints(n, monidx, splitidx)
        rms[n] = local_step(Dev, monidx, corridx, splitidx,
            Nelems, include_dispersion)
    if all(isnan(rms)):
        print('local optimization failed (few correctors). increase number of runs.')
    else:
        nmin = nanargmin(rms)

        dice_splitpoints(nmin, monidx, splitidx)
        return splitidx, rms


def dispersion_process(Dev, Dev_rc):
    Dev_res = Dev - Dev_rc
    #Dev_in = empty(Dev.shape,Dev.dtype)

    u, s, v = svd(Dev_res.reshape(Dev.shape[0], -1))
    s = sqrt(s[0])
    b = u[:, 0] * s
    v = v[0, :] * s
    dsp = v.reshape(Dev.shape[1], -1)
    for k in range(b.shape[0]):
        for w in range(Dev.shape[2]):
            Dev_res[k, :, w] -= b[k] * dsp[:, w]

    return Dev_res, dsp, b


def layer(response, trials = -1):
    """
    implementation of the Monitor-Corrector Subspace algorithm

    Parameters
    ----------
    response : object
        A valid :py:class:`cobea.model.Response` object.
    trials: int
        Number of different monitor subsets tried for MCS. If set to -1, value is set automatically.
    """

    # 1) Run MCS through a number (locruns) of monitor subsets and pick the one with the smallest chi^2.
    Dev_in = copy(response.matrix)
    line_len = sum(response.topology.S_jk.shape) #result.J + result.K
    monidx = topo_indices(response.topology.mon_names, response.topology.line)
    corridx = topo_indices(response.topology.corr_names, response.topology.line)

    if trials == -1:
        trials = dice_size(len(monidx), 2)
    print('MCS> running monitor doublet search (%i trials)' % trials)
    splitidx, rmserr = local_optimization(Dev_in, monidx, corridx,
                                          line_len, response.include_dispersion, trials)
    print('MCS> monitor doublet search finished. Using')
    for split_line in splitidx:
        print('       %s -- %s' % tuple([response.topology.line[x] for x in split_line]))

    # 2) (re)Compute the result for the best (smallest chi^2) monitor subset.
    result = Result(response) # preallocate result (initialized by 'empty')
    ResM, Dev_rc, result.R_jmw[:], result.A_km[:], \
        result.mu_m[:], result.d_jw[:], result.b_k[:], \
        pcaDevs, Sg = mcs_core(Dev_in, monidx, corridx, splitidx,
                               line_len, result.include_dispersion)
    mcs_dict = {'monidx': monidx, 'corridx': corridx, 'splitidx': splitidx,
                'pca_orbits': pcaDevs, 'pca_singvals': Sg}
    result.additional['MCS'] = mcs_dict
    return result
