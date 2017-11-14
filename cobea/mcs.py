"""
Monitor-Corrector Subset (MCS) algorithm submodule

MCS can be used as start-value layer of COBEA.
"""

from numpy import arange, angle, reshape, empty, dot, asarray, exp, sign, sqrt, NaN, \
    nanargmin, isnan, mod, nonzero, sum, abs, in1d
from numpy.linalg import pinv
from scipy.linalg import lstsq, svd, eig, eigh

from .model import Result


# Equations and residuals

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


# Corrector-Monitor (CM) mapping

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


def corrector_systems(Dev, monvec, bpm_s , corr_s, mus, printmsg=True, E=None):
    """set up and solve the corrector equation systems.
    Dev[k,f,d]: Deviations at all correctors for fast BPMs.
    monvec: all input monitor vectors.
    returns:
    D[k,m]: corrector parameters
    complexsolv parameters as arrays
    """
    if E is None:
        cE = phasejump_coeffs(bpm_s, corr_s, mus).conj()
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


# 'Two-Linac' part of MCS

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



def decomposite_eigenvec(Z):
    # monvecs_f = zeros((Z.shape[0]/2,2,2),dtype=Z.dtype)  #[f,m,d]
    M = int( Z.shape[0] / 2 )
    monvecs_fmw = empty((2, M, M), dtype=Z.dtype)
    for w in range(M):
        monvecs_fmw[0, :, w] = Z[w]
        monvecs_fmw[1, :, w] = Z[M + w]
    return monvecs_fmw


def part_mons_corrs(mon_idx, cor_idx, split_idx, L):
    # partmons[half], partcorrs[half] for each half:
    # L: should be J+K, len(line)
    partmons = [[], []]
    partcorrs = [[], []]

    # first part
    for elnum in range(split_idx[0, 0], split_idx[1, 1] + 1):
        if elnum in mon_idx:
            partmons[0].append(mon_idx.index(elnum))
        elif elnum in cor_idx:
            # remember: correctors must be outside, not inside of resp. part
            ci = cor_idx.index(elnum)
            if (elnum > split_idx[0, 1]) & (elnum < split_idx[1, 0]):
                partcorrs[1].append(ci)
        else:
            pass

    # second part. runs over s=0 point, hence mod is required
    for elnum in mod(range(split_idx[1, 0], split_idx[0, 1] + 1 + L), L):
        if elnum in mon_idx:
            partmons[1].append(mon_idx.index(elnum))
        elif elnum in cor_idx:
            # remember: correctors must be outside, not inside of resp. part
            ci = cor_idx.index(elnum)
            if (elnum > split_idx[1, 1]) | (elnum < split_idx[0, 0]):
                partcorrs[0].append(ci)
        else:
            pass
    return partmons, partcorrs


def _flat_dev(Dev, sym_dot):
    # Index transform of Deviation matrix (k,j,w) to squared PCA processing matrix (k,j*w)
    pcaproc = empty((Dev.shape[0],
                     Dev.shape[2] * Dev.shape[1]), dtype=Dev.dtype)
    pcaproc[:, :Dev.shape[1]] = Dev[:, :, 0]
    if Dev.shape[2] == 2:
        pcaproc[:, Dev.shape[1]:] = Dev[:, :, 1]
    return dot(pcaproc.T, pcaproc) if sym_dot else pcaproc


def _unflat_dev(pcaproc, Devshp):
    # Index transform of PCA processing matrix (k,j*w) to Deviation matrix (k,j,w)
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


def _dev_svd(rkjw_slice):
    u, s, v = svd(_flat_dev(rkjw_slice, False), full_matrices=False, overwrite_a=True, check_finite=False)
    return _unflat_dev(v[:4], rkjw_slice.shape)  # s, pcaCoeff = u[:, :4]


def _dev_eigh(rkjw_slice):
    x = _flat_dev(rkjw_slice, True)
    w, ve = eigh(x, overwrite_a=True, check_finite=False, eigvals=(x.shape[0]-5, x.shape[0]-1))
    return _unflat_dev(ve[:, -4:].T, rkjw_slice.shape)  # w, ()


def pca_tracking(Dev, partmons, partcorrs):
    # pcaDevs = []
    # for part in range(2):
    #     pcad = _dev_eigh(Dev[partcorrs[part]][:, partmons[part]])
    #     pcaDevs.append(pcad)
    pcaDevs = [_dev_svd(Dev[partcorrs[part]][:, partmons[part]]) for part in range(2)]

    Tpart = []
    for part in range(2):
        # make two-orbit vectors (similar to phase space vectors)
        # at beginning and end of partial orbits
        compvecs = composite_vectors(pcaDevs[part])
        # use only order=-1 for now (order=1 broken)?
        Tp, psi = compvecs_to_sectionmap(compvecs[0], compvecs[1], order=-1)
        Tpart.append(Tp)
    return Tpart, pcaDevs


def oneturn_eigen(T_zero, T_one):
    mum, Z = eig(dot(T_one, T_zero))
    #print('abs eigenvalues: %s' % abs(mum[[0,2]]) )

    if len(mum) == 4:
        # M=2
        return angle(mum[[0, 2]]), Z[:, [0, 2]]
    else:
        # M=1
        return [angle(mum[0])], Z[:, 0]


def mcs_core(result, mon_idx, cor_idx, split_idx):
    """
    MCS routine for a given monitor quadruplet.

    Parameters
    ----------
    result : object
        A valid :py:class:`cobea.model.Result` object.
    mon_idx : array
        1d array of integer positions of all considered monitors in result.line
    cor_idx: array
        1d array ... considered correctors in result.line
    split_idx : array_like
        a 2x2 array of monitor indices for the monitor quadruplet.

    Returns
    -------
    output : list
        ToDo for documentation
    """

    # PCA for mum, Z, local systems elsewhere
    partmons, partcorrs = part_mons_corrs(mon_idx, cor_idx, split_idx, result.J + result.K)

    if any([len(partcorrs[m]) < 2 * result.M for m in range(2)]):
        #print('     not enough correctors. next.')
        return NaN, 0, 0, 0, 0, 0, 0, 0

    Tpart, pcaDevs = pca_tracking(result.input_matrix, partmons, partcorrs)

    result.mu_m[:], Z = oneturn_eigen(Tpart[0], Tpart[1])

    monvecs_f = decomposite_eigenvec(Z)
    if monvecs_f.dtype==float:
        #print('     real eigenvectors / defective matrix. next.')
        return NaN, 0, 0, 0, 0, 0, 0, 0

    result.A_km[:], Res, Dev_fast_rc, SV = corrector_systems(result.input_matrix[:, in1d(mon_idx, split_idx[0]), :],
                                                             monvecs_f, split_idx[0], cor_idx, result.mu_m,
                                                             printmsg=False)
    result.R_jmw[:], rmsResidual, Dev_rc, SvM = monitor_systems(result.input_matrix, result.A_km, mon_idx,
                                                                cor_idx, result.mu_m, printmsg=False)

    if result.include_dispersion:
        Dev_res, result.d_jw[:], result.b_k[:] = dispersion_process(result.input_matrix, Dev_rc)
    else:
        Dev_res = result.input_matrix - Dev_rc
        result.d_jw[:] = 0# dsp = 0
        # b = 0 # dsp, b are casted into zeros at final run in function 'layer'
    rmsResidual = sum(Dev_res**2)
    #print('       mons: %i+%i, corrs: %i+%i, chi^2 = %.3e' %
    #      (len(partmons[0]), len(partmons[1]),
    #       len(partcorrs[0]), len(partcorrs[1]), rmsResidual))
    return rmsResidual, Dev_res, pcaDevs


def dice_splitpoints(n, mon_idx, split_idx):
    """
    Map the linear index n to bpm quadruplet index split_idx.
    As numpy arrays are passed by reference, split_idx is overwritten by this function.
    """
    if n < 0:
        shift = 0
        m = -n
    else:
        elmo = len(mon_idx) - 1
        shift = int(n / elmo)
        m = n - shift * elmo
        shift += 1
        if m >= len(mon_idx) / 2 - shift:
            m -= int( len(mon_idx) / 2 ) - shift
            shift = -(shift - 1)
    shift += int( len(mon_idx) / 2 )
    split_idx[0, 0] = mon_idx[m]
    split_idx[0, 1] = mon_idx[m + 1]
    split_idx[1, 0] = mon_idx[m + shift - 1]
    split_idx[1, 1] = mon_idx[m + shift]


def local_optimization(result, mon_idx, cor_idx, trials):
    split_idx = empty((2, 2), dtype=int)

    rms = empty(trials, dtype=float)
    for n in range(trials):
        dice_splitpoints(n, mon_idx, split_idx)
        rms[n] = mcs_core(result, mon_idx, cor_idx, split_idx)[0]
    if all(isnan(rms)):
        print('local optimization failed (few correctors). increase number of runs.')
    else:
        nmin = nanargmin(rms)

        dice_splitpoints(nmin, mon_idx, split_idx)
        return split_idx, rms


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


def layer(response, trials=-1):
    """
    implementation of the Monitor-Corrector Subspace algorithm

    Parameters
    ----------
    result : object
        A valid :py:class:`cobea.model.Response` object.
    trials: int
        Number of different monitor subsets tried for MCS. If set to -1, value is set automatically.
    """

    result = Result(response) # preallocate result (initialized by 'empty')

    mon_idx = list( result.topology.line_index(result.topology.mon_names) )
    cor_idx = list( result.topology.line_index(result.topology.corr_names) )

    if trials == -1:
        trials = 2 * (result.J - 1)
    print('MCS layer: running monitor quadruplet search (%i trials)...' % trials)
    split_idx, rmserr = local_optimization(result, mon_idx, cor_idx, trials)
    mon_strs = ['%s -- %s' % tuple([result.topology.line[x] for x in split_line]) for split_line in split_idx]
    print('    ...finished. Using %s.' % ' and '.join(mon_strs))

    # (re)compute the result for the best (smallest chi^2) monitor subset.
    ResM, Dev_rc, pcaDevs = mcs_core(result, mon_idx, cor_idx, split_idx)
    print('    chi^2 = %.3e (%s)^2' % (ResM, result.unit))
    mcs_dict = {'mon_idx': mon_idx, 'cor_idx': cor_idx, 'split_idx': split_idx, 'pca_orbits': pcaDevs}
    result.additional['MCS'] = mcs_dict
    return result
