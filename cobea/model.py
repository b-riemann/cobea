"""
This COBEA submodule defines all classes used by :py:class:`cobea`.
Besides input (:py:class:`Response`) and output (:py:class:`Result`) containers, this also includes gradient-based
optimization procedures in :py:class:`BE_Model`.

Bernard Riemann (bernard.riemann@tu-dortmund.de)
"""
from numpy import abs, angle, argsort, asarray, cumsum, dot, einsum, empty, exp, in1d, real, pi, \
    ravel_multi_index, ones, savez, sqrt, sum, zeros, nonzero, diff, mean
from scipy.linalg import svd, eigh
from warnings import warn
from pickle import dump

version = '0.24'


class BasicModel():
    """simple representation of the Bilinear-Exponential model (without topology or optimization attributes).

    Parameters
    ----------
    K,J,M : int
        dimensions of the model, with K being the number of correctors, J being the number of monitors,
        and M the number of modes respectively directions.
    init_fun : function
        a (possibly self-defined) initialization function like :py:func:`zeros` or :py:func:`empty` from numpy.

    Attributes
    ----------
    K : int
        total number of correctors. defines limit of corrector index k.
    J : int
        total number of monitors. defines limit of monitor index j.
    M : int
        number of directions respectively modes. defines limits of mode index m and direction index w.
    R_jmw : array
        monitor vectors in format [monitor, mode, direction]
    A_km : array
        corrector parameters, format [corrector, mode]
    d_jw : array
        unnormalized dispersion function at monitors, format [monitor, direction]
    b_k : array
        unnormalized dispersion coefficients at correctors, format [corrector] 
    mu_m : array
        fractional phase advances per turn (in rad)


    A reduced model class without topology or gradient computation"""
    def __init__(self, K,J,M, include_dispersion, init_fun=empty):
        self.R_jmw = init_fun((J, M, M), dtype=complex)
        self.A_km = init_fun((K, M), dtype=complex)
        self.d_jw = init_fun((J, M), dtype=float)
        self.b_k = init_fun((K,), dtype=float)
        self.mu_m = init_fun((M,), dtype=float)

        self.K = K
        self.J = J
        self.M = M
        
        # offset array for state vector, see _from_statevec
        # to model parameter conversion
        sizes = [M, self.A_km.size, self.A_km.size, self.R_jmw.size, self.R_jmw.size]
        if include_dispersion:
            sizes.extend([self.b_k.size, self.d_jw.size])
        self._offsets = cumsum(sizes)
        self.ndim = self._offsets[-1]  # number of search space dimensions

        self.include_dispersion = include_dispersion


    def _from_statevec(self, x):
        """read in all BE model variables from a state vector x.

        Parameters
        ----------
        x : array
            a vector representation all model parameters in the (flattened) order
             (mu_m, Re A_km, Im A_km, Re R_jmw, Im R_jmw, b_k, d_jw)
        """
        self.mu_m = x[:self._offsets[0]]  # do not use mod here, as this influences gradients and thus convergence
        self.A_km[:].real = x[self._offsets[0]:self._offsets[1]].reshape(self.K, self.M)
        self.A_km[:].imag = x[self._offsets[1]:self._offsets[2]].reshape(self.K, self.M)
        self.R_jmw[:].real = x[self._offsets[2]:self._offsets[3]].reshape(self.J, self.M, self.M)
        self.R_jmw[:].imag = x[self._offsets[3]:self._offsets[4]].reshape(self.J, self.M, self.M)

        if self.include_dispersion:
            self.b_k[:] = x[self._offsets[4]:self._offsets[5]]
            self.d_jw[:] = x[self._offsets[5]:self._offsets[6]].reshape(self.J, self.M)

    def _to_statevec(self):
        """
        convert all BE model parameters into a state vector x.

        Returns
        -------
        x : array
            a vector representation all model parameters in the (flattened) order
             (mu_m, Re A_km, Im A_km, Re R_jmw, Im R_jmw, b_k, d_jw)
        """
        x = empty(self.ndim, dtype=float)

        x[:self._offsets[0]] = self.mu_m
        x[self._offsets[0]:self._offsets[1]] = self.A_km.real.flatten()
        x[self._offsets[1]:self._offsets[2]] = self.A_km.imag.flatten()
        x[self._offsets[2]:self._offsets[3]] = self.R_jmw.real.flatten()
        x[self._offsets[3]:self._offsets[4]] = self.R_jmw.imag.flatten()

        if self.include_dispersion:
            x[self._offsets[4]:self._offsets[5]] = self.b_k
            x[self._offsets[5]:self._offsets[6]] = self.d_jw.flatten()

        return x


class ErrorModel(BasicModel):
    """
    A class computing and storing all BE model errors,
    including additional attributes for Ripken-Mais errors.
    """
    def __init__(self, K, J, M, include_dispersion, chi_squared, input_rms, unit):
        # this method contains out-commented code for an alternative to SVD usage (eigh).
        # the eigh values are sorted in increasing order, while svd is in decreasing order.

        BasicModel.__init__(self, K, J, M, include_dispersion)

        self.cbeta_jmw = empty(self.R_jmw.shape, dtype=float)
        self.phi_jmw = empty(self.R_jmw.shape, dtype=float)
        self.delphi_jmw = empty((self.J - 1, self.M, self.M), dtype=float)

        self.input_rms = input_rms
        self.unit = unit
        self.ri2v = -1.0
        self.additional = {}

        # variance by formula depending on effective degrees of freedom (without scaling invariants)
        # (``Error computations and approximate Hessian'')
        self.n_invariants = 2*self.M + (1 if self.include_dispersion else 0)  # each mode: 2 (complex), dispersion: 1 (real)
        effective_dof = self.ndim - self.n_invariants
        denominator = self.K*self.J*self.M - effective_dof
        self.chi_squared = chi_squared
        self.variance = chi_squared / denominator

    def parse_jacobian(self, jacobian_matrix, svd_mode=False):
        sqv = sqrt(self.variance)
        self.ri2v = self.input_rms / sqv
        # preparations for Hessian error estimation (H approx J J.T)
        if svd_mode:
            u, s, vh = svd( jacobian_matrix, full_matrices=False, overwrite_a=True, check_finite=False)
            # cut off the smallest values according to the remaining number of scaling invariants.
            # s = sigma * S_inv
            s[:-self.n_invariants] = sqv / s[:-self.n_invariants]
            s[-self.n_invariants:] = 0
        else:
            s, u = eigh(dot(jacobian_matrix, jacobian_matrix.T), overwrite_a=True, check_finite=False)
            s[self.n_invariants:] = sqv / sqrt(s[self.n_invariants:])
            s[:self.n_invariants] = 0

        u = einsum('ab,b->ab', u, s)  # = U sigma * S_inv
        # we now have sigma_rho^2 = A.T u u.T A
        self._from_statevec(sqrt(einsum('pq,pq->p', u, u)))  # errors of state vector variables
        return u

    def tune(self, m):
        return self.mu_m[m] / (2*pi)

    def __str__(self):
        s =  '                 chi^2: %.3e (%s)^2\n' % (self.chi_squared, self.unit)
        s += '    variance (exp.val): %.3e (%s)^2\n' % (self.variance, self.unit)
        s += '    input response rms: %.3e %s\n' % (self.input_rms, self.unit)
        s += '  root(input/variance): %.2f\n' % self.ri2v
        return s


def _sort_list_by_line(elem_list, line, desc):
    elem_list = asarray(elem_list)
    elem_nums = list()
    for elem in elem_list:
        try:
            elem_nums.append(line.index(elem))
        except ValueError:
            warn_str = 'line does not contain element %s; this %s will be omitted.' % (elem, desc)
            warn(warn_str)
    elem_nums = asarray(elem_nums)
    elem_nums_i = argsort( elem_nums )
    elem_nums = elem_nums[elem_nums_i]
    return elem_list[elem_nums_i], elem_nums, elem_nums_i

class Topology():
    """
    Representation of corrector/monitor labels and the order between them along the ring. During creation, all columns and rows of the input matrix, together with their labels in corr_names, mon_names, are re-ordered in ascending s-position according to the line list.

    Parameters
    ----------
    corr_names : list
        corrector labels (strings), e.g. ['HK01', 'VCM1', 'special_Hcorr', ...].
        The list index should correspond to the monitor_index, e.g. matrix[1,:,:]
        holds all information for the corrector named 'VCM1' in the above example.
    mon_names : list
        monitor labels (strings), e.g. ['BPM1','BPM2a','buggy_BPM',..,'important-bpm42'].
        the list index should correspond to the monitor_index, e.g. matrix[:,0,:]
        holds all information for the monitor named 'BPM1' in the above example.
    line : list
        corrector and monitor labels in ascending s position, downstream of the storage ring.
    corr_filters : list
        (optional)
        a list of filter strings with special character *. Example: To create
        one corrector set for all correctors with names starting with Cx, and another
        ending with dy, enter ('Cx*','*dy')
    assume_sorted : str
        cobea (especially cobea.mcs) requires rows and columns of the response to be sorted along the beam path.
        Only change this if you know what you are doing!

    """
    def __init__(self, corr_names, mon_names, line, corr_filters=(), assume_sorted=False):
        # sort correctors in s-position order
        # corr_names = asarray(corr_names)
        # cnums = list()
        # for corr in corr_names:
        #     try:
        #         cnums.append(line.index(corr))
        #     except ValueError:
        #         warn_str = 'line does not contain corrector %s of response; corrector will be omitted.' % corr
        #         warn(warn_str)
        # cnums = asarray(cnums)
        # cnums_i = argsort( cnums )
        # cnums = cnums[cnums_i]
        # self.corr_names = corr_names[cnums_i]
        self.corr_names, cnums, self.argsort_k = _sort_list_by_line( corr_names, line, 'corrector')

        self.corr_masks = {'all': ones(len(self.corr_names), dtype=bool)} if len(corr_filters) == 0 else {}
        for filter in corr_filters:
            self.corr_masks[filter] = filter_to_mask(filter, self.corr_names)

        # sort monitors in s-position order
        # mon_names = asarray(mon_names)
        # mnums = list()
        # for mon in mon_names:
        #     try:
        #         mnums.append(line.index(mon))
        #     except ValueError:
        #         warn_str = 'line does not contain corrector %s of response; corrector will be omitted.' % corr
        #         warn(warn_str)
        # mnums = asarray(mnums)
        # if len(mnums) < len(mon_names):
        #     warn('line does not contain all monitors of the response. the missing monitor information is removed.')
        # mnums_i = argsort( mnums )
        # mnums = mnums[mnums_i]
        # self.mon_names = mon_names[mnums_i]
        self.mon_names, mnums, self.argsort_j = _sort_list_by_line( mon_names, line, 'monitor')

        # generate S_jk matrix
        self.S_jk = empty((len(self.mon_names),len(self.corr_names)), dtype=int)
        for k, cnum in enumerate(cnums):
            self.S_jk[:,k] = 2*(cnum < mnums) - 1
        self.line = line
        # self.argsort_k = cnums_i
        # self.argsort_j = mnums_i

    def monitor_index(self, monitor_label):
        """
        find the index/indices of monitor label(s) in self.monitor_names
        """
        return _find_index(self.mon_names, monitor_label)

    def corrector_index(self, corrector_label):
        """
        find the index/indices of corrector label(s) in self.monitor_names
        """
        return _find_index(self.corr_names, corrector_label)

    def line_index(self, item_label):
        return _find_index(self.line, item_label)


class BEModel(BasicModel):
    """
    Bilinear-Exponential model with topology information and optimization routines.
    Besides the attributes and methods contained in :py:class:`Bare_Model`, the following information is included.

    Parameters
    ----------
    K,J,M : int
        dimensions of the model, with K being the number of correctors, J being the number of monitors, and M the number of modes respectively directions.
    init_fun : function
        a (possibly self-defined) initialization function like :py:func:`zeros` or :py:func:`empty` from numpy.

    Attributes
    ----------
    topology : object
        input topology, represented as :py:class:`Topology` object
    """
    def __init__(self, K, J, M, topology, include_dispersion, init_fun=empty):
        BasicModel.__init__(self, K, J, M, include_dispersion, init_fun)
        self.topology = topology

        # pre-allocate worker arrays for optimization
        self._c = empty((M, K, J, M), dtype=complex)
        self._x = empty(self.ndim)


    def E_jkm(self):
        if isinstance(self.mu_m, float):
            self.mu_m = [self.mu_m]
        E = empty((self.J, self.K, self.M), dtype=complex)
        for m in range(self.M):
            E[:, :, m] = exp(1.j * self.topology.S_jk * self.mu_m[m] / 2)
        return E

    def E_kjm(self):
        if isinstance(self.mu_m, float):
            self.mu_m = [self.mu_m]
        E = empty((self.K, self.J, self.M), dtype=complex)
        for m in range(self.M):
            E[:, :, m] = exp(1.j * self.topology.S_jk.T * self.mu_m[m] / 2)
        return E

    def _gradient_unwrapped(self, Dev, opt):
        """
        objective function value and gradient for all variables of the response problem.

        Parameters
        ----------
        Dev : array
            a general response matrix in format[k,j,w]
        opt: int
            a number to select what will be computed.
                =0: only function value (gradient vector is empty)
                =1: function value, mu gradient
                =2: f.val. and gradient (with or without dispersion depending on self.include_dispersion)

        Returns
        -------
        chi^2: float
            the function value
        grad: array
            a real-valued vector that contains dependent parameters and can be reshaped by
            :py:func:`Bare_Model._from_statevec`
        """
        xi = -Dev
        E = self.E_jkm()

        # self._c[m,k,j,w] = R*[j,m,w] E[j,k,m] D[k,m]
        einsum('jmw,jkm,km->mkjw',self.R_jmw.conj(), E, self.A_km, out=self._c)

        # xi[k,j,d] += sum_m self._c[m,k,j,d].real
        xi += sum(self._c.real, axis=0)

        if self.include_dispersion:
            # xi[k,j,w] += self.b_k[k] * self.d_jw[j,w]
            xi += einsum('k,jw->kjw',self.b_k, self.d_jw)

        # self._x acts as a gradient vector in the following

        if opt > 0:
            # grad mu[m] -= sum_jkw S[j,k] xi[k,j,w] self._c[m,k,j,w].imag
            self._x[:self.M] = -einsum('jk,kjw,mkjw->m',self.topology.S_jk,xi,self._c.imag)

            if opt > 1:
                pos = self.M
                pos2 = pos + self.A_km.size

                # grad A[k,m] = 2 sum_jw xi[k,j,w] R[j,m,w] E*[j,k,m]
                cmplx = 2*einsum('kjw,jmw,jkm->km',xi,self.R_jmw,E.conj()).flatten()
                self._x[pos:pos2] = cmplx.real
                pos = pos2 + self.A_km.size
                self._x[pos2:pos] = cmplx.imag

                pos2 = pos + self.R_jmw.size

                # grad R[j,m,w] = 2 sum_k xi[k,j,w] A[k,m] E[j,k,m]
                cmplx = 2*einsum('kjw,km,jkm->jmw',xi,self.A_km,E).flatten()
                self._x[pos:pos2] = cmplx.real
                pos = pos2 + self.R_jmw.size
                self._x[pos2:pos] = cmplx.imag

                if self.include_dispersion: #opt > 2:
                    pos2 = pos + self.K

                    # grad b[k] = 2 sum_jw xi[k,j,w] d[j,w]
                    self._x[pos:pos2] = 2*einsum('kjw,jw->k',xi,self.d_jw)

                    # grad d[j,w] = 2 sum_k xi[k,j,w] b[k]
                    self._x[pos2:] = 2*einsum('kjw,k->jw',xi,self.b_k).flatten()
        xi *= xi
        return sum(xi), self._x


    def _gradient(self, x, Dev, opt=3):
        """compute residual squared error and gradient (see :py:func:`gradient_unwrapped`)
        for the BE model parameters contained in x (using :py:func:`opt_unwrap`).
        This function is useful for optimization procedures using e.g. :py:class:`scipy.optimize`.

        Parameters
        ----------
        x : array
            a vector representation of data for :py:data:`mu_m`, :py:data:`R_jmw`, :py:data:`A_km`,
            :py:data:`d_jw`, and :py:data:`b_k`.
        Dev : array
            a general response matrix in format[k,j,w]
        opt: int
            a number to select what will be computed.
                =0: only function value (gradient vector is empty)
                =1: function value, mu gradient
                =2: f.val. and gradient (with or without dispersion depending on self.include_dispersion)

        Returns
        -------
        chi^2: float
            the function value
        grad: array
            a real-valued vector that contains gradients for all dependent model parameters
            and can be reshaped by :py:func:`Bare_Model.opt_unwrap`
        """
        self._from_statevec(x)
        return self._gradient_unwrapped(Dev, opt)

    def _jacobian_unwrapped(self):
        """compute the full Jacobian (and an array of offsets)

        Returns
        -------
        jacomat : array
            Full Jacobian matrix for the vector representation of the BE model.
        """

        jacomat = zeros([self.ndim, self.K * self.J * self.M])
        E_kjm = self.E_kjm()

        def kjw(k, j, w):
            return ravel_multi_index([k, j, w], (self.K, self.J, self.M))

        # mu_m
        for m in range(self.M):
            for k in range(self.K):
                for j in range(self.J):
                    for w in range(self.M):
                        cmplx = (self.R_jmw[j, m, w].conj() * E_kjm[k, j, m] * self.A_km[k, m]).imag
                        jacomat[m, kjw(k, j, w)] = -self.topology.S_jk[j, k] * cmplx / 2
                        # grad[m] = - sum(S.T *sum( xi * c[m].imag,axis=2)) #- sum_jk(sum_d( .. ))
                        # delmu[m] -= sum_jkd S[j,k] xi[k,j,d] c[m,k,j,d].imag

        # A_km
        pos = self._offsets[0]
        for k in range(self.K):
            for m in range(self.M):
                for j in range(self.J):
                    for w in range(self.M):
                        cmplx = self.R_jmw[j, m, w] * E_kjm[k, j, m].conj()
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = cmplx.real
                        jacomat[pos + self.A_km.size, idx] = cmplx.imag
                # cmplx = 2*sum(E[k,:,m].conj() *sum(xi[k] * R[:,m,:],axis=1)) #sum_j(sum_d(..))
                # = 2 sum_jd xi[k,j,d] * R[j,m,d] E[k,j,m].conj()
                pos += 1

        # R_jmw
        pos = self._offsets[2]
        for j in range(self.J):
            for m in range(self.M):
                for w in range(self.M):
                    for k in range(self.K):
                        cmplx = self.A_km[k, m] * E_kjm[k, j, m]
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = cmplx.real
                        jacomat[pos + self.R_jmw.size, idx] = cmplx.imag
                    # cmplx = 2*sum( xi[:,j,d] * D[:,m] * E[:,j,m] )
                    # = 2 sum_k xi[j,k,d] D[k,m] E[j,k,m]
                    pos += 1

        if self.include_dispersion:
            # b_k
            pos = self._offsets[4]  # offs[2]  # start index for b_k elements
            for k in range(self.K):
                for j in range(self.J):
                    for w in range(self.M):
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = self.d_jw[j, w]
                # grad[pos] = 2*sum( xi[k] * self.d_jw )
                pos += 1
            # d_jw
            for j in range(self.J):
                for w in range(self.M):
                    for k in range(self.K):
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = self.b_k[k]
                    # grad[pos] = 2*sum( xi[:,j,d] * self.b_k )
                    pos += 1
        return jacomat

    def response_matrix(self):
        """
        generate a 'simulated' response matrix from the present model parameters

        Returns
        -------
        rsim_kjw: array
            response array of shape (:py:data:`K`, :py:data:`J`, :py:data:`M`)
        """
        rsim_kjw = real(einsum('jmw,jkm,km->kjw', self.R_jmw,
                          self.E_jkm().conj(),self.A_km.conj()))

        if self.include_dispersion:
            rsim_kjw += einsum('k,jw->kjw',self.b_k,self.d_jw)
        return rsim_kjw

    @property
    def phi_jmw(self):
        """Compute Ripken-Mais betatron phases in units of degrees"""
        return angle(self.R_jmw*self.R_jmw[0].conj(), deg=True)

    @property
    def delphi_jmw(self):
        """Ripken-Mais phase advances per element"""
        return angle(self.R_jmw[1:]*self.R_jmw[:-1].conj(), deg=True)

    @property
    def cbeta_jmw(self):
        """Ripken-Mais beta parameters * constant.
        If self.R_jmw is normalized, constant = 1."""
        return self.R_jmw.real**2 + self.R_jmw.imag**2

    @property
    def delphi_km(self):
        """
        Betatron phase advances per corrector assuming
        decoupled optics and thin correctors
        """
        return angle(self.A_km[1:]*self.A_km[:-1].conj(), deg=True)

    @property
    def cbeta_km(self):
        """
        const*beta at correctors assuming
        decoupled optics and thin correctors
        """
        return self.A_km.real**2 + self.A_km.imag**2

    def flip_mu(self,m):
        """switch the sign of mu_m for given m, simultaneously changing the
        conjugation of :py:data:`R_jmw` and :py:data:`A_km` so that the response matrix remains unchanged"""
        self.mu_m[m] = -self.mu_m[m]
        self.R_jmw[:,m,:] = self.R_jmw[:,m,:].conj()
        self.A_km[:,m] = self.A_km[:,m].conj()


    def nu_m(self,m):
        nu_m = self.mu_m[m] / (2*pi)
        if nu_m < 0:
            nu_m += 1
        return nu_m

    def phase_integral(self, m):
        """integrated phase from first to last BPM (not one turn!),
        used for :py:func:`tune` computation"""
        return abs(sum( angle(self.R_jmw[1:, m, m] * self.R_jmw[:-1, m, m].conj()) ))

    def tune(self, m):
        """compute tune including integer part for a given mode m"""
        Q = self.nu_m(m)
        Q_min = self.phase_integral(m) / (2*pi)
        while Q < Q_min:
            Q += 1
        return Q

    def normalize(self, invariants):
        for m in range(self.M):
            if invariants[m] < 0:
                self.flip_mu(m)
                invariants[m] = -invariants[m]
            sq = sqrt(invariants[m])
            self.R_jmw[:, m] /= sq
            self.A_km[:, m] *= sq


def filter_to_mask(filter, labels):
    # generate a mask array from a filter, e.g. filter='BPM*' or filter='HK*'
    if filter.endswith('*'):
        x = [label.startswith(filter[:-1]) for label in labels]
    if filter.startswith('*'):
        x = [label.endswith(filter[1:]) for label in labels]
    return asarray(x, dtype=bool)


def _find_index(array, item_list):
    return nonzero( in1d(array, item_list) )[0]


class DriftSpace():
    """
    Representation of drift space information

    Parameters
    ----------
    mon_names : list
        list of monitor labels for the drift space
    length : float
        length of drift space
    """
    def __init__(self, mon_names, length):
        self.mon_names = mon_names
        self.length = length

    def r_prime_upstream(self, rj_drift, rj_drift_err=None):
        """
        Spatial derivative of spatial vector at the beginning of the drift space
        given spatial vectors at its ends.

        Returns
        -------
        r_prime : array_like
            spatial derivative
        r_prime_err : array_like
            error of spatial derivative for given rj_drift_err, else None
        """
        return diff(rj_drift, axis=0) / self.length, \
               None if rj_drift_err is None else diff(rj_drift_err, axis=0) / self.length

    def inside_tracking(self, rj_drift, delta_s, rj_drift_err=None):
        r_prime, r_prime_err = self.r_prime_upstream( rj_drift, rj_drift_err )
        return rj_drift[0] + r_prime * delta_s, \
               None if rj_drift_err is None else rj_drift_err[0] + r_prime_err * delta_s

    def __str__(self):
        return 'drift space in %s -- %s with length %.4f m.' % (self.mon_names[0], self.mon_names[1], self.length)


class Response():
    """
    Representation of COBEA input, used as such for the function :py:func:`cobea.cobea`

    During creation of the this object, py:data:`matrix` rows and columns, as well as the corresponding
    py:data:`corr_names` and py:data:`mon_names`, are resorted to their respective order in py:data:`line`.

    Parameters
    ----------
    matrix : array
        input response matrix of shape (correctors, monitors, directions).
        If only one direction is considered, the last dimension can be omitted.
    corr_names: list
        a list of corrector labels corresponding to each row of the matrix.
        See also corr_filters.
    mon_names: list
        a list of monitor labels corresponding to each column of the matrix
    line : list
        a list of element names in ascending s order
    include_dispersion : bool
        whether to use a model with or without dispersion for fitting. default: True
    unit : str
        (optional, default: '')
        unit for the input values of the matrix,
        containing a '/' character to separate monitor unit (e.g. mm) and corrector unit (e.g. A or mrad)
    drift_space : iterable
        (optional, default: None)
        a tuple or list with 3 elements (monitor name 1, monitor name 2, drift space length / m)
    corr_filters : list
        (optional)
        a list of filter strings with special character *. Example: To create
        one corrector set for all correctors with names starting with Cx, and another
        ending with dy, enter ('Cx*','*dy')
    name : str
        (optional, default: '')
        a short description of the measured response
    assume_sorted : str
        cobea (especially cobea.mcs) requires rows and columns of the response to be sorted along the beam path.
        Only change this if you know what you are doing!

    Attributes
    ----------
    topology : object
        A :py:class:`Topology` object holding the re-ordered py:data:'corr_names', py:data:'mon_names',
        and py:data:'line' as attributes.
    input_matrix : array
        re-ordered input response matrix.
    mon_unit : str
        unit for BPM readings (extracted from unit argument)
    corr_unit : str
        unit for corrector strengths (extracted from unit argument)
    known_element: object
        Known-element object used for normalization. At the moment, only :py:class:`DriftSpace` can be used.
    """
    def __init__(self, matrix, corr_names, mon_names, line, include_dispersion=True,
                 unit='a.u./a.u.', drift_space=None, corr_filters=(), name='', assume_sorted=False):
        self.version = version

        # self.input_matrix array must always have 3 dimensions, shape = (K, J, M)
        self.input_matrix = asarray(matrix)
        if matrix.shape[0] != len(corr_names):
            raise Exception('length of corr_names != number of correctors')
        elif matrix.shape[1] != len(mon_names):
            raise Exception('length of mon_names != number of monitors')
        elif matrix.ndim == 2:
            self.input_matrix = self.input_matrix[:, :, None]

        # sort monitors and correctors to line ("downstream") order
        self.topology = Topology(corr_names, mon_names, line, corr_filters, assume_sorted)
        if not assume_sorted:
            self.input_matrix = self.input_matrix[self.topology.argsort_k, :, :]
            self.input_matrix = self.input_matrix[:, self.topology.argsort_j, :]

        self.include_dispersion = include_dispersion
        self.mon_unit, self.corr_unit = unit.split('/')
        self.known_element = None if drift_space is None else DriftSpace(drift_space[:2], drift_space[2])
        self.name = name

    def pop_monitor(self,monitor_name):
        monitor_mask = self.topology.mon_names != monitor_name
        self.topology.mon_names = self.topology.mon_names[monitor_mask]
        self.topology.S_jk = self.topology.S_jk[monitor_mask, :]
        self.topology.argsort_j = self.topology.argsort_j[monitor_mask]

        self.input_matrix = self.input_matrix[:, monitor_mask]
        print('rsp> monitor %s removed' % monitor_name)

    def save(self, filename):
        """
        save the Response object as a pickle file with the given filename.
        The object can be reloaded using :py:func:`cobea.load_result`
        (which simply uses pickle)
        """
        with open(filename, 'wb') as f:
            dump(self, f, protocol=2)
        print('Response object saved in '+filename)


def error_str(val, err, fmt='.3e', sigma_scale=3, label=''):
    return ('{val:%s} +- {scaled_err:%s} ({sigma_scale:d} sigma)' % (fmt,fmt)).format(val=val,
                                                                                      scaled_err=sigma_scale*err,
                                                                                      sigma_scale=sigma_scale)


class Result(BEModel):
    """
    A container for all COBEA results that also computes secondary outputs on demand.

    Attributes
    ----------
    input_matrix : array
        Original input response matrix
    error : object
        computed BE model errors, represented as :py:class:`ErrorModel` object
    known_element: object
        Known-element object used for normalization. At the moment, only :py:class:`DriftSpace` can be used.
    additional : dict
        may contain the following keywords

        coretime : float
            time used for computation in the start and optimization layer.
        err : dict
            dictionary with additional model parameter error estimates.
        conv : dict
            dictionary with L-BFGS convergence information (if convergence_info was True)
        invariants : array
            computed during normalization of monitor vectors if drift_space is given.
            These are just returned for completeness and do not contain information about beam physics.
        pca_singvals : array
            custom info from MCS algorithm
        pca_orbits : array
            custom info from MCS algorithm
        version : str
            version of the object 
    """
    def __init__(self, response, additional={}, **kwargs):
        self.version = version
        self.input_matrix = response.input_matrix
        self.mon_unit = response.mon_unit
        self.corr_unit = response.corr_unit
        self.known_element = response.known_element
        self.name = response.name

        K, J, M = response.input_matrix.shape
        BEModel.__init__(self, K, J, M, response.topology, response.include_dispersion, **kwargs)
        self.error = None
        self.additional = additional

    @property
    def unit(self):
        return '/'.join((self.mon_unit, self.corr_unit))

    def chi_squared(self):
        return sum((self.input_matrix - self.response_matrix())**2)

    def input_rms(self):
        return sqrt(mean(self.input_matrix**2))

    def update_errors(self):
        """
        compute errors in attribute :py:data:`error` for given BE-Model parameters and input response,
        including errors for Ripken-Mais parameters
        """

        self.error = ErrorModel(self.K, self.J, self.M, self.include_dispersion, self.chi_squared(),
                                self.input_rms(), self.unit)

        u = self.error.parse_jacobian( self._jacobian_unwrapped() )

        real_idx = self._offsets[2]
        imag_idx = self._offsets[3]
        for j in range(self.J):
            for m in range(self.M):
                for w in range(self.M):
                    # compute beta error
                    A_Re = 2 * self.R_jmw[j, m, w].real
                    A_Im = 2 * self.R_jmw[j, m, w].imag
                    Atu = A_Re * u[real_idx, :] + A_Im * u[imag_idx, :]  # compute A.T u
                    self.error.cbeta_jmw[j, m, w] = sqrt(dot(Atu, Atu))

                    # compute phi error in DEGREES
                    beta = self.cbeta_jmw
                    A_Re = -self.R_jmw[j, m, w].imag / beta[j, m, w]
                    A_Im = self.R_jmw[j, m, w].real / beta[j, m, w]
                    Atu = A_Re * u[real_idx, :] + A_Im * u[imag_idx, :]  # compute A.T u
                    self.error.phi_jmw[j, m, w] = 180 * sqrt(dot(Atu, Atu)) / pi

                    real_idx += 1
                    imag_idx += 1

        for j in range(1, self.J):
            self.error.delphi_jmw[j-1] = sqrt(self.error.phi_jmw[j]**2 + self.error.phi_jmw[j - 1]**2)

    def save(self, filename):
        """
        save the Result object as a pickle file with the given filename.
        The object can be reloaded using :py:func:`cobea.load_result`
        (which simply uses pickle)
        """
        #if npz:
        #    model_attributes = 'unit', 'input_matrix', 'version', 'b_k', 'd_jw', 'mu_m', 'A_km', 'R_jmw',
        #    topology_attributes = 'mon_names', 'corr_names', 'line', 'argsort_j', 'argsort_k'
        #    kw_args = {attr: getattr(self, attr) for attr in model_attributes}
        #    kw_args.update({attr: getattr(self.topology, attr) for attr in topology_attributes})
        #    # if self.error is not None:
        #    #     error_attributes = list(model_attributes[-5:] if self.include_dispersion else model_attributes[-3:])
        #    #     error_attributes.extend(('cbeta_jmw', 'phi_jmw', 'delphi_jmw', 'input_rms', 'ri2v', 'additional'))
        #    #     kw_args.update({'err_'+attr: getattr(self.error, attr) for attr in error_attributes})
        #    savez(filename, **kw_args)
        #
        # else:
        with open(filename,'wb') as f:
            dump(self, f, protocol=2)
        print('Result object saved in '+filename)

    def __str__(self):
        s = 'parameter numbers:\n'
        s += '  K = %i correctors, J = %i monitors, M = %i directions\n' % (self.K, self.J, self.M)
        s += '  JKM = %i response elements\n  (JM+K)*' % self.input_matrix.size
        s += '(2M+1)' if self.include_dispersion else '2M'
        s += '+M = %i model parameters\n' % self.ndim

        s += 'tunes:\n'
        for m in range(self.M):
            s += error_str(self.tune(m), self.error.tune(m), fmt='.4f', label='mode %i:' % m) + '\n'
        # s += '(no errors computed yet)' if self.error is None else  self.error.__str__()
        return s


