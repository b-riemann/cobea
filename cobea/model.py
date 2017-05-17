"""
This submodule defines all classes used by :py:class:`cobea`. Besides input (:py:class:`Response`) and output (:py:class:`Result`) containers, this also includes gradient-based optimization procedures in :py:class:`BE_Model`
"""
from numpy import abs, angle, arange, argsort, asarray, cumsum, diag, \
    dot, empty, exp, NaN, real, rollaxis, outer, pi, \
    ravel_multi_index, sqrt, sum, zeros
from scipy.linalg import svd # eigh
from warnings import warn
from pickle import dump


class BasicModel():
    """simple representation of the Bilinear-Exponential model (without topology or optimization attributes).

    Parameters
    ----------
    (K,J,M) : tuple
        dimensions of the model, with K being the number of correctors, J being the number of monitors, and M the number of modes respectively directions.
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
    def __init__(self, (K,J,M), init_fun=empty):
        self.R_jmw = init_fun((J, M, M), dtype=complex)
        self.A_km = init_fun((K, M), dtype=complex)
        self.d_jw = init_fun((J, M))
        self.b_k = init_fun((K,))
        self.mu_m = init_fun((M,))

        self.K = K
        self.J = J
        self.M = M

    def _opt_unwrap(self, x):
        """read in all BE model variables from their vector representation.

        Parameters
        ----------
        x : array
            a vector representation of data for :py:data:`mu_m`, :py:data:`R_jmw`, :py:data:`A_km`, :py:data:`d_jw`, and :py:data:`b_k`.
        """
        self.mu_m = x[:self.M]  # do not use mod here, as this influences xgrad and thus convergence of l-bfgs

        # K=(len(x)-M)/(2*M)
        pos = arange(self.M, 2 * self.M)
        for k in xrange(self.K):
            self.A_km[k].real = x[pos]
            self.A_km[k].imag = x[pos + self.A_km.size]
            pos += self.M
        pos += self.A_km.size
        for j in xrange(self.J):
            for m in range(self.M):
                self.R_jmw[j, m].real = x[pos]
                self.R_jmw[j, m].imag = x[pos + self.R_jmw.size]
                pos += self.M
        pos += self.R_jmw.size
        if pos[0] != len(x):  # ..then there is dispersion info
            self.b_k = x[pos[0]:pos[0] + self.K]
            pos += self.K
            for j in xrange(self.J):
                self.d_jw[j] = x[pos]
                pos += self.M

    def _opt_wrap(self):
        """
        convert all BE model parameters into the vector representation.

        Returns
        -------
        x : array
            a vector representation of the data contained in :py:data:`mu_m`, :py:data:`R_jmw`, :py:data:`A_km`, :py:data:`d_jw`, and :py:data:`b_k`.
        """
        pos = self.M
        self._x[:pos] = self.mu_m
        for k in xrange(self.K):
            for m in range(self.M):
                self._x[pos] = self.A_km[k, m].real
                self._x[pos + self.A_km.size] = self.A_km[k, m].imag
                pos += 1
        pos += self.A_km.size
        for j in xrange(self.J):
            for m in range(self.M):
                for d in range(self.M):
                    self._x[pos] = self.R_jmw[j, m, d].real
                    self._x[pos + self.R_jmw.size] = self.R_jmw[j, m, d].imag
                    pos += 1
        if self.include_dispersion:
            pos += self.R_jmw.size
            if self.b_k.ndim == 1:  # coupled dispersion
                self._x[pos:pos + self.b_k.size] = self.b_k.flatten()
                pos += self.b_k.size
                for j in xrange(self.J):
                    self._x[pos:pos + self.M] = self.d_jw[j]
                    pos += self.M
        return self._x


class ErrorModel(BasicModel):
    def __init__(self, (K,J,M)):
        BasicModel.__init__(self, (K, J, M))
        # additional containers for Ripken-Mais errors. In BEModel, the Ripken-Mais
        # expectation values are generated from eigenorbits directly
        self.cbeta_jmw = empty(self.R_jmw.shape,dtype=float)
        self.phi_jmw = empty(self.R_jmw.shape,dtype=float)
        self.delphi_jmw = empty((self.J-1,self.M,self.M),dtype=float)


class BEModel(BasicModel):
    """
    Bilinear-Exponential model with topology information and optimization routines.
    Besides the attributes and methods contained in :py:class:`Bare_Model`, the following information is included.

    Parameters
    ----------
    (K,J,M) : tuple
        dimensions of the model, with K being the number of correctors, J being the number of monitors, and M the number of modes respectively directions.
    init_fun : function
        a (possibly self-defined) initialization function like :py:func:`zeros` or :py:func:`empty` from numpy.

    Attributes
    ----------
    topology : object
        input topology, represented as :py:class:`Topology` object
    """
    def __init__(self, (K, J, M), topology, include_dispersion, init_fun=empty):
        BasicModel.__init__(self, (K, J, M), init_fun)
        self.topology = topology

        # number of search space dimensions:
        self.ndim = self.M + 2 * self.A_km.size + 2 * self.R_jmw.size  # D[k,m], R[j,m,d]
        if self.include_dispersion:
            self.ndim += self.K + self.J * self.M  # += K+J*D

        # pre-allocate worker arrays for optimization
        self._c = empty((M, K, J, M), dtype=complex)
        self._x = empty(self.ndim)
        self.include_dispersion = include_dispersion

    def _offsets(self):
        """
        the offsets array contains offsets for :py:data:`A_km`, :py:data:`R_jmw`, :py:data:`b_k` and :py:data:`d_jw` in that order.
        """
        if self.include_dispersion:
            return cumsum((self.M, 2 * self.A_km.size, 2 * self.R_jmw.size, self.b_k.size, self.d_jw.size))
        else:
            return cumsum((self.M, 2 * self.A_km.size, 2 * self.R_jmw.size))


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

    def _gradient_unwrapped(self, Dev, opt=1):
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
                =2: f.val., gradient without dispersion (no b_k and d_jw components)
                (=3: f.val, full gradient with dispersion.)
        
        Returns
        -------
        chi^2: float
            the function value
        grad: array
            a real-valued vector that contains dependent parameters and can be reshaped by :py:func:`Bare_Model.opt_unwrap`
        """
        xi = -Dev
        E_kjm = self.E_kjm()

        # compute self._c[m,k,j,d], equivalent to:
        # for m in xrange(self.M):
        #    for k in xrange(self.K):
        #        for j in xrange(self.J):
        #            for d in xrange(self.M):
        #                self._c[m,k,j,d] = R[j,m,d].conj() * E[k,j,m] * D[k,m]
        for m in xrange(self.M):
            for k in xrange(self.K):
                self._c[m, k] = self.R_jmw[:, m].conj()
                self._c[m, k] *= self.A_km[k, m]
        self._c *= rollaxis(E_kjm, 2)[:, :, :, None]

        xi += sum(self._c.real, axis=0)  # xi[k,j,d] += self._c[m,k,j,d].real #sum_m
        if self.include_dispersion:
            for w in range(self.M):
                # xi[k,j,w] += self.b_k[k] * self.d_jw[j,w]
                xi[:, :, w] += outer(self.b_k, self.d_jw[:, w])

        # self._x acts as a gradient vector in the following

        if opt > 0:
            # grad mu_m
            for m in xrange(self.M):
                # - sum_jk(sum_d( .. ))
                self._x[m] = - sum(self.topology.S_jk.T * sum(xi * self._c[m].imag, axis=2))
                # delmu[m] -= sum_jkd S[j,k] xi[k,j,d] self._c[m,k,j,d].imag

            if opt > 1:
                pos = self.M
                # grad A_km
                for k in xrange(self.K):
                    for m in xrange(self.M):
                        # cmplx = 2 sum_jw xi[k,j,w] * R[j,m,w] E[k,j,m].conj()
                        cmplx = 2 * sum(E_kjm[k, :, m].conj() *
                                        sum(xi[k] * self.R_jmw[:, m, :], axis=1))
                        self._x[pos] = cmplx.real
                        self._x[pos + self.A_km.size] = cmplx.imag
                        pos += 1
                pos += self.A_km.size
                # grad R_jmw
                for j in xrange(self.J):
                    for m in xrange(self.M):
                        for w in xrange(self.M):
                            # cmplx = 2 sum_k xi[j,k,w] D[k,m] E[j,k,m]
                            cmplx = 2 * sum(xi[:, j, w] * self.A_km[:, m] * E_kjm[:, j, m])
                            self._x[pos] = cmplx.real
                            self._x[pos + self.R_jmw.size] = cmplx.imag
                            pos += 1
                if self.include_dispersion: #opt > 2:
                    pos += self.R_jmw.size
                    # grad b_k
                    for k in xrange(self.K):
                        self._x[pos] = 2 * sum(xi[k] * self.d_jw)
                        pos += 1
                    # grad d_jw
                    for j in xrange(self.J):
                        for w in xrange(self.M):
                            self._x[pos] = 2 * sum(xi[:, j, w] * self.b_k)
                            pos += 1
        xi *= xi
        return sum(xi), self._x

    def _gradient(self, x, Dev, opt=3):
        """compute residual squared error and gradient (see :py:func:`gradient_unwrapped`)
        for the BE model parameters contained in x (using :py:func:`opt_unwrap`).
        This function is useful for optimization procedures using e.g. :py:class:`scipy.optimize`.

        Parameters
        ----------
        x : array
            a vector representation of data for :py:data:`mu_m`, :py:data:`R_jmw`, :py:data:`A_km`, :py:data:`d_jw`, and :py:data:`b_k`.
        Dev : array
            a general response matrix in format[k,j,w]
        opt: int
            a number to select what will be computed.

                =0: only function value (gradient vector is empty)

                =1: function value, mu gradient

                =2: f.val., gradient with/without dispersion (b_k and d_jw components) depending on self.include_dispersion

        Returns
        -------
        chi^2: float
            the function value
        grad: array
            a real-valued vector that contains gradients for all dependent model parameters
            and can be reshaped by :py:func:`Bare_Model.opt_unwrap`
        """
        self._opt_unwrap(x)
        return self._gradient_unwrapped(Dev, opt)

    def _jacobian_unwrapped(self):
        """compute the full Jacobian (and an array of offsets)

        Returns
        -------
        jacomat : array
            Full Jacobian matrix for the vector representation of the BE model.
        offs : array
            A list of vector representation offsets computed by :py:func:`offsets`.
        """

        offs = self._offsets()
        jacomat = zeros([offs[-1], self.K * self.J * self.M])
        E_kjm = self.E_kjm()

        def kjw(k, j, w):
            return ravel_multi_index([k, j, w], (self.K, self.J, self.M))

        # mu_m
        for m in range(self.M):
            for k in xrange(self.K):
                for j in xrange(self.J):
                    for w in range(self.M):
                        cmplx = (self.R_jmw[j, m, w].conj() * E_kjm[k, j, m] * self.A_km[k, m]).imag
                        jacomat[m, kjw(k, j, w)] = -self.topology.S_jk[j, k] * cmplx / 2
                        # grad[m] = - sum(S.T *sum( xi * c[m].imag,axis=2)) #- sum_jk(sum_d( .. ))
                        # delmu[m] -= sum_jkd S[j,k] xi[k,j,d] c[m,k,j,d].imag

        pos = offs[0]
        # A_km
        for k in xrange(self.K):
            for m in range(self.M):
                for j in xrange(self.J):
                    for w in range(self.M):
                        cmplx = self.R_jmw[j, m, w] * E_kjm[k, j, m].conj()
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = cmplx.real
                        jacomat[pos + self.A_km.size, idx] = cmplx.imag
                # cmplx = 2*sum(E[k,:,m].conj() *sum(xi[k] * R[:,m,:],axis=1)) #sum_j(sum_d(..))
                # = 2 sum_jd xi[k,j,d] * R[j,m,d] E[k,j,m].conj()
                pos += 1
        # R_jmw
        pos = offs[1]  # start index for R_jmw elements
        for j in xrange(self.J):
            for m in xrange(self.M):
                for w in xrange(self.M):
                    for k in xrange(self.K):
                        cmplx = self.A_km[k, m] * E_kjm[k, j, m]
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = cmplx.real
                        jacomat[pos + self.R_jmw.size, idx] = cmplx.imag
                    # cmplx = 2*sum( xi[:,j,d] * D[:,m] * E[:,j,m] )
                    # = 2 sum_k xi[j,k,d] D[k,m] E[j,k,m]
                    pos += 1

        if self.include_dispersion:
            # b_k
            pos = offs[2]  # start index for b_k elements
            for k in xrange(self.K):
                for j in xrange(self.J):
                    for w in xrange(self.M):
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = self.d_jw[j, w]
                # grad[pos] = 2*sum( xi[k] * self.d_jw )
                pos += 1
            # self.d_jw
            for j in xrange(self.J):
                for w in xrange(self.M):
                    for k in xrange(self.K):
                        idx = kjw(k, j, w)
                        jacomat[pos, idx] = self.b_k[k]
                    # grad[pos] = 2*sum( xi[:,j,d] * self.b_k )
                    pos += 1
        return jacomat, offs

    def response_matrix(self, dispersion=True):
        """
        generate a 'simulated' response matrix from the present model parameters
        
        Returns
        -------
        Dev: array
            response array of shape (:py:data:`K`, :py:data:`J`, :py:data:`M`)
        """
        # generated Dev[k,j,d] = "real( R  E* D* )  [ + dsp * b ]"
        Dev = empty((self.K, self.J, self.M))
        cE = self.E_jkm().conj()
        for k in xrange(self.K):
            for j in xrange(self.J):
                for w in range(self.M):
                    Dev[k, j, w] = sum(
                        real(self.R_jmw[j, :, w] * cE[j, k] * self.A_km[k].conj()))
        if dispersion:
            for w in range(self.M):
                Dev[:, :, w] += outer(self.b_k, self.d_jw[:, w])
        return Dev

    @property
    def phi_jmw(self):
        """Compute Ripken-Mais betatron phases"""
        return angle(self.R_jmw*self.R_jmw[0].conj(),deg=True)

    @property
    def delphi_jmw(self):
        """Ripken-Mais phase advances per element"""
        return angle(self.R_jmw[1:]*self.R_jmw[:-1].conj(),deg=True)

    @property
    def cbeta_jmw(self):
        """Ripken-Mais beta parameters * constant.
        If self.R_jmw is normalized, constant = 1."""
        return self.R_jmw.real**2 + self.R_jmw.imag**2

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
    
    """
    def __init__(self, corr_names, mon_names, line):
        # sort correctors in s-position order
        corr_names = asarray(corr_names)
        cnums = asarray([line.index(corr) for corr in corr_names])
        if len(cnums) < len(corr_names):
            warn('line does not contain all correctors of the response. the missing corrector information is removed.')
        cnums_i = argsort( cnums )
        cnums = cnums[cnums_i]
        self.corr_names = corr_names[cnums_i]

        # sort monitors in s-position order
        mon_names = asarray(mon_names)
        mnums = asarray([line.index(mons) for mons in mon_names])
        if len(mnums) < len(mon_names):
            warn('line does not contain all monitors of the response. the missing monitor information is removed.')
        mnums_i = argsort( mnums )
        mnums = mnums[mnums_i]
        self.mon_names = mon_names[mnums_i]

        # generate S_jk matrix
        self.S_jk = empty((len(mnums),len(cnums)), dtype=int)
        for k, cnum in enumerate(cnums):
            self.S_jk[:,k] = 2*(cnum < mnums) - 1
        self.line = line
        self.argsort_k = cnums_i
        self.argsort_j = mnums_i


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
        a list of corrector labels corresponding to each row of the matrix
    mon_names: list
        a list of monitor labels corresponding to each column of the matrix
    line : list
        a list of element names in ascending s order
    unit : str
        unit for the input values of the matrix (optional)

    Attributes
    ----------
    topology : object
        A :py:class:`Topology` object holding the re-ordered py:data:'corr_names', py:data:'mon_names',
        and py:data:'line' as attributes.
    matrix : array
        re-ordered input response matrix.

    """
    def __init__(self, matrix, corr_names, mon_names, line,
                 include_dispersion=True, unit=''):
        self.matrix = asarray(matrix)
        if matrix.shape[0] != len(corr_names):
            raise Exception('list of corr_names != number of correctors')
        elif matrix.shape[1] != len(mon_names):
            raise Exception('list of mon_names != number of monitors')
        elif matrix.ndim == 2:
            # the matrix array must always have 3 dimensions, shape = [K,J,M]
            self.matrix = self.matrix[:, :, None]

        self.topology = Topology(corr_names, mon_names, line)
        self.matrix = self.matrix[self.topology.argsort_k, :, :]
        self.matrix = self.matrix[:, self.topology.argsort_j, :]
        self.include_dispersion = include_dispersion
        self.unit = unit

    def pop_monitor(self,monitor_name):
        monitor_mask = self.topology.mon_names != monitor_name
        self.topology.mon_names = self.topology.mon_names[monitor_mask]
        self.topology.S_jk = self.topology.S_jk[monitor_mask,:]
        self.topology.argsort_j = self.topology.argsort_j[monitor_mask]

        self.matrix = self.matrix[:, monitor_mask]
        print('Rsp> monitor %s removed' % monitor_name)


class Result(BEModel):
    """
    COBEA Result.

    Besides the attributes and methods contained in :py:class:`BEModel`, the following information is included.

    Attributes
    ----------
    matrix : array
        Original input response matrix
    error : object
        computed BE model errors, represented as :py:class:`ErrorModel` object
    additional : dict
        may contain the following keywords

        coretime : float
            time used for computation in the start and optimization layer.
        err : dict
            dictionary with additional model parameter error estimates.
        conv : dict
            dictionary with L-BFGS convergence information (if convergence_info was True)
        invariants : array
            computed during normalization of monitor vectors if drift space is given. These are just returned for completeness and do not contain information about beam physics.
        pca_singvals : array
            custom info from MCS algorithm
        pca_orbits : array
            custom info from MCS algorithm
        version : str
            version of the object 
    """
    def __init__(self, response, additional={}, **kwargs):
        # this init function is only used inside the module and is thus not commented.
        self.version = '0.11'
        self.matrix = response.matrix
        self.include_dispersion = response.include_dispersion
        self.unit = response.unit
        BEModel.__init__(self, response.matrix.shape, response.topology, response.include_dispersion, **kwargs)
        self.error = ErrorModel(response.matrix.shape)
        self.additional = additional

    def update_errors(self):
        """
        compute errors in attribute :py:data:`error` for given BE-Model parameters and input response, including errors for Ripken-Mais parameters
        """
        # Note: this function contains out-commented code for an alternative to SVD usage (eigh).
        # the eigh values are sorted in increasing order, while svd is in decreasing order.

        # preparations for Hessian error estimation (H approx J J.T)
        jacomat, offs = self._jacobian_unwrapped()
        u, s, vh = svd(jacomat, full_matrices=False)
        #w, v = eigh(dot(jacomat,jacomat.T),overwrite_a=True)
        del jacomat
    
        # we cut off the smallest values according to the remaining number of scaling invariants.
        # 2 values are cut for each mode, and 1 for dispersion.
        cutoff = 2 * self.M + 1

        s[:-cutoff] = 1 / s[:-cutoff]
        s[-cutoff:] = 0
        #w[:cutoff] = 0
        #w[cutoff:] = 1/sqrt(w[cutoff:])

        u = dot(u, diag(s))
        # v = dot(v,diag(w))
        # print dot(u,u.T) / dot(v,v.T)

        # we now have
        # sigma_rho^2 = A.T v v.T A = A.T u u.T A
    
        # compute variance and related quantities
        Dev_res = self.matrix - self.response_matrix()
        Dev_subdispers = self.matrix - self.response_matrix(dispersion=False)
        chisq = sum(Dev_res**2)
        # effective degrees of freedom D-D_inv, BE+d model
        dof = 2 * (self.A_km.size + self.R_jmw.size) + self.K + self.J * self.M - self.M - 1
        # variance formula (``Error computations and approximate Hessian'')
        vari = chisq * self.matrix.size / (self.matrix.size - dof)
    
        rmsin = sum(self.matrix**2)
        print('     input response RMS: %.3e %s' % (sqrt(rmsin / self.matrix.size),self.unit))
        s2n = sqrt(rmsin / chisq)
        print('     Expl. funct. ratio (EVR): %.3f' % s2n)
    
        xsigmas = empty(s.shape[0])
    
        sqv = sqrt(vari)
        print('     input sigma: %.2e %s' % (sqv,self.unit))
        for p in xrange(s.shape[0]):
            # rhoTv = u[p,:] #v[p,:]
            xsigmas[p] = sqrt(dot(u[p], u[p])) * sqv
        
        self.error._opt_unwrap(xsigmas)
    
        self.additional['err'] = {
            'chi_kjw': Dev_res,
            'chi^2': chisq,
            'variance': vari,
            's2n': s2n,
            'eigvals': s,
            'cutoff': cutoff}
    
        del xsigmas
    
        Re_idx = offs[1]  # index of Re(R_jmw)
        Im_idx = Re_idx + self.R_jmw.size
        for j in xrange(self.J):
            for m in xrange(self.M):
                for w in xrange(self.M):
                    # compute beta error
                    A_Re = 2 * self.R_jmw[j,m,w].real
                    A_im = 2 * self.R_jmw[j,m,w].imag
                    Atu = A_Re * u[Re_idx, :] + A_im * u[Im_idx, :]  # compute A.T u
                    self.error.cbeta_jmw[j, m, w] = sqrt(dot(Atu, Atu)) * sqv
    
                    # compute phi error in DEGREES
                    beta = self.cbeta_jmw
                    A_Re = -self.R_jmw[j,m,w].imag / beta[j, m, w]
                    A_im = self.R_jmw[j,m,w].real / beta[j, m, w]
                    Atu = A_Re * u[Re_idx,:] + A_im * u[Im_idx, :]  # compute A.T u
                    self.error.phi_jmw[j, m, w] = 180 * sqrt(dot(Atu, Atu)) * sqv / pi
    
                    Re_idx += 1
                    Im_idx += 1
    
        for j in range(1, self.J):
            self.error.delphi_jmw[j-1] = sqrt(self.error.phi_jmw[j]**2 + self.error.phi_jmw[j - 1]**2)
    
        #print("error check 3: %e" % sum(self.additional['err']['chi^2']))

    def save(self,filename):
        """
        save the Result object as a pickle file with the given filename.
        The object can be reloaded using :py:func:`cobea.load_result`
        (which simply uses pickle)
        """
        with open(filename,'w') as f:
            dump(self, f, protocol=2)
        print('Result saved in '+filename)
