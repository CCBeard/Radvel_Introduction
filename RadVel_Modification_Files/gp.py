import sys
import radvel
import scipy
from scipy import spatial
import abc
import numpy as np
import warnings

warnings.simplefilter('once')

# implemented kernels & examples of their associated hyperparameters
KERNELS = {
    "SqExp": ['gp_length', 'gp_amp'],
    "Per": ['gp_per', 'gp_length', 'gp_amp'],
    "QuasiPer": ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp'],
    "Celerite": ['gp_B', 'gp_C', 'gp_L', 'gp_Prot'],
    "Chromatic_1": ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amplist'],
    "Chromatic_2": ['gp_per', 'gp_perlength', 'gp_explength', 'gp_amp_0', 'gp_amp_lambda'],
    }

if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC


# celerite is an optional dependency
def _try_celerite():
    try:
        import celerite
        from celerite.solver import CholeskySolver
        return True
    except ImportError:
        warnings.warn("celerite not installed. GP kernals using celerite will not work. \
Try installing celerite using 'pip install celerite'", ImportWarning)
        return False


_has_celerite = _try_celerite()
if _has_celerite:
    import celerite
    from celerite.solver import CholeskySolver


class Kernel(ABC):
    """
    Abstract base class to store kernel info and compute covariance matrix.
    All kernel objects inherit from this class.

    Note:
        To implement your own kernel, create a class that inherits
        from this class. It should have hyperparameters that follow
        the name scheme 'gp_NAME_SUFFIX'.

    """

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def compute_distances(self, x1, x2):
        pass

    @abc.abstractmethod
    def compute_covmatrix(self, errors):
        pass


class SqExpKernel(Kernel):
    """
    Class that computes and stores a squared exponential kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly two objects, 'gp_length*'
            and 'gp_amp*', where * is a suffix identifying
            these hyperparameters with a likelihood object.

    """

    @property
    def name(self):
        return "SqExp"

    def __init__(self, hparams):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_length'):
                self.hparams['gp_length'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams['gp_amp'] = hparams[par]

        assert len(hparams) == 2, \
            "SqExpKernel requires exactly 2 hyperparameters with names" \
            + "'gp_length*' and 'gp_amp*'."

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
        except KeyError:
            raise KeyError("SqExpKernel requires hyperparameters 'gp_length*'" \
                           + " and 'gp_amp*'.")
        except AttributeError:
            raise AttributeError("SqExpKernel requires dictionary of" \
                                 + " radvel.Parameter objects as input.")

    def __repr__(self):
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        return "SqExp Kernel with length: {}, amp: {}".format(length, amp)

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature.
        """
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value

        K = amp**2 * scipy.exp(-self.dist/(length**2))

        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError: # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix


class PerKernel(Kernel):
    """
    Class that computes and stores a periodic kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -\\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3^2 } ) }{ 2\\eta_2^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly three objects, 'gp_length*',
            'gp_amp*', and 'gp_per*', where * is a suffix identifying
            these hyperparameters with a likelihood object.

    """

    @property
    def name(self):
        return "Per"

    def __init__(self, hparams):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_length'):
                self.hparams['gp_length'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams['gp_amp'] = hparams[par]
            if par.startswith('gp_per'):
                self.hparams['gp_per'] = hparams[par]

        assert len(hparams) == 3, \
            "PerKernel requires exactly 3 hyperparameters with names 'gp_length*'," \
            + " 'gp_amp*', and 'gp_per*'."

        try:
            self.hparams['gp_length'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
        except KeyError:
            raise KeyError("PerKernel requires hyperparameters 'gp_length*'," \
                           + " 'gp_amp*', and 'gp_per*'.")
        except AttributeError:
            raise AttributeError("PerKernel requires dictionary of " \
                                 + "radvel.Parameter objects as input.")

    def __repr__(self):
        length = self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        return "Per Kernel with length: {}, amp: {}, per: {}".format(
            length, amp, per
        )

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist = scipy.spatial.distance.cdist(X1, X2, 'euclidean')

    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature.
        """
        length= self.hparams['gp_length'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value

        K = amp**2 * scipy.exp(-np.sin(np.pi*self.dist/per)**2. / (2.*length**2))
        self.covmatrix = K
        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix

class QuasiPerKernel(Kernel):
    """
    Class that computes and stores a quasi periodic kernel matrix.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_1^2 * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } -
                 \\frac{ \\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3 } ) }{ 2\\eta_4^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly four objects, 'gp_explength*',
            'gp_amp*', 'gp_per*', and 'gp_perlength*', where * is a suffix
            identifying these hyperparameters with a likelihood object.

    """
    @property
    def name(self):
        return "QuasiPer"

    def __init__(self, hparams):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_perlength'):
                self.hparams['gp_perlength'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams['gp_amp'] = hparams[par]
            if par.startswith('gp_per') and not 'length' in par:
                self.hparams['gp_per'] = hparams[par]
            if par.startswith('gp_explength'):
                self.hparams['gp_explength'] = hparams[par]

        assert len(hparams) == 4, \
            "QuasiPerKernel requires exactly 4 hyperparameters with names" \
            + " 'gp_perlength*', 'gp_amp*', 'gp_per*', and 'gp_explength*'."

        try:
            self.hparams['gp_perlength'].value
            self.hparams['gp_amp'].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except KeyError:
            raise KeyError("QuasiPerKernel requires hyperparameters" \
                           + " 'gp_perlength*', 'gp_amp*', 'gp_per*', " \
                           + "and 'gp_explength*'.")
        except AttributeError:
            raise AttributeError("QuasiPerKernel requires dictionary of" \
                                 + " radvel.Parameter objects as input.")

    def __repr__(self):
        perlength = self.hparams['gp_perlength'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        msg = (
            "QuasiPer Kernel with amp: {}, per length: {}, per: {}, "
            "exp length: {}"
        ).format(amp, perlength, per, explength)
        return msg

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature.
        """
        perlength = self.hparams['gp_perlength'].value
        amp = self.hparams['gp_amp'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        K = np.array(amp**2
                     * scipy.exp(-self.dist_se/(explength**2))
                     * scipy.exp((-np.sin(np.pi*self.dist_p/per)**2.) / (2.*perlength**2)))

        self.covmatrix = K

        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix


class CeleriteKernel(Kernel):
    """
    Class that computes and stores a matrix approximating the quasi-periodic
    kernel.

    See `radvel/example_planets/k2-131_celerite.py` for an example of a setup
    file that uses this Kernel object.

    See celerite.readthedocs.io and Foreman-Mackey et al. 2017. AJ, 154, 220
    (equation 56) for more details.

    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = B/(2+C) * exp( -|t_i - t_j| / L) * (\\cos(\\frac{ 2\\pi|t_i-t_j| }{ P_{rot} }) + (1+C) )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly four objects, 'gp_B*',
            'gp_C*', 'gp_L*', and 'gp_Prot*', where * is a suffix
            identifying these hyperparameters with a likelihood object.
    """

    @property
    def name(self):
        return "Celerite"

    def __init__(self, hparams):

        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_B'):
                self.hparams['gp_B'] = hparams[par]
            if par.startswith('gp_C'):
                self.hparams['gp_C'] = hparams[par]
            if par.startswith('gp_L'):
                self.hparams['gp_L'] = hparams[par]
            if par.startswith('gp_Prot'):
                self.hparams['gp_Prot'] = hparams[par]

        assert len(self.hparams) == 4, """
CeleriteKernel requires exactly 4 hyperparameters with names 'gp_B', 'gp_C', 'gp_L', and 'gp_Prot'.
        """

        try:
            self.hparams['gp_Prot'].value
            self.hparams['gp_C'].value
            self.hparams['gp_B'].value
            self.hparams['gp_L'].value
        except KeyError:
            raise KeyError("""
CeleriteKernel requires hyperparameters 'gp_B*', 'gp_C*', 'gp_L', and 'gp_Prot*'.
                """)
        except AttributeError:
            raise AttributeError("CeleriteKernel requires dictionary of radvel.Parameter objects as input.")

    # get arrays of real and complex parameters
    def compute_real_and_complex_hparams(self):

        self.real = np.zeros((1, 4))
        self.complex = np.zeros((1, 4))

        B = self.hparams['gp_B'].value
        C = self.hparams['gp_C'].value
        L = self.hparams['gp_L'].value
        Prot = self.hparams['gp_Prot'].value

        # Foreman-Mackey et al. (2017) eq 56
        self.real[0,0] = B*(1+C)/(2+C)
        self.real[0,2] = 1/L
        self.complex[0,0] = B/(2+C)
        self.complex[0,1] = 0.
        self.complex[0,2] = 1/L
        self.complex[0,3] = 2*np.pi/Prot

    def __repr__(self):

        B = self.hparams['gp_B'].value
        C = self.hparams['gp_C'].value
        L = self.hparams['gp_L'].value
        Prot = self.hparams['gp_Prot'].value

        msg = (
            "Celerite Kernel with B = {}, C = {}, L = {}, Prot = {}."
        ).format(B, C, L, Prot)
        return msg

    def compute_distances(self, x1, x2):
        """
        The celerite.solver.CholeskySolver object does
        not require distances to be precomputed, so
        this method has been co-opted to define some
        unchanging variables.
        """
        self.x = x1

        # blank matrices (corresponding to Cholesky decomp of kernel) needed for celerite solver
        self.A = np.empty(0)
        self.U = np.empty((0,0))
        self.V = self.U


    def compute_covmatrix(self, errors):
        """ Compute the Cholesky decomposition of a celerite kernel

            Args:
                errors (array of float): observation errors and jitter added
                    in quadrature

            Returns:
                celerite.solver.CholeskySolver: the celerite solver object,
                with Cholesky decomposition computed.
        """
        # initialize celerite solver object
        solver = CholeskySolver()

        self.compute_real_and_complex_hparams()
        solver.compute(
            0., self.real[:,0], self.real[:,2],
            self.complex[:,0], self.complex[:,1],
            self.complex[:,2], self.complex[:,3],
            self.A, self.U, self.V,
            self.x, errors**2
        )

        return solver


class Chromatic_1Kernel(Kernel):
    """
    Class that computes and stores a quasi periodic kernel matrix. Modified to have a wavelength dependence
    from Cale et al. 2021. This is kernel "KJ1" in the paper.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_{\sigma,s(i)}\\eta_{\sigma,s(j)} * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } -
                 \\frac{ \\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3^2 } ) }{ \\eta_4^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Contains a variable numper of objects, 'gp_explength*',
            'gp_per*', 'gp_perlength*', and a 'gp_amp_instrument' term for every instrument where * is a suffix
            identifying these hyperparameters with a likelihood object.

    """
    @property
    def name(self):
        return "Chromatic_1Kernel"

    def __init__(self, hparams, telnames):
        self.covmatrix = None
        self.hparams = {}
        self.telnames = telnames
        for par in hparams:
            if par.startswith('gp_perlength'):
                self.hparams['gp_perlength'] = hparams[par]
            if par.startswith('gp_amp'):
                self.hparams[par] = hparams[par]
            if par.startswith('gp_per') and not 'length' in par:
                self.hparams['gp_per'] = hparams[par]
            if par.startswith('gp_explength'):
                self.hparams['gp_explength'] = hparams[par]

        assert len(hparams) == 3 + len(self.telnames), \
            "Chromatic Kernel 1 requires exactly "+ str(3+len(self.telnames))+" hyperparameters with names" \
            + " 'gp_perlength*', 'gp_amp_instrument*', 'gp_per*', and 'gp_explength*'."

        try:
            self.hparams['gp_perlength'].value
            for tel in self.telnames:
                self.hparams['gp_amp_'+tel].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except KeyError:
            raise KeyError("Chromatic Kernel requires hyperparameters" \
                           + " 'gp_perlength*', 'gp_amp_instrument*', 'gp_per*', " \
                           + "and 'gp_explength*'.")
        except AttributeError:
            raise AttributeError("ChromaticKernel 1 requires dictionary of" \
                                 + " radvel.Parameter objects as input.")

    def __repr__(self):
        perlength = self.hparams['gp_perlength'].value
        amplist = []
        for tel in self.telnames:
            amplist.append(self.hparams['gp_amp_'+tel].value)
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        msg = (
            "Chromatic Kernel 1 with amp: {}, per length: {}, per: {}, "
            "exp length: {}"
        ).format(amplist, perlength, per, explength)
        return msg

    def compute_distances(self, x1, x2):

        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
        self.D1 = len(X1)
        self.D2 = len(X2)

    def get_amplitudes(self, tel=None, inst=None):


        if tel is not None:
            self.N = len(tel)
            amp_train = []
            for i in range(len(tel)):
                amp_train.append(self.hparams['gp_amp_'+tel[i]].value)
            self.amp_train = np.array(amp_train)
        if self.D1 == self.N:
            self.amp_i = self.amp_train
        else:
            self.amp_i = np.repeat(self.hparams['gp_amp_'+inst].value, self.D1)
        if self.D2 == self.N:
            self.amp_j = self.amp_train
        else:
            self.amp_j = np.repeat(self.hparams['gp_amp_'+inst].value, self.D2)



    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature.
        """
        perlength = self.hparams['gp_perlength'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value


        K = np.array(np.outer(self.amp_i, self.amp_j)
                     * np.exp(-self.dist_se/(2*explength**2))
                     * np.exp((-np.sin(np.pi*self.dist_p/per)**2.) / (2*perlength**2)))

        self.covmatrix = K

        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix


class Chromatic_2Kernel(Kernel):
    """
    Class that computes and stores a quasi periodic kernel matrix. Modified to have a wavelength dependence
    from Cale et al. 2021.
    An arbitrary element, :math:`C_{ij}`, of the matrix is:

    .. math::

        C_{ij} = \\eta_0^2 * (lambda_0/\\sqrt(lambda_i * lambda_j))^(2*eta_lambda) * exp( \\frac{ -|t_i - t_j|^2 }{ \\eta_2^2 } -
                 \\frac{ \\sin^2(\\frac{ \\pi|t_i-t_j| }{ \\eta_3^2 } ) }{ \\eta_4^2 } )

    Args:
        hparams (dict of radvel.Parameter): dictionary containing
            radvel.Parameter objects that are GP hyperparameters
            of this kernel. Must contain exactly five objects, 'gp_explength*',
            'gp_amp0*', 'gp_amplambda*' 'gp_per*', and 'gp_perlength*', where * is a suffix
            identifying these hyperparameters with a likelihood object.

    """
    @property
    def name(self):
        return "Chromatic_2Kernel"

    def __init__(self, hparams, telnames):
        self.covmatrix = None
        self.hparams = {}
        for par in hparams:
            if par.startswith('gp_perlength'):
                self.hparams['gp_perlength'] = hparams[par]
            if par.startswith('gp_amp0'):
                self.hparams['gp_amp0'] = hparams[par]
            if par.startswith('gp_amplambda'):
                self.hparams['gp_amplambda'] = hparams[par]
            if par.startswith('gp_per') and not 'length' in par:
                self.hparams['gp_per'] = hparams[par]
            if par.startswith('gp_explength'):
                self.hparams['gp_explength'] = hparams[par]

        assert len(hparams) == 5, \
            "Chromatic Kernel 2 requires exactly 5 hyperparameters with names" \
            + " 'gp_perlength*', 'gp_amp0*', 'gp_amplambda*', 'gp_per*', and 'gp_explength*'."

        try:
            self.hparams['gp_perlength'].value
            self.hparams['gp_amp0'].value
            self.hparams['gp_amplambda'].value
            self.hparams['gp_per'].value
            self.hparams['gp_explength'].value
        except KeyError:
            raise KeyError("Chromatic Kernel 2 requires hyperparameters" \
                           + " 'gp_perlength*', 'gp_amp0*', 'gp_amplambda*', 'gp_per*', " \
                           + "and 'gp_explength*'.")
        except AttributeError:
            raise AttributeError("ChromaticKernel requires dictionary of" \
                                 + " radvel.Parameter objects as input.")

    def __repr__(self):
        perlength = self.hparams['gp_perlength'].value
        amp_0 = self.hparams['gp_amp0'].value
        amp_lambda = self.hparams['gp_amplambda'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value

        msg = (
            "Chromatic Kernel 2 with amp0: {}, amplambda: {}, per length: {}, per: {}, "
            "exp length: {}"
        ).format(amp_0, amp_lambda, perlength, per, explength)
        return msg

    def compute_distances(self, x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        self.dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        self.dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
        self.D1 = len(X1)
        self.D2 = len(X2)

    def get_wavelengths(self, tel=None, inst=None):
        wav = {'hires':565, 'tres':650, 'carmenesvis':750, 'carmenesnir':1350, 'ishell':2350, 'spirou':1650,
                'hpf':1350, 'NEID':750, 'hires_pre':565, 'hires_post':565,'carmenes':750, 'apf':565, 'harpsn':750,
                'hires_j':565,'PFS_pre':565,'PFS_post':565, 'harps':658, 'HARPS':658}
        if tel is not None:
            self.N = len(tel)
            lambda_train = []
            for i in range(self.N):
                lambda_train.append(wav[tel[i]])
            self.lambda_train = np.array(lambda_train)
        if self.D1 == self.N:
            self.lambda_i = self.lambda_train
        else:
            self.lambda_i = np.repeat(wav[inst], self.D1)

        if self.D2 == self.N:
            self.lambda_j = self.lambda_train
        else:
            self.lambda_j = np.repeat(wav[inst], self.D2)



    def compute_covmatrix(self, errors):
        """ Compute the covariance matrix, and optionally add errors along
            the diagonal.

            Args:
                errors (float or numpy array): If covariance matrix is non-square,
                    this arg must be set to 0. If covariance matrix is square,
                    this can be a numpy array of observational errors and jitter
                    added in quadrature.
        """
        perlength = self.hparams['gp_perlength'].value
        amp0 = self.hparams['gp_amp0'].value
        amplambda = self.hparams['gp_amplambda'].value
        per = self.hparams['gp_per'].value
        explength = self.hparams['gp_explength'].value
        lambda_0 = 565 #arbitrary wavelength, in nm


        K = np.array(amp0**2
                     * (lambda_0/np.sqrt(np.outer(self.lambda_i, self.lambda_j)))**(2*amplambda)
                     * np.exp(-self.dist_se/(2*explength**2))
                     * np.exp((-np.sin(np.pi*self.dist_p/per)**2.) / (2*perlength**2)))

        self.covmatrix = K

        # add errors along the diagonal
        try:
            self.covmatrix += (errors**2) * np.identity(K.shape[0])
        except ValueError:  # errors can't be added along diagonal to a non-square array
            pass

        return self.covmatrix
