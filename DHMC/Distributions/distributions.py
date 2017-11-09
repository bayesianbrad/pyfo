import numpy as np
import torch
from torch.autograd import Variable
from Utils.core import ContinuousRandomVariable, DiscreteRandomVariable, VariableCast

# ---------------------------------------------------------------------
# CONTINUOUS DISTRIBUTIONS
# ---------------------------------------------------------------------

class Normal(ContinuousRandomVariable):
    """1d Normal random variable"""
    def __init__(self, mean, std):
        """Initialize this distribution with mean, variance.

        input:
            mean:
            std:  standard deviation
        """

        self.mean = VariableCast(mean)
        if len(mean.data.shape) == 1:
            self.mean = mean.unsqueeze(1)
        self.std = VariableCast(std)


    def sample(self, num_samples = 1):
        x = torch.randn(1)
        sample = Variable(x * self.std.data + self.mean.data, requires_grad = True)
        return sample #.detach()


    def logpdf(self, value):
        mean = self.mean
        var = self.std ** 2
        value = VariableCast(value)
        if len(value.data.shape) == 1:
            value = value.unsqueeze(1)
        if isinstance(mean, torch.IntTensor):
            mean = mean.type(torch.FloatTensor)
        # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
        return (-0.5 * torch.pow(value - mean, 2) / var - 0.5 * torch.log(2 * var * np.pi))


class MultivariateNormal(ContinuousRandomVariable):
    """Normal random variable"""
    def __init__(self, mean, cov):
        """Initialize this distribution with mean, cov.

        input:
            mean: n by 1
            cov: covariance matrix, n by n
        """
        self.mean = VariableCast(mean)
        self.cov = VariableCast(cov)
        assert self.mean.data.size()[0] == self.cov.data.size()[0] #, "ERROR! mean and cov have different size!")
        self.dim = self.mean.data.size()[0]
        self.chol_std = VariableCast(torch.t(torch.potrf(self.cov.data)))  # lower triangle
        self.chol_std_inv = torch.inverse(self.chol_std)

    def sample(self, num_samples=1 ):
        zs = torch.randn(self.dim, 1)
        # print("zs", zs)
        samples = Variable( self.mean.data + torch.matmul(self.chol_std.data, zs), requires_grad = True)
        return samples

    def logpdf(self, value):
        """
        value : obs value, should be n by 1
        :return: scalar, log pdf value
        """
        value = VariableCast(value)
        cov_det = self.chol_std.diag().prod() ** 2
        log_norm_constant = 0.5 * self.dim * torch.log(torch.Tensor([2 * np.pi])) \
                            + 0.5*torch.log(cov_det.data)
        right = torch.matmul( self.chol_std_inv, value - self.mean)
        # print(value, self.mean, value - self.mean)
        log_p = - Variable(log_norm_constant) - 0.5 * torch.matmul(torch.t(right), right)
        return log_p



# class Laplace(ContinuousRandomVariable):
#     '''
#     Laplace random variable
#
#     Methods
#     -------
#     sample X ~ Laplace(location, scale)
#     logpdf
#
#     Attributes
#     ----------
#     location - Type torch.autograd.Variable, torch.Tensor, nparray
#                Size \mathbb{R}^{1 x N}
#     scale    - Type torch.autograd.Variable, torch.Tensor, nparray
#                Size \mathbb{R}^{1 x N}
#     '''
#     def __init__(self, location, scale):
#         self.location = VariableCast(location)
#         self.scale    = VariableCast(scale)
#     def sample(self):
#         # https: // en.wikipedia.org / wiki / Laplace_distribution
#         uniforms = torch.Tensor(self.location.size()).uniform_() - 0.5
#         uniforms = VariableCast(uniforms)
#         return self.location - self._scale * torch.sign(uniforms) * \
#                                 torch.log(1 - 2 * torch.abs(uniforms))
#
#     def logpdf(self, value):
#         return -torch.div(torch.abs(value - self._location),self._scale) - torch.log(2 * self._scale)
#
# class Normal(ContinuousRandomVariable):
#     """Normal random variable
#     Returns  a normal distribution object class with mean - mean
#     and standard deviation - std.
#
#     Methods
#     --------
#     sample   - Returns a sample X ~ N(mean,std) as a Variable.
#     pdf
#     logpdf   -
#
#     Attributes
#     ----------
#     mean     - Type: torch.autograd.Variable, torch.Tensor, nparray
#                Size: \mathbb{R}^{1 x N}
#     std      - Type: torch.autograd.Variable, torch.Tensor, nparray
#                Size: \mathbb{R}^{1 x N}
#
#     """
#     def __init__(self, mean, std):
#         self.mean = VariableCast(mean)
#         self.std  = VariableCast(std)
#
#
#     def sample(self, num_samples = 1):
#         # x ~ N(0,1)
#         self.unifomNorm = Variable(torch.randn(1,num_samples))
#         return self.unifomNorm * (self.std) + self.mean
#
#     def logpdf(self, value):
#         value = VariableCast(value)
#         # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
#         return (-0.5 *  torch.pow(value - self.mean, 2) / self.std**2) -  torch.log(self.std)
#
# class MultivariateNormal(ContinuousRandomVariable):
#    """Normal random variable"""
#    def __init__(self, mean, cov):
#        """Initialize this distribution with mean, cov.
#
#        input:
#            mean: n by 1
#            cov: covariance matrix, n by n
#        """
#        self.mean = VariableCast(mean)
#        self.cov = VariableCast(cov)
#        assert self.mean.data.size()[0] == self.cov.data.size()[0] #, "ERROR! mean and cov have different size!")
#        self.dim = self.mean.data.size()[0]
#        self.chol_std = VariableCast(torch.potrf(self.cov.data).t())  # lower triangle
#        self.chol_std_inv = torch.inverse(self.chol_std)
#
#    def sample(self, num_samples=1):
#        zs = torch.randn(self.dim, 1)
#        # print("zs", zs)
#        # samples = Variable( self.mean.data + torch.matmul(self.chol_std.data, zs), requires_grad = True)
#        return self.mean.data + torch.matmul(self.chol_std.data, zs)
#
#    def logpdf(self, value):
#        """
#        value : obs value, should be n by 1
#        :return: scalar, log pdf value
#        """
#        value = VariableCast(value)
#        cov_det = self.chol_std.diag().prod() ** 2
#        log_norm_constant = 0.5 * self.dim * torch.log(torch.Tensor([2 * np.pi])) \
#                            + 0.5*torch.log(cov_det.data)
#        right = torch.matmul( self.chol_std_inv, value - self.mean)
#        # print(value, self.mean, value - self.mean)
#        log_p = - Variable(log_norm_constant) - 0.5 * torch.matmul(torch.t(right), right)
#        return log_p

# class MultivariateNormal(ContinuousRandomVariable):
#     """MultivariateIndependentNormal simple class
#     Returns  a normal distribution object class with mean - mean
#     and standard deviation - std.
#
#     Methods
#     --------
#     sample   - Returns a sample X ~ N(mean,std) as a Variable. Takes an additional
#                argument grad. If we need the differential of X ~ N(mu, var)
#     pdf
#     logpdf   -
#
#     Attributes
#     ----------
#     mean        -  Type: torch.Tensor, Variable, ndarray
#                    Size: [ 1, ...., N]
#     covariance  -  Type: torch.Tensor, Variable, ndarray
#                     Size: \mathbb{R}^{N \times N}
#     """
#
#     def __init__(self, mean, covariance):
#         """Initialize this distribution with given mean and covariance.
#         """
#         assert (mean.size()[0] == covariance.size()[0])
#         assert (mean.size()[0] == covariance.size()[1])
#         self.mean      = VariableCast(mean)
#         self.covariance = VariableCast(covariance)
#         # cholesky decomposition returns upper triangular matrix. Will not accept Variables
#         self.L = VariableCast(torch.potrf(self.covariance.data))
#
#     def sample(self):
#         # Returns a sample of a multivariate normal X ~ N(mean, cov)
#         # A column vecotor of X ~ N(0,I)  - IS THIS ACTUALLY TRUE ?? NOT SURE, ALL POINTS HAVE
#         # THE RIGHT MEAN AND VARIANCE N(0,1)
#         self.uniformNorm  = torch.Tensor(self.mean.size()).normal_()
#         samples           = self.mean.data  + Variable(self.L.t().data.mm(self.uniformNorm))
#         return samples
#
#     # def sample_grad(self):
#     #     x          = Variable(self.uniformNorm.data, requires_grad = True)
#     #     logSample  = self.uniformNorm * (self.L)
#     #     sampleGrad = torch.autograd.grad([logSample],[x], grad_outputs= torch.ones(x.size()))[0]
#     #     return sampleGrad
#     # def pdf(self, value):
#     #     assert (value.size() == self._mean.size())
#     #     # CAUTION: If the covariance is 'Unknown' then we will
#     #     # not be returned the correct derivatives.
#     #     print('****** Warning ******')
#     #     print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
#     #     WILL NOT RECORD THE GRAPH STRUCTURE OF THE FULL PASS, ONLY THE CALCUALTION OF THE PDF')
#     #     value = VariableCast(value)
#     #     # the sqrt root of a det(cov) : sqrt(det(cov)) == det(L.t()) = \Pi_{i=0}^{N} L_{ii}
#     #     self._constant = torch.pow(2 * np.pi, value.size()[1]) * self._L.t().diag().prod()
#     #     return self._constant * torch.exp(
#     #         -0.5 * (value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm(
#     #             (value - self._mean).t()))
#
#     def logpdf(self, value):
#         print('****** Warning ******')
#         print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
#         WILL NOT RECORD THE GRAPH STRUCTURE OF THE FULL PASS, ONLY THE CALCUALTION OF THE LOGPDF')
#         assert (value.size() == self.mean.size())
#         value = VariableCast(value)
#         self._constant = ((2 * np.pi) ** value.size()[1]) * self.L.t().diag().prod()
#         return torch.log(-0.5 * (value - self.mean).t().mm(self.L.inverse().mm(self.L.inverse().t())).mm(
#             (value - self.mean))) \
#                + self._constant
#

# ---------------------------------------------------------------------
# DISCRETE DISTRIBUTIONS
# ---------------------------------------------------------------------
class Categorical(DiscreteRandomVariable):
    """categorical Normal random variable"""

    def __init__(self, p):
        """Initialize this distribution with p =[p0, p1, ..., pn].

        input:
            mean:
            std:  standard deviation

        output:
            integer in [1, ..., n]
        """

        self.p = VariableCast(p)

    def sample(self):
        onedraw = np.random.multinomial(1, self.p.data.numpy())
        index = np.argwhere(onedraw == 1)[0,0]
        var = Variable(torch.Tensor([int(index)]) +1 ,requires_grad = True)
        return var

    def logpdf(self, value):
        int_value =  int(value.data.numpy())
        index = int_value -1
        if 1 <= int_value <= self.p.data.shape[0]:
            return torch.log(self.p[index])
        else:
            return torch.Tensor([-np.inf])

class Categorical_Trans():
    """categorical Tranformed random variable"""

    def __init__(self, p, method=None):
        """Initialize this distribution with p =[p0, p1, ..., pn].

        input:
            mean:
            std:  standard deviation

        output:
            integer in [1, ..., n]
        """

        self.p = VariableCast(p)
        if method is None:
            self.method = "standard"
        else:
            self.method = method

    def logpdf(self, value):  # logp: 1*1
        if self.method == "standard":

            value =  VariableCast(value)
            if len(value.data.shape) == 1:
                value = value.unsqueeze(1)

            int_value = int(torch.floor(value.data)[0,0])
            index = int_value - 1

            #returning logp is [-0.93838], wrapped by tensor
            if 1 <= value.data[0,0] <= self.p.data.shape[0] + 1:
                logp = torch.log(self.p[index])   # grad does not survive through this embedding
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            else:
                return Variable(torch.Tensor([[-np.inf]]))
        else:
            raise ValueError("implement categorical transformed method")
            return 0


class Binomial_Trans(DiscreteRandomVariable):
    """ binomial distribution, into contnuous space
       discrete distribution does not support grad for now
    """

    def __init__(self, n, p, method=None):
        """Initialize this distribution with
        :parameter
         n; N_0, non-negative integar
         p: [0, 1]

        output:
            integer in [0, 1, ..., n]
        """

        self.p = VariableCast(p)
        self.n = VariableCast(n)
        if method is not None:
            self.method = method
        else:
            self.method = "standard"

    def logpdf(self, k):  # logp: 1*1
        k = VariableCast(k)
        n = self.n
        p = self.p
        if len(k.data.shape) == 1:
            k = k.unsqueeze(1)
        if len(self.n.data.shape) == 1:
            n = n.unsqueeze(1)
        if len(self.p.data.shape) == 1:
            p = p.unsqueeze(1)

        if self.method == "standard":
            int_k = int(torch.floor(k.data)[0,0])
            int_n = int(torch.floor(n.data)[0,0])
            np_p = p.data[0,0]

            #returning logp is [-0.93838], wrapped by tensor
            if 0 <= int_k <= int_n:
                logpmf = ss.binom.logpmf(int_k, int_n, np_p)
                logp = Variable(torch.Tensor([logpmf]))
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            else:
                return Variable(torch.Tensor([[-np.inf]]))
        else:
            raise ValueError("implement categorical transformed method")
            return 0


class Bernoulli_Trans(DiscreteRandomVariable):
    """ bernoulli distribution, into contnuous space
       discrete distribution does not support grad for now
    """

    def __init__(self, p, method=None):
        """Initialize this distribution with
        :parameter
         p: [0, 1]
        """

        self.p = VariableCast(p)
        if len(self.p.data.shape) == 1:
            self.p = self.p.unsqueeze(1)
        if method is not None:
            self.method = method
        else:
            self.method = "standard"
    def sample(self):
        """
        :return: x in [0,1], [1, 2]
        """
        x = torch.bernoulli(self.p) + Variable(torch.rand(1))
        if len(x.data.shape) == 1:
            x = x.unsqueeze(1)
        return x

    def logpdf(self, x):  # logp: 1*1
        x = VariableCast(x)
        p = self.p
        if len(x.data.shape) == 1:
            x = x.unsqueeze(1)

        if self.method == "standard":

            #returning logp is 1 by 1
            if 0 <= x.data[0,0] < 1:
                logp = torch.log(1-p)
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            elif 1 <= x.data[0, 0] < 2:
                logp = torch.log(p)
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            else:
                return Variable(torch.Tensor([[-np.inf]]))
        else:
            raise ValueError("implement categorical transformed method")
            return 0



# ---------------------------------------------------------------------------------------
# Unused and maybe used in the future
# ---------------------------------------------------------------------------------------
# class MultivariateIndependentLaplace(ContinuousRandomVariable):
#     """MultivariateIndependentLaplace random variable"""
#     def __init__(self, location, scale):
#         """Initialize this distribution with location, scale.
#
#         input:
#             location: Tensor/Variable
#                 [ 1, ..., N]
#             scale: Tensor/Variable
#                 [1, ..., N]
#         """
#         self._location = location
#         self._scale = scale
#
#     def sample(self, batch_size, num_particles):
#         uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
#         if isinstance(self._location, Variable):
#             uniforms = Variable(uniforms)
#             return self._location.detach() - self._scale.detach() * \
#                 torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
#         else:
#             return self._location - self._scale * torch.sign(uniforms) * \
#                 torch.log(1 - 2 * torch.abs(uniforms))
#     def sample(self,num_samples):
#         uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
#         # why the half that would make the scale between [-0.5,0.5]
#         if isinstance(self._location, Variable):
#             uniforms = Variable(uniforms)
#             return self._location.detach() - self._scale.detach() * \
#                 torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
#         else:
#             return self._location - self._scale * torch
#     def sample_reparameterized(self, num_samples):
#
#         standard_laplace = MultivariateIndependentLaplace(
#             location=VariableCast(torch.zeros(self._location.size())),
#             scale=VariableCast(torch.ones(self._scale.size()))
#         )
#
#         return self._location + self._scale * standard_laplace.sample(num_samples)
#         )
#
#     def pdf(self, value, batch_size, num_particles):
#         assert(value.size() == self._location.size())
#         assert(list(self._location.size()[:2]) == [batch_size, num_particles])
#
#         return torch.prod(
#             (
#                 torch.exp(-torch.abs(value - self._location) / self._scale) /
#                 (2 * self._scale)
#             ).view(batch_size, num_particles, -1),
#             dim=2
#         ).squeeze(2)
#
#     def logpdf(self, value, batch_size, num_particles):
#         assert(value.size() == self._location.size())
#         assert(list(self._location.size()[:2]) == [batch_size, num_particles])
#
#         return torch.sum(
#             (
#                 -torch.abs(value - self._location) /
#                 self._scale - torch.log(2 * self._scale)
#             ).view(batch_size, num_particles, -1),
#             dim=2
#         ).squeeze(2)

# class MultivariateNormal(ContinuousRandomVariable):
#     """MultivariateIndependentNormal simple class"""
#     def __init__(self, mean, covariance):
#         """Initialize this distribution with mean, covariance.
#
#         input:
#             mean: Tensor/Variable
#                 [ dim_1, ..., dim_N]
#             covariance: Tensor/Variable
#                 covariance \in \mathbb{R}^{N \times N}
#         """
#         assert(mean.size()[0] == covariance.size()[0])
#         assert (mean.size()[0] == covariance.size()[1])
#         self._mean = mean
#         self._covariance = covariance
#         # cholesky decomposition returns upper triangular matrix. Will not accept Variables
#         self._L = torch.potrf(self._covariance.data)
#     def sample(self):
#         # Returns a sample of a multivariate normal X ~ N(mean, cov)
#         # A column vecotor of X ~ N(0,I)
#         uniform_normals = torch.Tensor(self._mean.size()).normal_().t()
#
#         if isinstance(self._mean, Variable):
#             return self._mean.detach() + \
#                 Variable(self._L.t().mm(uniform_normals))
#         else:
#             return self._L.t().mm(uniform_normals) + self._mean
#
#     def pdf(self, value):
#         assert(value.size() == self._mean.size())
#         # CAUTION: If the covariance is 'Unknown' then we will
#         # not be returned the correct derivatives.
#         print('****** Warning ******')
#         print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
#         WILL NOT RECORD THE GRAPH STRUCTURE OF THE COVARIANCE' )
#         value = VariableCast(value)
#         # the sqrt root of a det(cov) : sqrt(det(cov)) == det(L.t()) = \Pi_{i=0}^{N} L_{ii}
#         self._constant = torch.pow(2*np.pi,value.size()[1]) * self._L.t().diag().prod()
#         return self._constant * torch.exp(-0.5*(value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm((value - self._mean).t()))
#         #     torch.prod(
#         #     (
#         #         1 / torch.sqrt(2 * self._variance * np.pi) * torch.exp(
#         #             -0.5 * (value - self._mean)**2 / self._variance
#         #         )
#         #     ).view(-1),
#         #     dim=0
#         # ).squeeze(0)
#     # squeeze doesn't do anything here, for our use.
#     # view(-1), infers to change the structure of the
#     # calculation, so it is transformed to a column vector
#     # dim = 0, implies that we take the products all down the
#     # rows