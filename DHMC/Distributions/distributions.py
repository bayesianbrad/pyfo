import numpy as np
import torch
from torch.autograd import Variable
from Utils.core import ContinuousRandomVariable, DiscreteRandomVariable, VariableCast
# TO DO : comment seeds when running program for real.
# torch.manual_seed(1234) # For testing only
# np.random.seed(1234)

# Note to user: Only the Laplace pdf is implemented as that is that is needed for this
# current project. The code is written for the pdfs but not
# ---------------------------------------------------------------------
# CONTINUOUS DISTRIBUTIONS
# ---------------------------------------------------------------------


class Laplace(ContinuousRandomVariable):
    '''
    Laplace random variable

    Methods
    -------
    sample X ~ Laplace(location, scale)
    logpdf

    Attributes
    ----------
    location - Type torch.autograd.Variable, torch.Tensor, nparray
               Size \mathbb{R}^{1 x N}
    scale    - Type torch.autograd.Variable, torch.Tensor, nparray
               Size \mathbb{R}^{1 x N}
    '''
    def __init__(self, location, scale):
        self.location = VariableCast(location)
        self.scale    = VariableCast(scale)
    def sample(self):
        # https: // en.wikipedia.org / wiki / Laplace_distribution
        uniforms = torch.Tensor(self.location.size()).uniform_() - 0.5
        uniforms = VariableCast(uniforms)
        return self.location - self._scale * torch.sign(uniforms) * \
                                torch.log(1 - 2 * torch.abs(uniforms))

    def logpdf(self, value):
        return -torch.div(torch.abs(value - self._location),self._scale) - torch.log(2 * self._scale)

class Normal(ContinuousRandomVariable):
    """Normal random variable
    Returns  a normal distribution object class with mean - mean
    and standard deviation - std.

    Methods
    --------
    sample   - Returns a sample X ~ N(mean,std) as a Variable.
    pdf
    logpdf   -

    Attributes
    ----------
    mean     - Type: torch.autograd.Variable, torch.Tensor, nparray
               Size: \mathbb{R}^{1 x N}
    std      - Type: torch.autograd.Variable, torch.Tensor, nparray
               Size: \mathbb{R}^{1 x N}

    """
    def __init__(self, mean, std):
        self.mean = VariableCast(mean)
        self.std  = VariableCast(std)


    def sample(self, num_samples = 1):
        # x ~ N(0,1)
        self.unifomNorm = Variable(torch.randn(1,num_samples))
        return self.unifomNorm * (self.std) + self.mean

    def logpdf(self, value):
        value = VariableCast(value)
        # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
        return (-0.5 *  torch.pow(value - self.mean, 2) / self.std**2) -  torch.log(self.std)

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
       self.chol_std = VariableCast(torch.potrf(self.cov.data).t())  # lower triangle
       self.chol_std_inv = torch.inverse(self.chol_std)

   def sample(self, num_samples=1):
       zs = torch.randn(self.dim, 1)
       # print("zs", zs)
       # samples = Variable( self.mean.data + torch.matmul(self.chol_std.data, zs), requires_grad = True)
       return self.mean.data + torch.matmul(self.chol_std.data, zs)

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
    """
    Categorical over 0,...,N-1 with arbitrary probabilities, 1-dimensional rv, long type.
    """
    def __init__(self, p, p_min=1E-6):
        p =     VariableCast(p)
        self._p = torch.clamp(p, p_min)
    def sample(self):
        return self._p.multinomial(1, True)

    def logpmf(self, value):
        return torch.log(self._p.gather(1, value)).squeeze()

class Bernoulli(DiscreteRandomVariable):
    """bernoulli random variable

    Methods
    --------
    sample   - returns a sample X ~ Bern(0,1) as a Variable
    pdf
    logpdf   -

    Attributes
    ----------
    probabilty    - Type: torch.autograd.Variable, torch.Tensor, nparray
                    Size: \mathbb{R}^{1 x N}


    """

    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
        probability - Type: Float tensor
        """
        self.probability = VariableCast(probability)

    def sample(self, max = 1):
        ''' Generate random samples from a Bernoulli dist for given tensor of probabilities'''
        uniformInt  = torch.Tensor(self.probability.size()).uniform(0,max)
        sampCond    = uniformInt / max
        # is greater than p
        sample     = torch.gt(uniformInt, sampCond).float()
        return sample


    def logpmf(self, value, epsilon=1e-10):
        assert (value.size() == self._probability.size())
        value       = VariableCast(value)
        test        = value * torch.log(self.probability + epsilon) +\
                      (1 - value) * torch.log(1 - self.probability + epsilon)
        print(test)
        # return torch.sum(
        #         value * torch.log(self.probability + epsilon) +
        #         (1 - value) * torch.log(1 - self.probability + epsilon))




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