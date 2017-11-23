from __future__ import absolute_import, division, print_function

# distribution classes
from pyfo.distributions.bernoulli import Bernoulli
from pyfo.distributions.beta import Beta
from pyfo.distributions.categorical import Categorical
from pyfo.distributions.cauchy import Cauchy
from pyfo.distributions.delta import Delta
from pyfo.distributions.dirichlet import Dirichlet
from pyfo.distributions.distribution import Distribution  # noqa: F401
from pyfo.distributions.exponential import Exponential
from pyfo.distributions.gamma import Gamma
from pyfo.distributions.half_cauchy import HalfCauchy
from pyfo.distributions.log_normal import LogNormal
from pyfo.distributions.multinomial import Multinomial
from pyfo.distributions.normal import Normal
from pyfo.distributions.multivariate_normal import MultivariateNormal
from pyfo.distributions.poisson import Poisson
from pyfo.distributions.random_primitive import RandomPrimitive
from pyfo.distributions.uniform import Uniform

# function aliases
bernoulli = RandomPrimitive(Bernoulli)
beta = RandomPrimitive(Beta)
categorical = RandomPrimitive(Categorical)
cauchy = RandomPrimitive(Cauchy)
delta = RandomPrimitive(Delta)
dirichlet = RandomPrimitive(Dirichlet)
exponential = RandomPrimitive(Exponential)
gamma = RandomPrimitive(Gamma)
halfcauchy = RandomPrimitive(HalfCauchy)
lognormal = RandomPrimitive(LogNormal)
multinomial = RandomPrimitive(Multinomial)
normal = RandomPrimitive(Normal)
multivariatenormal = RandomPrimitive(MultivariateNormal)
poisson = RandomPrimitive(Poisson)
uniform = RandomPrimitive(Uniform)
