from __future__ import absolute_import, division, print_function

# distribution classes
from DHMC.distributions.bernoulli import Bernoulli
from DHMC.distributions.beta import Beta
from DHMC.distributions.categorical import Categorical
from DHMC.distributions.cauchy import Cauchy
from DHMC.distributions.delta import Delta
from DHMC.distributions.dirichlet import Dirichlet
from DHMC.distributions.distribution import Distribution  # noqa: F401
from DHMC.distributions.exponential import Exponential
from DHMC.distributions.gamma import Gamma
from DHMC.distributions.half_cauchy import HalfCauchy
from DHMC.distributions.log_normal import LogNormal
from DHMC.distributions.multinomial import Multinomial
from DHMC.distributions.normal import Normal
from DHMC.distributions.multivariate_normal import MultivariateNormal
from DHMC.distributions.poisson import Poisson
from DHMC.distributions.random_primitive import RandomPrimitive
from DHMC.distributions.uniform import Uniform

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
