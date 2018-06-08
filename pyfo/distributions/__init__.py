r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions distributions.

Policy gradient methods can be implemented using the
:meth:`~torch distributions.distributions distributions.Distribution distributions.log_prob` method, when the probability
density function is differentiable with respect to its parameters distributions. A basic
method is the REINFORCE rule:

 distributions. distributions. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta` distributions.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function distributions. Note that we use a negative because optimisers use gradient
descent, whilst the rule above assumes gradient ascent distributions. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # NOTE: this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m distributions.sample()
    next_state, reward = env distributions.step(action)
    loss = -m distributions.log_prob(action) * reward
    loss distributions.backward()
"""

from  distributions.bernoulli import Bernoulli
from  distributions.beta import Beta
from  distributions.binomial import Binomial
from  distributions.categorical import Categorical
from  distributions.cauchy import Cauchy
from  distributions.chi2 import Chi2
from  distributions.delta import Delta
from  distributions.dirichlet import Dirichlet
from  distributions.exponential import Exponential
from  distributions.fishersnedecor import FisherSnedecor
from  distributions.gamma import Gamma
from  distributions.geometric import Geometric
from  distributions.gumbel import Gumbel
from  distributions.laplace import Laplace
from  distributions.log_gamma import LogGamma
from  distributions.multinomial import Multinomial
from  distributions.normal import Normal
from  distributions.mvn import MultivariateNormal
from  distributions.one_hot_categorical import OneHotCategorical
from  distributions.pareto import Pareto
from  distributions.poisson import Poisson
from  distributions.uniform import Uniform
from  distributions.studentt import StudentT

__all__ = [
    'Bernoulli',
    'Beta',
    'Binomial',
    'Categorical',
    'Cauchy',
    'Chi2',
    'Delta',
    'Dirichlet',
    'distribution_pyro',
    'Exponential',
    'FisherSnedecor',
    'Gamma',
    'Geometric',
    'Gumbel',
    'Laplace',
    'LogGamma'
    'Multinomial',
    'MultivariateNormal',
    'Normal',
    'OneHotCategorical',
    'Poisson',
    'Pareto',
    'StudentT',
    'Uniform',
]
