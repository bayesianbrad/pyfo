r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functionspyfo.distributions.

Policy gradient methods can be implemented using the
:meth:`~torchpyfo.distributions.distributionspyfo.distributions.Distributionpyfo.distributions.log_prob` method, when the probability
density function is differentiable with respect to its parameterspyfo.distributions. A basic
method is the REINFORCE rule:

pyfo.distributions.pyfo.distributions. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`pyfo.distributions.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss functionpyfo.distributions. Note that we use a negative because optimisers use gradient
descent, whilst the rule above assumes gradient ascentpyfo.distributions. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # NOTE: this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = mpyfo.distributions.sample()
    next_state, reward = envpyfo.distributions.step(action)
    loss = -mpyfo.distributions.log_prob(action) * reward
    losspyfo.distributions.backward()
"""

from pyfo.distributions.bernoulli import Bernoulli
from pyfo.distributions.beta import Beta
from pyfo.distributions.binomial import Binomial
from pyfo.distributions.categorical import Categorical
from pyfo.distributions.cauchy import Cauchy
from pyfo.distributions.chi2 import Chi2
from pyfo.distributions.dirichlet import Dirichlet
from pyfo.distributions.Distribution_wrapper import TorchDistribution
from pyfo.distributions.mvn import MultivariateNormal
from pyfo.distributions.exponential import Exponential
from pyfo.distributions.fishersnedecor import FisherSnedecor
from pyfo.distributions.gamma import Gamma
from pyfo.distributions.geometric import Geometric
from pyfo.distributions.gumbel import Gumbel
from pyfo.distributions.laplace import Laplace
from pyfo.distributions.multinomial import Multinomial
from pyfo.distributions.normal import Normal
from pyfo.distributions.one_hot_categorical import OneHotCategorical
from pyfo.distributions.pareto import Pareto
from pyfo.distributions.studentT import StudentT
from pyfo.distributions.uniform import Uniform

__all__ = [
    'Bernoulli',
    'Beta',
    'Binomial',
    'Categorical',
    'Cauchy',
    'Chi2',
    'Dirichlet',
    'Distribution',
    'Exponential',
    'FisherSnedecor',
    'Gamma',
    'Geometric',
    'Gumbel',
    'Laplace',
    'Multinomial',
    'MultivariateNormal'
    'Normal',
    'OneHotCategorical',
    'Pareto',
    'StudentT',
    'Uniform',
]
