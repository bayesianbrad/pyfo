r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions pyfo.distributions.

Policy gradient methods can be implemented using the
:meth:`~torch pyfo.distributions.distributions pyfo.distributions.Distribution pyfo.distributions.log_prob` method, when the probability
density function is differentiable with respect to its parameters pyfo.distributions. A basic
method is the REINFORCE rule:

 pyfo.distributions. pyfo.distributions. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta` pyfo.distributions.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function pyfo.distributions. Note that we use a negative because optimisers use gradient
descent, whilst the rule above assumes gradient ascent pyfo.distributions. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # NOTE: this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m pyfo.distributions.sample()
    next_state, reward = env pyfo.distributions.step(action)
    loss = -m pyfo.distributions.log_prob(action) * reward
    loss pyfo.distributions.backward()
"""

from  pyfo.distributions.bernoulli import Bernoulli
from  pyfo.distributions.beta import Beta
from  pyfo.distributions.binomial import Binomial
from  pyfo.distributions.categorical import Categorical
from  pyfo.distributions.cauchy import Cauchy
from  pyfo.distributions.chi2 import Chi2
from  pyfo.distributions.delta import Delta
from  pyfo.distributions.dirichlet import Dirichlet
from  pyfo.distributions.exponential import Exponential
from  pyfo.distributions.fishersnedecor import FisherSnedecor
from  pyfo.distributions.gamma import Gamma
from  pyfo.distributions.geometric import Geometric
from  pyfo.distributions.gumbel import Gumbel
from  pyfo.distributions.laplace import Laplace
from  pyfo.distributions.log_gamma import LogGamma
from  pyfo.distributions.multinomial import Multinomial
from  pyfo.distributions.normal import Normal
from  pyfo.distributions.mvn import MultivariateNormal #as mvn
from  pyfo.distributions.one_hot_categorical import OneHotCategorical
from  pyfo.distributions.pareto import Pareto
from  pyfo.distributions.poisson import Poisson
from  pyfo.distributions.uniform import Uniform
from  pyfo.distributions.studentt import StudentT

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
