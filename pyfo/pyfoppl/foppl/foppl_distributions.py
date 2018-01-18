#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 21. Dec 2017, Tobias Kohn
# 18. Jan 2018, Tobias Kohn
#
discrete_distributions = {
    "Bernoulli",
    "Categorical",
    "Discrete",
    "Multinomial",
    "Poisson"
}

continuous_distributions = {
    "Beta",
    "Cauchy",
    "Dirichlet",
    "Exponential",
    "Gamma",
    "HalfCauchy",
    "LogNormal",
    "MultivariateNormal",
    "Normal",
    "Uniform"
}

distribution_map = {
    "bernoulli": "Bernoulli",
    "beta": "Beta",
    "categorical": "Categorical",
    "cauchy": "Cauchy",
    "dirichlet": "Dirichlet",
    "exponential": "Exponential",
    "gamma": "Gamma",
    "half_cauchy": "HalfCauchy",
    "log_normal": "LogNormal",
    "multinomial": "Multinomial",
    "mvn": "MultivariateNormal",
    "normal": "Normal",
    "poisson": "Poisson",
    "uniform": "Uniform"
}

distribution_params = {
    "Bernoulli": ["ps"],
    "Beta": ["alpha", "beta"],
    "Categorical": ["ps"],
    "Cauchy": ["mu", "gamma"],
    "Dirichlet": ["alpha"],
<<<<<<< HEAD
    "Exponential": ["lam"],   #ZY
=======
    "Exponential": ["lam"],
>>>>>>> 01968da378315fa6a38541e87b24797aab5d7505
    "LogNormal": ["mu", "sigma"],
    "Gamma": ["alpha", "beta"],
    "HalfCauchy": ["mu", "gamma"],
    "Multinomial": ["ps", "n"],
    "MultivariateNormal": ["mu", "covariance_matrix"],
    "Normal": ["mu", "sigma"],
    "Poisson": ["lam"],
    "Uniform": ["a", "b"]
}
