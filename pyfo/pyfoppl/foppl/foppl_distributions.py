#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 21. Dec 2017, Tobias Kohn
# 20. Jan 2018, Tobias Kohn
#
distributions = {
    "Bernoulli": "discrete",
    "Binomial": "discrete",
    "Categorical": "discrete",
    "Discrete": "discrete",
    "Multinomial": "discrete",
    "Poisson": "discrete",

    "Beta": "continuous",
    "Cauchy": "continuous",
    "Dirichlet": "continuous",
    "Exponential": "continuous",
    "Gamma": "continuous",
    "HalfCauchy": "continuous",
    "LogNormal": "continuous",
    "MultivariateNormal": "continuous",
    "Normal": "continuous",
    "Uniform": "continuous"
}

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
    "binomial": "Binomial",
    "categorical": "Categorical",
    "cauchy": "Cauchy",
    "dirichlet": "Dirichlet",
    "exponential": "Exponential",
    "gamma": "Gamma",
    "half_cauchy": "HalfCauchy",
    "log_normal": "LogNormal",
    "multinomial": "Multinomial",
    "mvn": "mvn",
    "normal": "Normal",
    "poisson": "Poisson",
    "uniform": "Uniform"
}

distribution_params = {
    "Bernoulli": ["ps"],
    "Beta": ["alpha", "beta"],
    "Binomial": ["ps"],
    "Categorical": ["ps"],
    "Cauchy": ["mu", "gamma"],
    "Dirichlet": ["alpha"],
    "Exponential": ["lam"],
    "LogNormal": ["mu", "sigma"],
    "Gamma": ["alpha", "beta"],
    "HalfCauchy": ["mu", "gamma"],
    "Multinomial": ["ps", "n"],
    "MultivariateNormal": ["mu", "covariance_matrix"],
    "Normal": ["mu", "sigma"],
    "Poisson": ["lam"],
    "Uniform": ["a", "b"]
}

def get_arg_count(distr: str):
    return len(distribution_params.get(distr, []))