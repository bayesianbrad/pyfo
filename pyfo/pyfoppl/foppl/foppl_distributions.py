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
    "Binomial"
    "Bernoulli",
    "Categorical",
    "Discrete",
    "Geometric"
    "Multinomial",
    "Poisson"
}

continuous_distributions = {
    "Beta",
    "Cauchy",
    "Chi2",
    "Dirichlet",
    "Delta"
    "Exponential",
    "Gamma",
    "Gumbel"
    "HalfCauchy",
    "LogNormal",
    "Laplace"
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
    "chi2":"Chi2",
    "delta":"Delta",
    "dirichlet": "Dirichlet",
    "exponential": "Exponential",
    "gamma": "Gamma",
    "half_cauchy": "HalfCauchy",
    "log_normal": "LogNormal",
    "multinomial": "Multinomial",
    "mvn": "MultivariateNormal",
    "normal": "Normal",
    "poisson": "Poisson",
    "studentt":"StudentT",
    "uniform": "Uniform"
}

distribution_params = {
    "Bernoulli": ["total_count","probs"],
    "Beta": ["alpha", "beta"],
    "Binomial": ["probs"],
    "Categorical": ["probs"],
    "Cauchy": ["mu", "gamma"],
    "Chi2": ["df"],
    "Dirichlet": ["alpha"],
    "Exponential": ["lam"],
    "LogNormal": ["mu", "sigma"],
    "Gamma": ["alpha", "beta"],
    "Geomteric":["probs"],
    "Gumbel":["loc","scale"],
    "HalfCauchy": ["mu", "gamma"],
    "Multinomial": ["total_count","probs"],
    "MultivariateNormal": ["mu", "covariance_matrix"],
    "Normal": ["mu", "sigma"],
    "Poisson": ["lam"],
    "Studentt":["df","loc","scale"],
    "Uniform": ["a", "b"]
}

def get_arg_count(distr: str):
    return len(distribution_params.get(distr, []))