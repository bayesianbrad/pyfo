

<div align="center">
  <a href="https://github.com/bradleygramhansen/pyfo"> <img width="150px" height="150px" src="docs/pyfologo.png"></a>
</div>


Pyfo enables one to write a model in the flexiable first order probabilistic programming framework
that is FOPPL. FOPPLs base language is Clojure, which enables the syntax to be neat, yet completely expressive.

Inference for FOPPL is performed with Discountinous HMC <sup>[1](#fn1)</sup>, which allows one to
perform inference in models that have discontinuities that are of measure 0 <sup>[2](#fn2)</sup>. In addition to this, we included an
automated framework for embedding discrete distributions, which allows one to perform inference in models contianign both discrete
and continous latent variables.

# Example

## Writing the model
Write model in foppl, for example one_dim_gauss.clj

```clojure
(def one-gaussian
    (foppl-query
        (let [x (sample (normal 1.0 5.0))]
            (observe (normal x 2.0) 7.0)
        x)))
```

## Performing the inference

```python
import pyfo
from pyfo.inference.DHMC as dhmc
n_burnin = 1000
n_sample = 10 ** 4
stepsize = 0.03
trajectorystep = [10, 20]
# Either this
DHMC    = dhmc(logp, start, step_size, n_steps, **kwargs)
dhmc.sample(n_samples, n_burnin, n_chains)
# or this, where we take the foppl input, compile it internal into the desired interface and then provide
# logp to the sampler ourselves. 
DHMC_object = dhmc(one_dim_gauss.clj, stepsize, trajectorystep, n_burnin, n_samples) # creates sampler object
samples = DHMC_object.samples # returns samples of the inferred posterior
```


<a name="fn1">1</a>: Nishimura, Akihiko, David Dunson, and Jianfeng Lu. "Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters." arXiv preprint arXiv:1705.08510 (2017).

<a name="fn2">2</a>: Gram-Hansen,Yuan, Hongsoek, Stanton, Wood. "Hamiltonian Monte Carlo for Non-Differentiable Points in Probabilistic Programming Languages."