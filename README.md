

<div align="center">
  <a href="https://github.com/bradleygramhansen/pyfo"> <img width="150px" height="150px" src="docs/pyfologo.png"></a>
</div>


Pyfo enables one to write a model in the flexiable first order probabilistic programming framework
that is FOPPL. FOPPLs base language is Clojure, which enables the syntax to be neat, yet completely expressive.

Inference for FOPPL is performed with Discountinous HMC <sup>[1](#fn1)</sup>, which allows one to
perform inference in models that have discontinuities that are of measure 0 <sup>[2](#fn2)</sup>. In addition to this, we included an
automated framework for embedding discrete distributions, which allows one to perform inference in models contianign both discrete
and continous latent variables.

# Requirements
 * clojure
 * pyfo
 * foppl (should be installed when the user installs leiningen as that will pull the libraries from clojars - it will be package by then)

# Installation instructions
 * Instructions for clojure can be found here: [https://clojure.org/guides/getting_started]
 * Pyro can be installed via pip.
  ```python
    pip install pyfo
   ```


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
## Compiling the model

Not yet completed. But it needs to be run in the terminal (a little) like this:
        ```clojure
        clojure compiler.clj <model_name>.clj
        Saves to directory where <model_name>.clj is.
        ```

## Performing the inference

```python
import pyfo
from pyfo.inference.DHMC as dhmc
burn_in = 1000
n_sample = 10 ** 4
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

import <model_name>

dhmc_    = dhmc(<model_name>, n_chains)

stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range)
samples = stats['samples'] # returns dataframe of all samples.
```

To do:
* Have a function that extracts the relevent information from the dataframe, prints a table
showing summary statistics for each chain.

<a name="fn1">1</a>: Nishimura, Akihiko, David Dunson, and Jianfeng Lu. "Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters." arXiv preprint arXiv:1705.08510 (2017).

<a name="fn2">2</a>: Gram-Hansen,Yuan, Hongsoek, Stanton, Wood. "Hamiltonian Monte Carlo for Non-Differentiable Points in Probabilistic Programming Languages."