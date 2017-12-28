

<div align="center">
  <a href="https://github.com/bradleygramhansen/pyfo"> <img width="150px" height="150px" src="docs/pyfologo.png"></a>
</div>


Pyfo enables one to write a model in the flexiable first order probabilistic programming framework
that is FOPPL <sup>[1](#fn1)</sup>. FOPPLs base language is Clojure, which enables the syntax to be neat, yet completely expressive.

Inference for FOPPL is performed with Discontinuous HMC <sup>[2](#fn2)</sup> and Reflection, refraction HMC <sup>[3](#fn3)</sup>, which allows one to
perform inference in models that have discontinuities that are of measure 0 <sup>[4](#fn4)</sup>. In addition to this, we included an
automated framework for embedding discrete distributions, which maps the discrete distributions to piece-wise constant functions, that have measure 0
discontinuities. This allows one to perform inference in models containing discrete latent varibles.
and continous latent variables.

# Requirements
 * clojure
 * pyfo
 * foppl (should be installed when the user installs leiningen as that will pull the libraries from clojars - it will be package by thenS)

# Installation instructions
 * Instructions for clojure can be found here: [https://clojure.org/guides/getting_started]
 * Pyfo can be installed via pip.
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
from pyfo.inference.dhmc import DHMCSampler as dhmc
burn_in = 1000
n_sample = 10 ** 4
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

import <model_name>

dhmc_    = dhmc(<model_name>, n_chains)

stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range)
parameters = stats['all_params'] # returns all the keys for the parameters
cont_params = states['cont_params'] # returns continuous keys
disc_params = states['disc_params'] # returns discrete keys

samples = stats['samples'] # returns dataframe of all samples. To get all samples for a given parameter simply do: samples_param = samples[<param_name>]
means = states['means'] # returns dictionary key:value, where key - parameter , value = mean of parameter
```

##Contributors

Bradley Gram-Hansen
Tobias Kohn
Yuan Zhou


<a name="fn1">1</a>: Jan-Willem van de Meent, Brooks Paige, Hongseok Yang, and Frank Wood. "A Tutorial on Probabilistic Programming.
"
<a name="fn2">2</a>: Nishimura, Akihiko, David Dunson, and Jianfeng Lu. "Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters." arXiv preprint arXiv:1705.08510 (2017).

<a name="fn3">3</a>: Hadi Mohasel Afshar, Justin Domke. "Reflection, Refraction, and Hamiltonian Monte Carlo."

<a name="fn4">4</a>: Bradley Gram-Hansen, Yuan Zhou, Tobias Kohn, Hongseok Yang, Sam Stanton, Frank Wood. "Hamiltonian Monte Carlo for Non-Differentiable Points in Probabilistic Programming Languages."
