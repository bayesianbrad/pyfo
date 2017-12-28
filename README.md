

<div align="center">
  <a href="https://github.com/bradleygramhansen/pyfo"> <img width="150px" height="150px" src="docs/pyfologo.png"></a>
</div>


Pyfo enables one to write a model in a subset of first order probabilistic programming language (FOPPL) <sup>[1](#fn1)</sup>. 
FOPPLs base language is Clojure, which enables the syntax to be neat, yet completely expressive.

Inference for FOPPL is performed with a generic HMC, which allows one to
perform inference in models that have potential energies with discontinuities of measure 0 <sup>[3](#fn3)</sup>. In addition to this, we included an
automated framework for embedding discrete distributions, which allows one to perform inference in models containing both discrete
and continous latent variables.

The alpha version of the HMC algorithm in pyfo is Discontinuous HMC <sup>[2](#fn2)</sup>, in which both discrete random variables and continuous random variables with discontinuous densities will be updated by the "discrete" integrator in DHMC. 
Once this version is completely finished and running, we will consider more advanced integrator, for example, RHMC. 

# Requirements
 * clojure
 * pyfo
 * foppl (should be installed when the user installs leiningen as that will pull the libraries from clojars - it will be package by then)
 (YZ: Need double confirm: should only have pyfo: 1. should not require clojure; 2. should not require foppl, which should be take as the semantic rule as writing model.)

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
samples = stats['samples'] # returns dataframe of all samples.
```

## Contributors

Bradley Gram-Hansen
Tobias Kohn
Sam Staton
Frank Wood
Hongseok Yang
Yuan Zhou

To do:
* Have a function that extracts the relevent information from the dataframe, prints a table
showing summary statistics for each chain. (ESS?)
* distinguish the variables with piecewise continuous densities automatedly in the compiler. 
* fully worked DHMC algorithm
* algorithms for baseline comparison

<a name="fn1">1</a>: Jan-Willem van de Meent, Brooks Paige, Hongseok Yang, and Frank Wood. "A Tutorial on Probabilistic Programming
" under development.

<a name="fn2">2</a>: Nishimura, Akihiko, David Dunson, and Jianfeng Lu. "Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters." arXiv preprint arXiv:1705.08510 (2017).

<a name="fn3">3</a>:  Zhou* , Gram-Hansen*, Kohn, Stanton,  Hongsoek, Wood. "Hamiltonian Monte Carlo for Non-Differentiable Points in Probabilistic Programming Languages."