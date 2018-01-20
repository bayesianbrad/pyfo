

<div align="center">
  <a href="https://github.com/bradleygramhansen/pyfo"> <img width="150px" height="150px" src="docs/pyfologo.png"></a>
</div>


`Pyfo` enables one to write a model in the flexible first order probabilistic programming framework
that is FOPPL <sup>[1](#fn1)</sup>. FOPPL's base language is Clojure, which enables the syntax to be neat, yet completely expressive.
What is great about `pyfo` is that one does not even need to install Clojure, nor need to understand how to use the REPL.
Simply write your model in Clojure, or alternatively you can write the model in pure Python code in accordance with the
[interface](https://github.com/bradleygramhansen/pyfo/blob/master/pyfo/utils/interface.py), and`pyfo` does the rest.

Inference for FOPPL is performed with Discontinuous HMC <sup>[2](#fn2)</sup> and Reflection, refraction HMC <sup>[3](#fn3)</sup>, which allows one to
perform inference in models that have discontinuities that are of measure 0 <sup>[4](#fn4)</sup>. In addition to this, we included an
automated framework for embedding discrete distributions, which maps the discrete distributions to piece-wise constant functions, that have measure 0
discontinuities. This allows one to perform inference in models containing discrete and continuous latent variables.


# Requirements
 * pyfo
 * pytorch

# Installation instructions
 * Instructions for pytorch can be found here: [http://pytorch.org/] ( You will need the distribution classes and will have to install from source (for now))
 * Pyfo can be installed via pip.
  ```python
    pip install pyfo
   ```


# Example

## Writing the model
Write the model in Clojue and contained within pyfo is FOPPL. This will take your model and compile it to python code.
It essentially constructs the logjoint of the model and ensures that the order of the direct acylic graph (DAG) is
preserved. For example one_dim_gauss.clj

```clojure

   (let [x (sample (normal 1.0 5.0))]
        (observe (normal x 2.0) 7.0)
    x)
```
which we save as `<model_name>.clj` .  In this instance `model_name = onedimgauss`, therefore we save as onedimgauss.clj
## Performing the inference

Ensure that your model, in this case `onedimgauss.clj`, is in the same directory in which you are placing the following
inference script.

```python
from pyfo.pyfoppl.foppl import imports # this uses a loader and finder module.
import <model_name> as model # when we do this `imports` is triggered, compiles the modle automatically and loads it as a module.
from pyfo.inference.dhmc import DHMCSampler as dhmc

print(model.code)
dhmc_ = dhmc(model.model)
burn_in = 10 ** 2
n_sample = 10 ** 3
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, print_stats=True,plot=True, save_samples=True)

samples =  stats['samples']
all_samples = stats['samples_wo_burin'] # type, panda dataframe

# means = stats['means']
# print(means)

```

## Contributors

- Bradley Gram-Hansen
- Tobias Kohn
- Yuan Zhou

## References

<a name="fn1">1</a>: Jan-Willem van de Meent, Brooks Paige, Hongseok Yang, and Frank Wood. "A Tutorial on Probabilistic Programming."

<a name="fn2">2</a>: Nishimura, Akihiko, David Dunson, and Jianfeng Lu. "Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters." arXiv preprint arXiv:1705.08510 (2017).

<a name="fn3">3</a>: Hadi Mohasel Afshar, Justin Domke. "Reflection, Refraction, and Hamiltonian Monte Carlo."

<a name="fn4">4</a>: Bradley Gram-Hansen*, Yuan Zhou*, Tobias Kohn, Sam Stanton, Hongseok Yang, Frank Wood. "Hamiltonian Monte Carlo for Non-Differentiable Points in Probabilistic Programming Languages."
