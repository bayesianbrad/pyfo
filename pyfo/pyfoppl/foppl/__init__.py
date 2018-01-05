#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
##
# 19. Dec 2017, Tobias Kohn
# 04. Jan 2018, Tobias Kohn
#
class Options(object):
    """
    This class provides flags and general options to control the compilation process.

    `eager_conditionals`:
        Controls whether conditional statements (`if`-expressions) should be evaluated eagerly or lazily.
        _At the moment, only eager evaluation is supported._

    `uniform_conditionals`:
        If this flag is set to `True`, the compiler will transform all comparisons (except for equality) to be
        in the form `X >= 0`. For instance, `x < 5` will thus be transformed to `not (x-5 >= 0)`.

    `inline_variables`:
        This flag controls if derived latent variables (those computed from other variables instead of samples
        from a distribution) are inlined. If set to `True`, a variable such as `a = x1 + x2` will not create a
        vertex of its own, while `a = sample()` always will.
        _At the moment, derived variables are always inlined._

    `model_imports`:
        This is a list of import statements to be included in the generated module for the model-class.

    `conditional_suffix`:
        A string suffix that is appended to conditional variables.
    """

    eager_conditionals = True

    uniform_conditionals = True

    inline_variables = True

    model_interface = ('object', '')
    #model_interface = ('interface', 'pyfo.utils.interface')

    model_imports = [
        'import math',
        'import numpy as np',
        # 'import torch',
        # 'from torch.autograd import Variable',
        # 'import pyfo.distributions as dist'
    ]

    conditional_suffix = '.data[0]'
