#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 18. Nov 2017, Tobias Kohn
# 04. Jan 2018, Bradley Gram-Hansen
#
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from .compiler import compile
from .model_generator import Model_Generator
import sys

_PATH = sys.path[0]

def compile_module(module, input_text):
    graph, expr = compile(input_text)
    model_gen = Model_Generator(graph)
    code = model_gen.generate_class()
    exec(code, module.__dict__)
    module.graph = graph
    module.code = code
    if module.model:
        module.model.graph = graph
    return module

class Clojure_Loader(_Loader):

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(module.__name__) as input_file:
            input_text = '\n'.join(input_file.readlines())
            compile_module(module, input_text)

class Clojure_Finder(_MetaPathFinder):

    def find_module(self, fullname, path=None):
        if path is None:
            path = _PATH
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path, target = None):
        import os.path
        from importlib.machinery import ModuleSpec

        fullname = fullname.split(sep='.')[-1]

        if '.' in fullname:
            raise NotImplementedError()

        if os.path.exists(fullname + ".clj"):
            return ModuleSpec(os.path.realpath(fullname + ".clj"), Clojure_Loader())
        else:
            return None

import sys
sys.meta_path.append(Clojure_Finder())
