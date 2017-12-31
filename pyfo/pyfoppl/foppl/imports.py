#
# (c) 2017, Tobias Kohn
#
# 18. Nov 2017
# 22. Dec 2017
#
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from .compiler import compile
from .model_generator import Model_Generator

# [Bradley] Testing : using import sys, the first argument of sys.path is the path to the script
# in which the python interpreter is first invoked. This path does not include the script name, just the
# directory including the script. Seems to work.
import sys
PATH = sys.path[0]

def compile_module(module, input_text):
    graph, expr = compile(input_text)
    model_gen = Model_Generator(graph)
    code = model_gen.generate_class()
    exec(code, module.__dict__)
    module.graph = graph
    module.code = code
    return module

class Clojure_Loader(_Loader):

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(module.__name__) as input_file:
            input_text = '\n'.join(input_file.readlines())
            compile_module(module, input_text)

class Clojure_Finder(_MetaPathFinder):

    def find_module(self, fullname, path=PATH):
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

# import sys removed as importing at top
sys.meta_path.append(Clojure_Finder())
