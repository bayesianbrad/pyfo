#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 07. Feb 2018, Tobias Kohn
# 19. Mar 2018, Tobias Kohn
#
from typing import Optional
from . import distributions, parser
from .backend import ppl_graph_generator


def compile_model(source, *,
                  language: Optional[str]=None,
                  imports=None,
                  # base_class: Optional[str]=None):
                  base_class: 'from pyfo.pyfoppl.pyppl.ppl_base_model import base_model')
    if type(imports) in (list, set, tuple):
        imports = '\n'.join(imports)
    ast = parser.parse(source, language=language, namespace=distributions.namespace)
    gg = ppl_graph_generator.GraphGenerator()
    gg.visit(ast)
    return gg.generate_model(base_class=base_class, imports=imports)
