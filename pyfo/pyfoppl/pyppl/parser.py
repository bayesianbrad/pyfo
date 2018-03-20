#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 22. Feb 2018, Tobias Kohn
# 20. Mar 2018, Tobias Kohn
#
from typing import Optional

from pyfo.pyfoppl.pyppl.transforms import ppl_simplifier
from pyfo.pyfoppl.pyppl.transforms import ppl_raw_simplifier
from pyfo.pyfoppl.pyppl.transforms import ppl_functions_inliner
from . import ppl_symbol_table, ppl_ast
from .fe_clojure import ppl_foppl_parser
from .fe_python import ppl_python_parser


def _detect_language(s:str):
    for char in s:
        if char in ['#']:
            return 'py'

        elif char in [';', '(']:
            return 'clj'

        elif 'A' <= char <= 'Z' or 'a' <= char <= 'z' or char == '_':
            return 'py'

        elif char > ' ':
            return 'py'

    return None


def parse(source:str, *, simplify:bool=True, language:Optional[str]=None, namespace:Optional[dict]=None):
    result = None
    if type(source) is str and str != '':
        lang = _detect_language(source) if language is None else language.lower()
        if lang in ['py', 'python']:
            result = ppl_python_parser.parse(source)

        elif lang in ['clj', 'clojure']:
            result = ppl_foppl_parser.parse(source)

        elif lang == 'foppl':
            result = ppl_foppl_parser.parse(source)

    if type(result) is list:
        result = ppl_ast.makeBody(result)

    if result is not None:
        raw_sim = ppl_raw_simplifier.RawSimplifier()
        result = raw_sim.visit(result)
        if simplify or True:
            result = ppl_functions_inliner.FunctionInliner().visit(result)
            result = raw_sim.visit(result)

    if simplify and result is not None:
        sym_table = ppl_symbol_table.SymbolTableGenerator()
        sym_table.visit(result)
        result = ppl_simplifier.simplify(result, sym_table.symbols)

    return result
