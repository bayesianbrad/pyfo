#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 22. Feb 2018, Tobias Kohn
# 21. Mar 2018, Tobias Kohn
#
from typing import Optional

from .transforms import (ppl_new_simplifier, ppl_raw_simplifier, ppl_functions_inliner,
                         ppl_symbol_simplifier, ppl_static_assignments)
from . import ppl_ast
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
        if namespace is None:
            namespace = {}
        raw_sim = ppl_raw_simplifier.RawSimplifier(namespace)
        result = raw_sim.visit(result)
        if simplify:
            result = ppl_functions_inliner.FunctionInliner().visit(result)
            result = raw_sim.visit(result)
        if len(raw_sim.imports) > 0:
            result = ppl_ast.makeBody([ppl_ast.AstImport(module_name=module) for module in raw_sim.imports], result)

    if simplify and result is not None:
        result = ppl_static_assignments.StaticAssignments().visit(result)
        result = ppl_new_simplifier.Simplifier().visit(result)

    result = ppl_symbol_simplifier.SymbolSimplifier().visit(result)
    return result
