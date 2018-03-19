#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 22. Feb 2018, Tobias Kohn
# 19. Mar 2018, Tobias Kohn
#
from typing import Optional

from pyppl.transforms import ppl_simplifier
from pyppl.transforms import ppl_raw_simplifier
from . import ppl_symbol_table
from .fe_clojure import ppl_clojure_parser
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

    if simplify and result is not None:
        result = ppl_raw_simplifier.RawSimplifier().visit(result)

    if result is not None:
        symbol_table = ppl_symbol_table.SymbolTableGenerator(namespace=namespace)
        symbol_table.visit(result)
        symbol_list = symbol_table.get_symbols()
    else:
        symbol_list = []

    if simplify and result is not None:
        result = ppl_simplifier.simplify(result, symbol_list)

    return result
