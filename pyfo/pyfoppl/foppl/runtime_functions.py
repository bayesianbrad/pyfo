#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 08. Jan 2018, Tobias Kohn
# 08. Jan 2018, Tobias Kohn
#
"""
This dictionary provides the Python implementations for various clojure/FOPPL-functions. If any of these functions
is used inside the FOPPL-code, the usage is recorded in the field `used_functions` of the graph-structure. The
model generator then uses this dictionary to include the implementation for the specific function(s) into the code
generated.
"""
runtime_functions = {
    'conj': 'conj = lambda S, *I: S + list(I)'
}