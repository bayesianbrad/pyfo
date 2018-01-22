#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 08. Jan 2018, Tobias Kohn
# 19. Jan 2018, Tobias Kohn
#
__all__ = ['conj']

def conj(seq, *items):
    return seq + list(items)
