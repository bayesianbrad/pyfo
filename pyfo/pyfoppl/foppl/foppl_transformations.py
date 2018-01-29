#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 26. Jan 2018, Tobias Kohn
# 26. Jan 2018, Tobias Kohn
#
class Transformations(object):
    """
    Some distributions require a transformation, so as to make sure that their support range covers all over
    the real numbers R. For each such distribution that requires a transformation, this class implements a
    class-method with the name of the distribution. The returned value of the method is a tuple comprising
    three string values:

    - The first value gives the name of a function to be called for a 'forward'-transformation. This is used in
      `observe`.

    - The second value names a function to be called by the code to revert the transformation.

    - The third value is the name of a new, transformed distribution to be used instead of the old one. Note
      that this new distributions could be the same as the old one.

    Example:
        We might want to transform `Gamma` to `LogGamma`. Then we have:
        * `sample(Gamma(X, Y))`     -> `bijector.Exp( sample( LogGamma(X, Y) ) )`
        * `observe(Gamma(X, Y), Z)` -> `observe( LogGamma(X, Y), bijector.Log(Z) )`
    """

    @classmethod
    def Gamma(cls):
        """
        We replace `Gamma(X, Y)` by `Exp(LogGamma(X, Y))`.
        """
        return "bijector.Log", "bijector.Exp", "LogGamma"
