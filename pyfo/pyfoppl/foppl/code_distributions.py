#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 17. Jan 2018, Tobias Kohn
# 17. Jan 2018, Tobias Kohn
#
from .code_types import *

class DistributionTypes(object):

    @classmethod
    def __cont_dist__(cls, args: list, index: int):
        arg = args[index]
        if isinstance(arg, SequenceType):
            return ListType(FloatType, arg.size)
        else:
            return FloatType()

    @classmethod
    def normal(cls, args: list):
        if len(args) == 2:
            return cls.__cont_dist__(args, 0)
        else:
            raise TypeError("wrong number of arguments for 'normal': {}".format(len(args)))


def get_result_type(name: str, args: list):
    if hasattr(DistributionTypes, name.lower()):
        method = getattr(DistributionTypes, name.lower())
        return method(args)
    return AnyType()
