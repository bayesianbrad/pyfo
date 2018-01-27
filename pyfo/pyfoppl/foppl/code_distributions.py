#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 17. Jan 2018, Tobias Kohn
# 27. Jan 2018, Tobias Kohn
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
    def __arg_count_error(cls, name:str, args):
        raise TypeError("wrong number or type of arguments for '{}': {}".format(name, len(args)))

    @classmethod
    def __check_arg_count(cls, name:str, arg_count:int, args):
        if len(args) != arg_count:
            cls.__arg_count_error(name, args)

    @classmethod
    def bernoulli(cls, args: list):
        cls.__check_arg_count('bernoulli', 1, args)
        return IntegerType()

    @classmethod
    def beta(cls, args: list):
        cls.__check_arg_count('beta', 2, args)
        return FloatType()

    @classmethod
    def binomial(cls, args: list):
        cls.__check_arg_count('binomial', 2, args)
        arg = args[1]
        if isinstance(arg, SequenceType):
            return ListType(IntegerType, arg.size)
        else:
            return FloatType()

    @classmethod
    def categorical(cls, args: list):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, ListType) and isinstance(arg.item_type, ListType):
                return ListType(IntegerType, arg.size)
            if isinstance(arg, SequenceType):
                return IntegerType()
        cls.__arg_count_error('categorical', args)

    @classmethod
    def cauchy(cls, args: list):
        cls.__check_arg_count('cauchy', 2, args)
        return FloatType()

    @classmethod
    def dirichlet(cls, args: list):
        cls.__check_arg_count('dirichlet', 1, args)
        arg = args[0]
        if isinstance(arg, SequenceType):
            return ListType(FloatType, arg.size)
        else:
            return FloatType()

    @classmethod
    def exponential(cls, args: list):
        cls.__check_arg_count('exponential', 1, args)
        return FloatType()

    @classmethod
    def gamma(cls, args: list):
        cls.__check_arg_count('gamma', 2, args)
        return cls.__cont_dist__(args, 0)

    @classmethod
    def loggamma(cls, args: list):
        cls.__check_arg_count('log-gamma', 2, args)
        return cls.__cont_dist__(args, 0)

    @classmethod
    def halfcauchy(cls, args: list):
        cls.__check_arg_count('half-cauchy', 2, args)
        return FloatType()

    @classmethod
    def lognormal(cls, args: list):
        cls.__check_arg_count('log-normal', 2, args)
        return FloatType()

    @classmethod
    def multinomial(cls, args: list):
        cls.__check_arg_count('multinomial', 2, args)
        return IntegerType()

    @classmethod
    def multivariatenormal(cls, args: list):
        cls.__check_arg_count('multivariate-normal', 2, args)
        arg = args[0]
        if isinstance(arg, SequenceType):
            return ListType(FloatType, arg.size)
        else:
            return FloatType()

    @classmethod
    def normal(cls, args: list):
        cls.__check_arg_count('normal', 2, args)
        return cls.__cont_dist__(args, 0)

    @classmethod
    def poisson(cls, args: list):
        cls.__check_arg_count('poisson', 1, args)
        return IntegerType()

    @classmethod
    def uniform(cls, args: list):
        cls.__check_arg_count('uniform', 2, args)
        return FloatType()


def get_result_type(name: str, args: list):
    if hasattr(DistributionTypes, name.lower()):
        method = getattr(DistributionTypes, name.lower())
        return method(args)
    return AnyType()
