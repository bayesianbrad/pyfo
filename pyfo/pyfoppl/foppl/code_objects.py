#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 16. Jan 2018, Tobias Kohn
# 17. Jan 2018, Tobias Kohn
#
from .code_types import *

class CodeObject(object):

    code_type = AnyType()

##############################################################################

class CodeBinary(CodeObject):

    def __init__(self, left: CodeObject, op: str, right: CodeObject):
        self.left = left
        self.op = op
        self.right = right
        self.code_type = apply_binary(left.code_type, op, right.code_type)

    def __repr__(self):
        return "({}{}{})".format(repr(self.left), self.op, repr(self.right))


class CodeDistribution(CodeObject):

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.code_type = DistributionType(name, [a.code_type for a in args])

    def __repr__(self):
        return "dist.{}({})".format(self.name, ', '.join([repr(a) for a in self.args]))


class CodeFunction(CodeObject):

    pass


class CodeFunctionCall(CodeObject):

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return "{}({})".format(self.name, ', '.join([repr(a) for a in self.args]))


class CodeObserve(CodeObject):

    def __init__(self, distribution: CodeDistribution, value: CodeObject):
        self.distribution = distribution
        self.value = value
        self.code_type = self.distribution.code_type.result_type().union(value.code_type)

    def __repr__(self):
        return "observe({}, {})".format(repr(self.distribution), repr(self.value))


class CodeSample(CodeObject):

    def __init__(self, distribution: CodeDistribution):
        self.distribution = distribution
        self.code_type = self.distribution.code_type.result_type()

    def __repr__(self):
        return "sample({})".format(repr(self.distribution))


class CodeSubscript(CodeObject):

    def __init__(self, seq: CodeObject, index):
        self.seq = seq
        self.index = index
        if isinstance(seq.code_type, SequenceType):
            self.code_type = seq.code_type.item_type
        else:
            raise TypeError("'{}' is not a sequuence".format(repr(seq)))

    def __repr__(self):
        if type(self.index) in [int, float]:
            index = repr(int(self.index))
        elif isinstance(self.index, CodeObject):
            index = repr(self.index)
        else:
            raise TypeError("invalid index: '{}'".format(self.index))
        return "{}[{}]".format(repr(self.seq), index)


class CodeSymbol(CodeObject):

    def __init__(self, name: str, code_type: AnyType):
        self.name = name
        self.code_type = code_type

    def __repr__(self):
        return self.name


class CodeUnary(CodeObject):

    def __init__(self, op: str, item: CodeObject):
        self.op = op
        self.item = item
        self.code_type = item.code_type

    def __repr__(self):
        return "{}{}".format(self.op, self.item)


class CodeValue(CodeObject):

    def __init__(self, value):
        self.value = value
        self.code_type = get_code_type_for_value(value)

    def __repr__(self):
        return repr(self.value)
