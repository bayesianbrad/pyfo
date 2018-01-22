#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 16. Jan 2018, Tobias Kohn
# 20. Jan 2018, Tobias Kohn
#
from .graphs import *
from .code_types import *

class CodeObject(object):

    code_type = AnyType()

    def to_py(self) -> str:
        return repr(self)

##############################################################################

class CodeBinary(CodeObject):

    def __init__(self, left: CodeObject, op: str, right: CodeObject):
        self.left = left
        self.op = op
        self.right = right
        self.code_type = apply_binary(left.code_type, op, right.code_type)

    def __repr__(self):
        return "({}{}{})".format(repr(self.left), self.op, repr(self.right))

    def to_py(self):
        return "({}{}{})".format(self.left.to_py(), self.op, self.right.to_py())


class CodeCompare(CodeObject):

    def __init__(self, left: CodeObject, op: str, right: CodeObject):
        self.left = left
        self.right = right
        self.op = op
        self.code_type = BooleanType()

    @property
    def is_normalized(self):
        return self.op == '>=' and repr(self.right) == '0'

    def __repr__(self):
        return "({}{}{})".format(repr(self.left), self.op, repr(self.right))

    def to_py(self):
        return "({}{}{})".format(self.left.to_py(), self.op, self.right.to_py())


class CodeDataSymbol(CodeObject):

    def __init__(self, node: DataNode):
        self.node = node
        self.name = node.name
        self.code_type = get_code_type_for_value(node.data)

    def __repr__(self):
        return self.name

    def to_py(self):
        return "state['{}']".format(self.name)


class CodeDistribution(CodeObject):

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.code_type = DistributionType(name, [a.code_type for a in args])

    def __repr__(self):
        return "dist.{}({})".format(self.name, ', '.join([repr(a) for a in self.args]))

    def to_py(self):
        return "dist.{}({})".format(self.name, ', '.join([a.to_py() for a in self.args]))

    def get_support_size(self):
        arg = self.args[0]
        if isinstance(arg, CodeValue) and type(arg.value) is list:
            if all([type(item) is list for item in arg.value]):
                return max([len(item) for item in arg.value])
            else:
                return len(arg.value)
        elif isinstance(arg, CodeVector) and len(arg.items) > 0:
            _is_vector = lambda v: isinstance(v, CodeVector) or (isinstance(v, CodeValue) and type(v.value) is list)
            if all([_is_vector(item) for item in arg.items]):
                return max([len(item) for item in arg.items])
            else:
                return len(arg.items)
        else:
            return None


class CodeFunctionCall(CodeObject):

    def __init__(self, name, args):
        if type(args) is not list:
            args = [args]
        self.name = name
        self.args = args
        self.code_type = AnyType()

    def __repr__(self):
        return "{}({})".format(self.name, ', '.join([repr(a) for a in self.args]))

    def to_py(self):
        return "{}({})".format(self.name, ', '.join([a.to_py() for a in self.args]))


class CodeIf(CodeObject):

    def __init__(self, cond:CodeObject, if_expr:CodeObject, else_expr:CodeObject=None):
        self.cond = cond
        self.if_expr = if_expr
        self.else_expr = else_expr
        if not isinstance(cond.code_type, BooleanType):
            raise TypeError("'if'-condition must be of type 'boolean'")
        if else_expr:
            self.code_type = if_expr.code_type.union(else_expr.code_type)
        else:
            self.code_type = if_expr.code_type

    def __repr__(self):
        else_expr = repr(self.else_expr) if self.else_expr else "None"
        return "{} if {} else {}".format(repr(self.if_expr), repr(self.cond), else_expr)

    def to_py(self):
        else_expr = self.else_expr.to_py() if self.else_expr else "None"
        return "{} if {} else {}".format(self.if_expr.to_py(), self.cond.to_py(), else_expr)


class CodeObserve(CodeObject):

    def __init__(self, vertex: Vertex):
        self.vertex = vertex
        self.distribution = vertex.distribution
        self.value = vertex.observation
        self.code_type = self.distribution.code_type.result_type().union(self.value.code_type)

    def __repr__(self):
        return self.vertex.name

    def to_py(self):
        return "state['{}']".format(self.vertex.name)


class CodeSample(CodeObject):

    def __init__(self, vertex: Vertex):
        self.vertex = vertex
        self.distribution = vertex.distribution
        self.code_type = self.distribution.code_type.result_type()

    def __repr__(self):
        return self.vertex.name

    def to_py(self):
        return "state['{}']".format(self.vertex.name)


class CodeSlice(CodeObject):

    def __init__(self, seq: CodeObject, beginIndex, endIndex):
        self.seq = seq
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        if isinstance(seq.code_type, SequenceType):
            self.code_type = seq.code_type
        else:
            raise TypeError("'{}' is not a sequence".format(repr(seq)))

    def __repr(self, seq_repr):
        if self.beginIndex is None:
            beginIndex = ''
        elif type(self.beginIndex) in [int, float]:
            beginIndex = repr(int(self.beginIndex))
        elif isinstance(self.beginIndex, CodeObject):
            beginIndex = repr(self.beginIndex)
        else:
            raise TypeError("invalid index: '{}'".format(self.beginIndex))

        if self.endIndex is None:
            endIndex = ''
        elif type(self.endIndex) in [int, float]:
            endIndex = repr(int(self.endIndex))
        elif isinstance(self.endIndex, CodeObject):
            endIndex = repr(self.endIndex)
        else:
            raise TypeError("invalid index: '{}'".format(self.endIndex))

        return "{}[{}:{}]".format(repr(self.seq), beginIndex, endIndex)

    def to_py(self):
        if self.beginIndex is None:
            beginIndex = ''
        elif type(self.beginIndex) in [int, float]:
            beginIndex = repr(int(self.beginIndex))
        elif isinstance(self.beginIndex, CodeObject):
            beginIndex = self.beginIndex.to_py()
        else:
            raise TypeError("invalid index: '{}'".format(self.beginIndex))

        if self.endIndex is None:
            endIndex = ''
        elif type(self.endIndex) in [int, float]:
            endIndex = repr(int(self.endIndex))
        elif isinstance(self.endIndex, CodeObject):
            endIndex = self.endIndex.to_py()
        else:
            raise TypeError("invalid index: '{}'".format(self.endIndex))

        return "{}[{}:{}]".format(self.seq.to_py(), beginIndex, endIndex)


class CodeSqrt(CodeObject):

    def __init__(self, item: CodeObject):
        self.item = item
        self.code_type = FloatType()

    def __repr__(self):
        return "math.sqrt({})".format(repr(self.item))

    def to_py(self):
        return "math.sqrt({})".format(self.item.to_py())


class CodeSubscript(CodeObject):

    def __init__(self, seq: CodeObject, index):
        self.seq = seq
        self.index = index
        if isinstance(seq.code_type, SequenceType):
            self.code_type = seq.code_type.item_type
        else:
            raise TypeError("'{}' is not a sequence".format(repr(seq)))

    def __repr__(self):
        if type(self.index) in [int, float]:
            index = repr(int(self.index))
        elif isinstance(self.index, CodeObject):
            index = repr(self.index)
        else:
            raise TypeError("invalid index: '{}'".format(self.index))
        return "{}[{}]".format(repr(self.seq), index)

    def to_py(self):
        if type(self.index) in [int, float]:
            index = repr(int(self.index))
        elif isinstance(self.index, CodeObject):
            index = self.index.to_py()
        else:
            raise TypeError("invalid index: '{}'".format(self.index))
        return "{}[{}]".format(self.seq.to_py(), index)


class CodeSymbol(CodeObject):

    def __init__(self, name: str, code_type: AnyType):
        self.name = name
        self.code_type = code_type

    def __repr__(self):
        return self.name

    def to_py(self):
        return "state['{}']".format(self.name)


class CodeUnary(CodeObject):

    def __init__(self, op: str, item: CodeObject):
        self.op = op
        self.item = item
        self.code_type = item.code_type

    def __repr__(self):
        op = self.op
        op = op + ' ' if len(op) > 1 else op
        return "{}{}".format(op, repr(self.item))

    def to_py(self):
        op = self.op
        op = op + ' ' if len(op) > 1 else op
        return "{}{}".format(op, self.item.to_py())


class CodeValue(CodeObject):

    def __init__(self, value):
        self.value = value
        self.code_type = get_code_type_for_value(value)

    def __len__(self):
        if type(self.value) is list:
            return len(self.value)
        else:
            return 0

    def __repr__(self):
        return repr(self.value)


class CodeVector(CodeObject):

    def __init__(self, items):
        self.items = items
        self.code_type = ListType.fromList([i.code_type for i in items])

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return "[{}]".format(', '.join([repr(i) for i in self.items]))

    @property
    def is_empty(self):
        return len(self.items) == 0

    @property
    def non_empty(self):
        return len(self.items) > 0

    @property
    def head(self):
        return self.items[0] if len(self.items) > 0 else None

    def to_py(self):
        return "[{}]".format(', '.join([i.to_py() for i in self.items]))


#################################################################################################33

def makeSubscript(seq, index):
    if not isinstance(index.code_type, IntegerType):
        if isinstance(index, CodeValue):
            index = CodeValue(int(index.value))
        else:
            index = CodeFunctionCall('int', [index])
    elif isinstance(index, CodeValue):
        if isinstance(seq, CodeValue) and type(seq.value) is list:
            return seq.value[index.value]
        elif isinstance(seq, CodeVector):
            return seq.items[index.value]
    if isinstance(seq, CodeValue) and len(set(seq.value)) == 1:
        return CodeValue(seq.value[0])
    return CodeSubscript(seq, index)

def makeVector(items: list):
    if all([isinstance(item, CodeValue) for item in items]):
        return CodeValue([item.value for item in items])
    else:
        return CodeVector(items)
