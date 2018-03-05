#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 24. Dec 2017, Tobias Kohn
# 20. Jan 2018, Tobias Kohn
#
from .foppl_ast import *
from . import Options

class Scope(object):

    def __init__(self, prev=None):
        self.prev = prev
        self.values = {}

    def find(self, name: str):
        if name in self.values:
            return self.values[name]
        elif self.prev:
            return self.prev.find(name)
        else:
            return None

    def add(self, name: str, value):
        self.values[name] = value

class Optimizer(Walker):

    __binary_ops = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '**': lambda x, y: x ** y,
        'and': lambda x, y: x & y,
        'or':  lambda x, y: x | y,
        'xor': lambda x, y: x ^ y
    }

    __inverse_cmp = {
        '>': '<=',
        '>=': '<',
        '<=': '>',
        '<': '>=',
    }

    def __init__(self, compiler=None):
        self.compiler = compiler
        self.scope = None

    def __begin_scope(self):
        self.scope = Scope(self.scope)

    def __end_scope(self):
        if self.scope:
            self.scope = self.scope.prev

    def __apply_function(self, function: AstFunction, args: list):
        if isinstance(function, AstFunction) and len(function.params) == len(args) and \
                all([isinstance(arg, AstValue) for arg in args]):
            self.__begin_scope()
            try:
                for (name, value) in zip(function.params, args):
                    if isinstance(name, Symbol):
                        name = name.name
                    self.scope.add(name, value)
                result = function.body.walk(self)
            finally:
                self.__end_scope()
            return result
        else:
            return None

    def visit_node(self, node: Node):
        return node

    def visit_binary(self, node: AstBinary):
        left = node.left.walk(self)
        right = node.right.walk(self)

        if isinstance(left, AstValue) and isinstance(right, AstValue):
            if node.op in self.__binary_ops:
                return AstValue(self.__binary_ops[node.op](left.value, right.value))

        if isinstance(left, AstValue):
            if (left.value == 0 and node.op in ['+', 'or']) or \
               (left.value == 1 and node.op in ['*']):
                return right

            if left.value == 0 and node.op == '-':
                return AstUnary('-', right)

        if isinstance(right, AstValue):
            if (right.value == 0 and node.op in ['+', '-', 'or']) or \
               (right.value == 1 and node.op in ['*', '/']):
                return left

        return node

    def visit_body(self, node: AstBody):
        items = [n.walk(self) for n in node.body]
        if len(items) == 1:
            return items[0]
        else:
            return AstBody(items)

    def visit_call_conj(self, node: AstFunctionCall):
        if len(node.args) == 2:
            vector = node.args[0].walk(self)
            item = node.args[1].walk(self)
            if isinstance(vector, AstValue) and isinstance(item, AstValue) and type(vector.value) is list:
                return AstValue(vector.value + [item])
            return AstFunctionCall(node.function, [vector, item])
        return node

    def visit_call_get(self, node: AstFunctionCall):
        if len(node.args) == 2:
            vector = node.args[0].walk(self)
            index = node.args[1].walk(self)
            if isinstance(vector, AstValue) and isinstance(index, AstValue):
                return AstValue(vector.value[int(index.value)])
            if isinstance(index, AstValue) and index.value == -1:
                if isinstance(vector, AstFunctionCall) and vector.function == 'conj' and len(vector.args) >= 2:
                    return vector.args[-1]
            return AstFunctionCall(node.function, [vector, index])
        return node

    def visit_call_map(self, node: AstFunctionCall):
        wrap = lambda x: x if isinstance(x, Node) else AstValue(x)
        if len(node.args) >= 2:
            function = node.args[0].walk(self)
            vectors = [arg.walk(self) for arg in node.args[1:]]
            if all([isinstance(v, AstValue) and type(v.value) is list for v in vectors]):
                vectors = [[wrap(w) for w in v.value] for v in vectors]
                vectors = list(zip(*vectors))
                if all([all([isinstance(v, AstValue) for v in V]) for V in vectors]):
                    return AstVector([AstFunctionCall(function, list(arg)) for arg in vectors]).walk(self)
        return node

    def visit_call_rest(self, node: AstFunctionCall):
        if len(node.args) == 1:
            vector = node.args[0].walk(self)
            if isinstance(vector, AstValue):
                return AstValue(vector.value[1:])
            return AstFunctionCall(node.function, [vector])
        return node

    def visit_compare(self, node: AstCompare):
        left = node.left.walk(self)
        right = node.right.walk(self)
        if isinstance(left, AstValue) and isinstance(right, AstValue):
            op = node.op
            value_l = left.value
            value_r = right.value
            if op == '=':
                return AstValue(value_l == value_r)
            elif op == '<':
                return AstValue(value_l < value_r)
            elif op == '>':
                return AstValue(value_l > value_r)
            elif op == '<=':
                return AstValue(value_l <= value_r)
            elif op == '>=':
                return AstValue(value_l >= value_r)
        return AstCompare(node.op, left, right)

    def visit_functioncall(self, node: AstFunctionCall):
        function = node.function
        args = [arg.walk(self) for arg in node.args]
        if all([isinstance(arg, AstValue) for arg in args]) and self.compiler:
            if isinstance(function, AstSymbol):
                f = self.compiler.scope.find_function(function.name)
                if f is not None:
                    function = f
            elif isinstance(function, str):
                f = self.compiler.scope.find_function(function)
                if f is not None:
                    function = f

            if isinstance(function, AstFunction):
                result = self.__apply_function(function, args)
                if isinstance(result, AstValue):
                    return result

        if len(args) > 0:
            return AstFunctionCall(node.function, args)
        else:
            return node

    def visit_if(self, node: AstIf):
        cond = node.cond.walk(self)
        if_body = node.if_body.walk(self)
        else_body = node.else_body.walk(self) if node.else_body else None

        if isinstance(cond, AstValue) and type(cond.value) is bool:
            if cond.value:
                return if_body
            elif else_body:
                return else_body

        # We get rid of a `not` in an `if`-`else`-expression by swaping
        # the `if`- and `else`-parts.
        if else_body and isinstance(cond, AstUnary) and cond.op == 'not':
            else_body, if_body = if_body, else_body
            cond = cond.item

        return AstIf(cond, if_body, else_body)

    def visit_loop(self, node: AstLoop):
        if node.iter_count == 0:
            return node.arg.walk(self)
        elif node.iter_count == 1:
            func = node.function.name if isinstance(node.function, AstSymbol) else node.function
            result = AstFunctionCall(func, [AstValue(0), node.arg] + node.args)
            return result.walk(self)
        else:
            arg = node.arg.walk(self) if node.arg else None
            args = [a.walk(self) for a in node.args]
            return AstLoop(node.iter_count, arg, node.function, args)

    def visit_sqrt(self, node: AstSqrt):
        from math import sqrt
        item = node.item.walk(self)

        if isinstance(item, AstValue):
            value = item.value
            if type(value) in [int, float]:
                return AstValue(sqrt(value))
            elif type(value) is list and all([type(x) in [int, float] for x in value]):
                return AstValue([sqrt(x) for x in value])

        if isinstance(item, AstVector):
            node = AstVector([AstSqrt(x) for x in item.items])
            return node.walk(self)

        return AstSqrt(item)

    def visit_symbol(self, node: AstSymbol):
        if self.scope:
            result = self.scope.find(node.name)
            if result:
                return result
        if self.compiler:
            result = self.compiler.scope.find_value(node.name)
            if result:
                if type(result) in [int, float, list, str, bool]:
                    return AstValue(result)
                else:
                    return result
            result = self.compiler.scope.find_symbol(node.name)
            if type(result) is tuple and len(result) == 2:
                graph, value = result
                if graph.is_empty and isinstance(value, AstValue):
                    return value
        return node

    def visit_unary(self, node: AstUnary):
        # reduce two applications of the same unary operation
        if isinstance(node.item, AstUnary) and node.op == node.item.op:
            if node.op in ['+', '-', 'not']:
                return node.item.item.walk(self)

        item = node.item.walk(self)
        # plus-signs are redundant
        if node.op == '+':
            return item

        # apply the unary operation to values
        elif isinstance(item, AstValue):
            if node.op == '-':
                return AstValue(-item.value)
            elif node.op == 'not':
                return AstValue(not item.value)

        # reduce two applications of the same unary operation
        elif isinstance(item, AstUnary):
            if item.op == '-' and node.op == '-':
                return item.item
            if item.op == 'not' and node.op == 'not':
                return item.item

        # simplify the patterns 'NOT COMPARISON', e.g. `not x < 0` to `x >= 0`
        elif isinstance(item, AstCompare) and node.op == 'not':
            if Options.uniform_conditionals:
                if item.op == '<':
                    return AstCompare('>=', item.left, item.right)
                elif item.op == '>':
                    return AstCompare('>=', item.right, item.left)
            else:
                if item.op in self.__inverse_cmp:
                    return AstCompare(self.__inverse_cmp[item.op], item.left, item.right)

        return AstUnary(node.op, item)

    def visit_vector(self, node: AstVector):
        children = [child.walk(self) for child in node.get_children()]
        if all(isinstance(child, AstValue) for child in children):
            return AstValue([child.value for child in children])
        return node