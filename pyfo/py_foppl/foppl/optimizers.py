#
# (c) 2017, Tobias Kohn
#
# 24. Dec 2017
# 28. Dec 2017
#
from .foppl_ast import *

class Optimizer(Walker):

    __binary_ops = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        'and': lambda x, y: x & y,
        'or':  lambda x, y: x | y,
        'xor': lambda x, y: x ^ y
    }

    def __init__(self, compiler=None):
        self.compiler = compiler

    def visit_node(self, node: Node):
        return node

    def visit_binary(self, node: AstBinary):
        left = node.left.walk(self)
        right = node.right.walk(self)
        if isinstance(left, AstValue) and isinstance(right, AstValue):
            if node.op in self.__binary_ops:
                return AstValue(self.__binary_ops[node.op](left.value, right.value))
        return node

    def visit_body(self, node: AstBody):
        items = [n.walk(self) for n in node.body]
        if len(items) == 1:
            return items[0]
        else:
            return AstBody(items)

    def visit_call_get(self, node: AstFunctionCall):
        if len(node.args) == 2:
            vector = node.args[0].walk(self)
            index = node.args[1].walk(self)
            if isinstance(vector, AstValue) and isinstance(index, AstValue):
                return AstValue(vector.value[int(index.value)])
            return AstFunctionCall(node.function, [vector, index])
        return node

    def visit_call_rest(self, node: AstFunctionCall):
        if len(node.args) == 1:
            vector = node.args[0].walk(self)
            if isinstance(vector, AstValue):
                return AstValue(vector.value[1:])
            return AstFunctionCall(node.function, vector)
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

    def visit_if(self, node: AstIf):
        cond = node.cond.walk(self)
        if_body = node.if_body.walk(self)
        else_body = node.else_body.walk(self) if node.else_body else None
        if isinstance(cond, AstValue) and type(cond.value) is bool:
            if cond.value:
                return if_body
            elif else_body:
                return else_body
        return AstIf(cond, if_body, else_body)

    def visit_symbol(self, node: AstSymbol):
        if self.compiler:
            result = self.compiler.scope.find_value(node.name)
            if result:
                return result
        return node

    def visit_unary(self, node: AstUnary):
        item = node.item.walk(self)
        if node.op == '+':
            return item

        elif isinstance(item, AstValue):
            if node.op == '-':
                return AstValue(-item.value)
            elif node.op == 'not':
                return AstValue(not item.value)

        elif isinstance(item, AstUnary):
            if item.op == '-' and node.op == '-':
                return item.item
            if item.op == 'not' and node.op == 'not':
                return item.item

        return node

    def visit_vector(self, node: AstVector):
        children = [child.walk(self) for child in node.get_children()]
        if all(isinstance(child, AstValue) for child in children):
            return AstValue([child.value for child in children])
        return node