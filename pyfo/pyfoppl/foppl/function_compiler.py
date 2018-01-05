#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 21. Dec 2017, Tobias Kohn
# 05. Jan 2018, Tobias Kohn
#
from .foppl_ast import *
from .graphs import *
from .foppl_objects import Symbol
from .foppl_parser import parse
from .foppl_reader import is_alpha, is_alpha_numeric
from .optimizers import Optimizer
from . import Options


def _is_identifier(symbol):
    """
    Checks if the given 'symbol' is a string and a valid Python identifier comprising letters, digits and an underscore.

    :param symbol: Any object which might be an identifier.
    :return:       `True` if the object is an identifier, `False` otherwise.
    """
    if type(symbol) is str and len(symbol) > 0 and is_alpha(symbol[0]):
        return all([is_alpha_numeric(c) for c in symbol])
    else:
        return False


class FunctionCompiler(Walker):
    """
    The function compiler creates the Python code for a function as a lambda-expression. It does not
    keep track of a graph but returns directly a string representation.
    """

    def __init__(self, compiler):
        self.compiler = compiler

    def _optimize(self, node: Node):
        if self.compiler:
            return self.compiler.optimize(node)
        else:
            return node

    def visit_node(self, node: Node):
        # We raise an exception as we want to handle all types of nodes explicitly.
        raise NotImplementedError("{}".format(type(node)))

    def visit_binary(self, node: AstBinary):
        node = self._optimize(node)
        if isinstance(node, AstBinary):
            left = node.left.walk(self)
            right = node.right.walk(self)
            return "({} {} {})".format(left, node.op, right)
        else:
            return node.walk(self)

    def visit_call_exp(self, node: AstFunctionCall):
        arg = self._optimize(node.args[0]).walk(self)
        return "math.exp({})".format(arg)

    def visit_call_get(self, node: AstFunctionCall):
        args = node.args
        seq_expr = args[0].walk(self)
        idx_expr = args[1].walk(self)
        if all(['0' <= x <= '9' for x in idx_expr]):
            return "{}[{}]".format(seq_expr, idx_expr)
        else:
            return "{}[int({})]".format(seq_expr, idx_expr)

    def visit_call_rest(self, node: AstFunctionCall):
        expr = node.args[0].walk(self)
        return "{}[1:]".format(expr)

    def visit_sqrt(self, node: AstSqrt):
        node = self._optimize(node)
        if isinstance(node, AstSqrt):
            return "sqrt({})".format(node.item.visit(self))
        else:
            return node.walk(self)

    def visit_symbol(self, node: AstSymbol):
        if self.compiler:
            result = self.compiler.scope.find_symbol(node.name)
            if result:
                _, expr = result
                if _is_identifier(expr):
                    return "state['{}']".format(expr)
        return "state['{}']".format(node.name)

    def visit_unary(self, node: AstUnary):
        node = self._optimize(node)
        if isinstance(node, AstUnary):
            return "{}{}".format(node.op, node.item.walk(self))
        else:
            return node.walk(self)

    def visit_value(self, node: AstValue):
        return repr(node.value)

    def visit_vector(self, node: AstVector):
        node = self._optimize(node)
        if isinstance(node, AstVector):
            items = [item.walk(self) for item in node.items]
            return "[{}]".format(', '.join(items))
        else:
            return node.walk(self)