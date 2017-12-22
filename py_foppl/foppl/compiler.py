#
# (c) 2017, Tobias Kohn
#
# 21. Dec 2017
# 22. Dec 2017
#
from .foppl_ast import *
from .graphs import *
from .foppl_objects import Symbol
from .foppl_parser import parse

class Scope(object):

    def __init__(self, prev=None):
        self.prev = prev
        self.symbols = {}
        self.functions = {}

    def find(self, name: str):
        if name in self.symbols:
            return self.symbols[name]
        elif self.prev:
            return self.prev.find(name)
        else:
            return None

    def find_function(self, name: str):
        if name in self.functions:
            return self.functions[name]
        elif self.prev:
            return self.prev.find_function(name)
        else:
            return None

    def add(self, name: str, value):
        self.symbols[name] = value

    def add_function(self, name: str, value):
        self.functions[name] = value

    @property
    def is_global_scope(self):
        return self.prev is None


class Compiler(Walker):

    def __init__(self):
        self.__symbol_counter = 20000
        self.scope = Scope()

    def gen_symbol(self, prefix: str):
        self.__symbol_counter += 1
        return "{}{}".format(prefix, self.__symbol_counter)

    def begin_scope(self):
        self.scope = Scope(self.scope)

    def end_scope(self):
        self.scope = self.scope.prev

    def visit_node(self, node: Node):
        return Graph.EMPTY, "None"

    def visit_def(self, node: AstDef):
        if self.scope.is_global_scope:
            if isinstance(node.value, AstFunction):
                self.scope.add_function(node.name, node.value)
            else:
                self.scope.add(node.name, node.value.walk(self))
            return Graph.EMPTY, "None"
        else:
            raise SyntaxError("'def' must be on the global level")

    def visit_let(self, node: AstLet):
        self.begin_scope()
        try:
            for (name, value) in node.bindings:
                if isinstance(name, Symbol):
                    name = name.name
                if isinstance(value, AstFunction):
                    self.scope.add_function(name, value)
                else:
                    self.scope.add(name, value.walk(self))
            result = node.body.walk(self)
        finally:
            self.end_scope()
        return result

    def visit_body(self, node: AstBody):
        result_graph = Graph.EMPTY
        result_expr = "None"
        for item in node.body:
            g, e = item.walk(self)
            result_graph = result_graph.merge(g)
            result_expr = e
        return result_graph, result_expr

    def visit_symbol(self, node: AstSymbol):
        return self.scope.find(node.name)

    def visit_value(self, node: AstValue):
        return Graph.EMPTY, repr(node.value)

    def visit_binary(self, node: AstBinary):
        l_g, l_e = node.left.walk(self)
        r_g, r_e = node.right.walk(self)
        result = "({} {} {})".format(l_e, node.op, r_e)
        return l_g.merge(r_g), result

    def visit_unary(self, node: AstUnary):
        graph, expr = node.item.walk(self)
        return graph, "{}{}".format(node.op, expr)

    def visit_sample(self, node: AstSample):
        dist = node.distribution
        name = self.gen_symbol('x')
        node.id = name
        graph, expr = dist.walk(self)
        graph = graph.merge(Graph({name}, set((v, name) for v in graph.vertices), {name: expr}))
        return graph, name

    def visit_observe(self, node: AstObserve):
        dist = node.distribution
        name = self.gen_symbol('y')
        node.id = name
        graph, expr = dist.walk(self)
        _, obs_expr = node.value.walk(self)
        graph = graph.merge(Graph({name}, set((v, name) for v in graph.vertices), {name: expr}, {name: obs_expr}))
        return graph, name

    def visit_distribution(self, node: AstDistribution):
        return node.repr_with_args(self)

    def visit_vector(self, node: AstVector):
        items = []
        graph = Graph.EMPTY
        for item in node.get_children():
            g, expr = item.walk(self)
            graph = graph.merge(g)
            items.append(expr)
        return graph, "[{}]".format(", ".join(items))

    def visit_call_get(self, node: AstFunctionCall):
        args = node.args
        if len(args) == 2:
            seq_graph, seq_expr = args[0].walk(self)
            idx_graph, idx_expr = args[1].walk(self)
            return seq_graph.merge(idx_graph), "{}[int({})]".format(seq_expr, idx_expr)
        else:
            raise SyntaxError("'get' expects exactly two arguments")

    def visit_call_rest(self, node: AstFunctionCall):
        args = node.args
        if len(args) == 1:
            graph, expr = args[0].walk(self)
            return graph, "{}[1:]".format(expr)
        else:
            raise SyntaxError("'rest' expects exactly one argument")


def compile(source):
    ast = parse(source)
    compiler = Compiler()
    return compiler.walk(ast)
