#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 21. Dec 2017, Tobias Kohn
# 16. Jan 2018, Tobias Kohn
#
from .foppl_ast import *
from .graphs import *
from .foppl_objects import Symbol
from .foppl_parser import parse
from .foppl_reader import is_alpha, is_alpha_numeric
from .optimizers import Optimizer
from .function_compiler import FunctionCompiler
from .foppl_distributions import distribution_params
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


class Scope(object):
    """
    The scope is basically a stack of dictionaries, implemented as a simply
    linked list of Scope-classes. Functions and other symbols/values are
    stored in distinct dictionaries, and hence also have distinct namespaces.

    If the value of a symbol is of type AstValue, we store this ast-node as
    well. This is then used by the optimizer.
    """

    def __init__(self, prev=None):
        self.prev = prev
        self.symbols = {}
        self.functions = {}
        self.values = {}

    def find_function(self, name: str):
        if name in self.functions:
            return self.functions[name]
        elif self.prev:
            return self.prev.find_function(name)
        else:
            return None

    def find_symbol(self, name: str):
        if name in self.symbols:
            return self.symbols[name]
        elif self.prev:
            return self.prev.find_symbol(name)
        else:
            return None

    def find_value(self, name: str):
        if name in self.values:
            return self.values[name]
        elif self.prev:
            return self.prev.find_value(name)
        else:
            return None

    def add_function(self, name: str, value):
        self.functions[name] = value

    def add_symbol(self, name: str, value):
        self.symbols[name] = value
        self.values[name] = None

    def add_value(self, name: str, value):
        if name in self.symbols:
            self.values[name] = value

    @property
    def is_global_scope(self):
        return self.prev is None


class Compiler(Walker):
    """
    The compiler walks the AST and creates a graph representing the FOPPL model. Each `visit_XXX`-method
    returns a tuple comprising a graph and a Python expression (as string).

    In order to support symbol/variable bindings (through `let`, `def` and functions), the compiler uses a stack
    of 'scopes'. For each new scope, a new item is pushed onto the stack and remains there to the end of the
    scope. All bindings are then put into the dictionaries of the current scope, while searching for a binding
    includes all open scopes.

    The compiler must also keep track of conditional expressions/statements. Particularly for the `observe`-
    statement, the compiler must know if it is inside a conditional branch. It can then create the appropriate
    edges within the graph and make some statements dependent on the condition. In order to keep track of
    conditional execution, the compiler uses a stack `conditions`; the top element is always the current
    condition (if any).
    """

    def __init__(self):
        # Used to create 'unique' symbols in `gen_symbol`:
        self.__symbol_counter = 20000
        # The scope makes sure all symbols defined by `let` and `def` are available:
        self.scope = Scope()
        # The optimizer is used to simplify expressions, e.g., 2+3 -> 5:
        self.optimizer = Optimizer(self)
        # The function-compiler is used to create lambda-functions
        self.function_compiler = FunctionCompiler(self)
        # When inside a conditional expression (if), we keep track of the current conditions with this stack:
        self.conditions = []

    def resolve_symbol(self, name: str):
        return self.scope.find_symbol(name)

    def gen_symbol(self, prefix: str):
        self.__symbol_counter += 1
        return "{}{}".format(prefix, self.__symbol_counter)

    def begin_scope(self):
        self.scope = Scope(self.scope)

    def end_scope(self):
        if self.scope.is_global_scope:
            raise RuntimeError("cannot close global scope/namespace")
        self.scope = self.scope.prev

    def begin_condition(self, cond):
        self.conditions.append(cond)

    def end_condition(self):
        if len(self.conditions) > 0:
            self.conditions.pop()

    def current_condition(self):
        if len(self.conditions) > 0:
            return self.conditions[-1]
        else:
            return None

    def optimize(self, node: Node):
        if node and self.optimizer:
            return node.walk(self.optimizer)
        return node

    def apply_function(self, function: AstFunction, args: list):
        """
        Applies a function to a series of arguments by simple substitution/replacement.

        The effect of this method is the same as if the arguments were bound to the parameters by `let`
        and the body of the function was then evaluated. This means, that the method opens a new scope,
        binds all arguments to their respective parameters, and then evaluates the body through walking
        it using the compiler as the walker.

        The major limitation of this approach is that values might actually be evaluated more than once
        and not in the order of the arguments. In other words: this works only if we assume that all
        arguments/values are pure in the functional sense, i.e. without side effects.

        :param function:  The function to be applied, as a `AstFunction`-object.
        :param args:      All arguments as a list of AST-nodes.
        :return:          A tuple (graph, expr).
        """
        assert isinstance(function, AstFunction)
        if len(function.params) != len(args):
            raise SyntaxError("wrong number of arguments for '{}'".format(function.name))
        self.begin_scope()
        try:
            for (name, value) in zip(function.params, args):
                if isinstance(name, Symbol):
                    name = name.name
                if isinstance(value, AstFunction):
                    self.scope.add_function(name, value)
                else:
                    if type(value) in [int, bool, str, float]:
                        self.scope.add_symbol(name, (Graph.EMPTY, AstValue(value)))
                    else:
                        self.scope.add_symbol(name, value.walk(self))
            result = function.body.walk(self)
        finally:
            self.end_scope()
        return result

    def define(self, name, node):
        """
        Binds the name to the node provided. The node can be a function or any value/node.

        :param name:  The name/symbol to be bound.
        :param node:  The value or function to be bound.
        :return:      None
        """
        if isinstance(name, Symbol):
            name = name.name

        if isinstance(node, AstFunction):
            self.scope.add_function(name, node)
        else:
            node = self.optimize(node)
            value = node.walk(self)
            if _is_identifier(value[1]):
                value[0].add_original_name(name, value[1])
            self.scope.add_symbol(name, value)
            if isinstance(node, AstValue):
                self.scope.add_value(name, node.value)

    def visit_node(self, node: Node):
        # We raise an exception as we want to handle all types of nodes explicitly.
        raise NotImplementedError("{}".format(type(node)))

    def visit_binary(self, node: AstBinary):
        node = self.optimize(node)
        if isinstance(node, AstBinary):
            l_g, l_e = node.left.walk(self)
            r_g, r_e = node.right.walk(self)
            result = "({} {} {})".format(l_e, node.op, r_e)
            return l_g.merge(r_g), result
        else:
            return node.walk(self)

    def visit_body(self, node: AstBody):
        result_graph = Graph.EMPTY
        result_expr = "None"
        for item in node.body:
            g, e = item.walk(self)
            result_graph = result_graph.merge(g)
            result_expr = e
        return result_graph, result_expr

    def visit_call_exp(self, node: AstFunctionCall):
        node = self.optimize(node)
        if isinstance(node, AstFunctionCall) and node.function == 'exp':
            if len(node.args) == 1:
                graph, arg = self.optimize(node.args[0]).walk(self)
                return graph, "math.exp({})".format(arg)
            else:
                raise SyntaxError("'exp' requires exactly one argument")
        else:
            return node.walk(self)

    def visit_call_get(self, node: AstFunctionCall):
        node = self.optimize(node)
        if isinstance(node, AstFunctionCall) and node.function == 'get':
            args = node.args
            if len(args) == 2:
                seq_graph, seq_expr = args[0].walk(self)
                idx_graph, idx_expr = args[1].walk(self)
                if len(seq_expr) > 2 and seq_expr[0] == '[' and seq_expr[-1] == ']' and \
                        _is_identifier(seq_expr[1:-1]) and idx_expr in ['0', '-1']:
                    return seq_graph.merge(idx_graph), seq_expr[1:-1]
                if all(['0' <= x <= '9' for x in idx_expr]) or idx_expr == '-1':
                    return seq_graph.merge(idx_graph), "{}[{}]".format(seq_expr, idx_expr)
                else:
                    return seq_graph.merge(idx_graph), "{}[int({})]".format(seq_expr, idx_expr)
            else:
                raise SyntaxError("'get' expects exactly two arguments")
        else:
            return node.walk(self)

    def visit_call_map(self, node: AstFunctionCall):
        node = self.optimize(node)
        if not (isinstance(node, AstFunctionCall) and node.function == "map"):
            return node.walk(self)

        f = node.args[0]
        args = [self.optimize(arg) for arg in node.args[1:]]
        if all([isinstance(arg, AstValue) for arg in args]):
            args = [arg.value for arg in args]
        elif isinstance(f, AstSymbol):
            graph, expr = AstVector(args).walk(self)
            graph.add_used_function(f.name)
            return graph, "list(map({}, {}))".format(f.name, expr)
        else:
            raise RuntimeError("Cannot apply 'map' to {}".format(args))
        if len(args) > 0 and isinstance(f, AstFunction):
            if len(args) > 1:
                mangled_args = []
                L = min([len(arg) for arg in args])
                for i in range(L):
                    mangled_args.append([arg[i] for arg in args])
                args = mangled_args
            else:
                args = [[arg] for arg in args[0]]
            Vec = [self.apply_function(f, arg) for arg in args]
            graph = merge(*[v[0] for v in Vec])
            expr = "[{}]".format(', '.join([v[1] for v in Vec]))
            return graph, expr

        else:
            raise SyntaxError("'map' expects a function and at least one vector")

    def visit_call_rest(self, node: AstFunctionCall):
        node = self.optimize(node)
        if isinstance(node, AstFunctionCall) and node.function == 'rest':
            args = node.args
            if len(args) == 1:
                graph, expr = args[0].walk(self)
                return graph, "{}[1:]".format(expr)
            else:
                raise SyntaxError("'rest' expects exactly one argument")
        else:
            return node.walk(self)

    def visit_compare(self, node: AstCompare):
        node = self.optimize(node)
        if isinstance(node, AstCompare):
            l_g, l_e = node.left.walk(self)
            r_g, r_e = node.right.walk(self)
            graph = l_g.merge(r_g)
            expr = "({} {} {}){}".format(l_e, node.op, r_e, Options.conditional_suffix)
            if not graph.is_empty:
                cond_name = self.gen_symbol('cond_')
                cur_cond = self.current_condition()
                if cur_cond:
                    graph = graph.merge(Graph({cur_cond}, {(cur_cond, cond_name)}))

                if Options.uniform_conditionals and node.op == '>=' and \
                        isinstance(node.right, AstValue) and node.right.value == 0:
                    f_name = self.gen_symbol('f')
                    graph = graph.merge(Graph({f_name}, {(v, f_name) for v in graph.vertices}, {f_name: l_e}))
                    graph = graph.merge(Graph({cond_name}, {(f_name, cond_name)},
                                              {cond_name: "({} >= 0){}".format(f_name, Options.conditional_suffix)}))
                    graph.add_conditional_function(cond_name, f_name)
                    if self.function_compiler:
                        try:
                            graph.add_conditional_function(f_name,
                                "lambda state: {}".format(self.function_compiler.walk(node.left)))
                        except NotImplementedError:
                            pass
                else:
                    graph = graph.merge(Graph({cond_name}, {(v, cond_name) for v in graph.vertices},
                                              {cond_name: expr}))
                expr = cond_name

            return graph, expr
        else:
            return node.walk(self)

    def visit_def(self, node: AstDef):
        if self.scope.is_global_scope:
            self.define(node.name, node.value)
            return Graph.EMPTY, "None"
        else:
            raise SyntaxError("'def' must be on the global level")

    def visit_distribution(self, node: AstDistribution):
        graph = Graph.EMPTY
        args = []
        for arg in node.args:
            gr, expr = arg.walk(self)
            graph = graph.merge(gr)
            args.append(expr)
        params = distribution_params[node.name].copy()
        if len(params) == len(args):
            for i in range(len(params)):
                params[i] += '=' + args[i]

            return graph, "dist.{}({})".format(node.name, ', '.join(params))
        else:
            raise SyntaxError("wrong number of arguments for distribution '{}'".format(node.name))

    def visit_distribution_categorical(self, node: AstDistribution):
        if len(node.args) >= 1 and isinstance(node.args[0], AstValue):
            ps = node.args[0].value
            if type(ps) is list and all([type(i) is list for i in ps]):
                size = len(ps), min([len(i) for i in ps])
            elif type(ps) is list and all([type(i) in [int, float] for i in ps]):
                size = (len(ps), 0)
            else:
                size = (0, 0)
            node.size = size
        return self.visit_distribution(node)

    def visit_expr(self, node: AstExpr):
        return node.value

    def visit_functioncall(self, node: AstFunctionCall):
        # NB: some functions are handled directly by visit_call_XXX-methods!
        func = node.function
        if isinstance(func, AstSymbol):
            func = func.name
        if type(func) is str:
            func_name = func
            func = self.scope.find_function(func)
        else:
            func_name = None
        if isinstance(func, AstFunction):
            return self.apply_function(func, node.args)
        elif func_name:
            exprs = []
            graph = Graph.EMPTY
            for a in node.args:
                g, e = a.walk(self)
                graph = graph.merge(g)
                exprs.append(e)
            graph.add_used_function(func_name)
            return graph, "{}({})".format(func_name, ", ".join(exprs))
        else:
            raise SyntaxError("'{}' is not a function".format(node.function))

    def visit_if(self, node: AstIf):
        # The optimizer might detect that the condition is static (can be determined at compile time), and
        # return just the if- or else-body, respectively. We only continue with this function if the node
        # is still an if-node after optimization.
        node = self.optimize(node)
        if not isinstance(node, AstIf):
            return node.walk(self)

        # We create two symbol for the entire if-expression.
        name = self.gen_symbol('c')

        # Compile the condition. If we are inside another conditional expression already, we link the new
        # condition to the current condition through a new edge in the graph, so that the new condition
        # depends on the current one.
        cond_graph, cond_name = node.cond.walk(self)

        # Compile if- and else-body (if present). During this compilation step, we push the new condition onto
        # the condition stack, so that expressions and statements within the branches are made aware of being
        # inside a conditional branch.
        _cond_name = cond_name[4:] if cond_name.startswith("not ") else cond_name
        if not cond_graph.is_empty and _is_identifier(_cond_name) and _cond_name.startswith("cond"):
            self.begin_condition(_cond_name)
            if_graph, if_body = node.if_body.walk(self)
            if node.else_body:
                else_graph, else_body = node.else_body.walk(self)
            else:
                else_graph, else_body = Graph.EMPTY, "None"
            self.end_condition()

        else:
            if_graph, if_body = node.if_body.walk(self)
            if node.else_body:
                else_graph, else_body = node.else_body.walk(self)
            else:
                else_graph, else_body = Graph.EMPTY, "None"

        # We put together the final if-expression as well as the graph. For the graph, we add all edges as needed.
        expr = "{} if {} else {}".format(if_body, cond_name, else_body)
        graph = cond_graph
        graph = graph.merge(if_graph.add_condition(cond_name))
        graph = graph.merge(else_graph.add_condition("not "+cond_name if _cond_name == cond_name else _cond_name))
        graph = graph.merge(Graph({name}, set((v, name) for v in graph.vertices), {name: expr}))
        return graph, name

    def visit_let(self, node: AstLet):
        self.begin_scope()
        try:
            for (name, value) in node.bindings:
                if isinstance(name, Symbol):
                    name = name.name
                self.define(name, value)
            result = node.body.walk(self)
        finally:
            self.end_scope()
        return result

    def visit_loop(self, node: AstLoop):
        node = self.optimize(node)
        if isinstance(node, AstLoop):
            if isinstance(node.function, AstSymbol):
                function = self.scope.find_function(node.function.name)
            elif isinstance(node.function, AstFunction):
                function = node.function
            else:
                raise SyntaxError("'loop' requires a function")
            iter_count = node.iter_count
            i = 0
            args = [AstExpr(*a.walk(self)) for a in node.args]
            result = node.arg.walk(self)
            while i < iter_count:
                result = self.apply_function(function, [AstValue(i), AstExpr(*result)] + args)
                i += 1
            return result
        else:
            return node.walk(self)

    def visit_observe(self, node: AstObserve):
        dist = node.distribution
        name = self.gen_symbol('y')
        node.id = name
        graph, expr = dist.walk(self)
        if hasattr(dist, 'size'):
            graph.add_distribution_size(name, dist.size)
        _, obs_expr = node.value.walk(self)
        graph = graph.merge(Graph({name}, set((v, name) for v in graph.vertices), {name: expr}, {name: obs_expr}))
        cond = self.current_condition()
        if cond:
            graph = graph.merge(Graph({cond}, {(cond, name)}))
        return graph, name

    def visit_sample(self, node: AstSample):
        dist = node.distribution
        name = self.gen_symbol('x')
        node.id = name
        graph, expr = dist.walk(self)
        if hasattr(dist, 'size'):
            graph.add_distribution_size(name, dist.size)
        graph = graph.merge(Graph({name}, set((v, name) for v in graph.vertices), {name: expr}))
        cond = self.current_condition()
        if cond:
            graph = graph.merge(Graph({cond}, {(cond, name)}))
        return graph, name

    def visit_sqrt(self, node: AstSqrt):
        node = self.optimize(node)
        if isinstance(node, AstSqrt):
            graph, expr = node.item.walk(self)
            return graph, "math.sqrt({})".format(expr)
        else:
            return node.walk(self)

    def visit_symbol(self, node: AstSymbol):
        result = self.scope.find_symbol(node.name)
        if result:
            return result
        else:
            raise SyntaxError("Unknown symbol: '{}'".format(node.name))

    def visit_unary(self, node: AstUnary):
        node = self.optimize(node)
        if isinstance(node, AstUnary):
            graph, expr = node.item.walk(self)
            return graph, "{}{}".format(node.op, expr)
        else:
            return node.walk(self)

    def visit_value(self, node: AstValue):
        # We must use `repr` here instead of `str`, as `repr` returns a string with delimiters.
        return Graph.EMPTY, repr(node.value)

    def visit_vector(self, node: AstVector):
        items = []
        graph = Graph.EMPTY
        for item in node.get_children():
            g, expr = item.walk(self)
            graph = graph.merge(g)
            items.append(expr)
        return graph, "[{}]".format(", ".join(items))


def compile(source):
    if isinstance(source, Node):
        ast = source
    else:
        ast = parse(source)
    compiler = Compiler()
    return compiler.walk(ast)
