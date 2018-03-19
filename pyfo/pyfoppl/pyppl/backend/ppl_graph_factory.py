#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 12. Mar 2018, Tobias Kohn
# 19. Mar 2018, Tobias Kohn
#
from pyppl.ppl_ast import *
from pyppl.graphs import *
from .ppl_code_generator import CodeGenerator
from .ppl_graph_codegen import GraphCodeGenerator


class GraphFactory(object):

    def __init__(self, code_generator=None):
        if code_generator is None:
            code_generator = CodeGenerator()
            code_generator.state_object = 'state'
        self._counter = 30000
        self.nodes = []
        self.code_generator = code_generator

    def _generate_code_for_node(self, node: AstNode):
        return self.code_generator.visit(node)

    def generate_symbol(self, prefix: str):
        self._counter += 1
        return prefix + str(self._counter)

    def create_node(self, parents: set):
        assert type(parents) is set
        return None

    def create_condition_node(self, test: AstNode, parents: set):
        name = self.generate_symbol('cond_')
        code = self._generate_code_for_node(test)
        if isinstance(test, AstCompare) and is_zero(test.right) and test.second_right is None:
            result = ConditionNode(name, ancestors=parents, condition=code,
                                   function=self._generate_code_for_node(test.left), op=test.op)
        else:
            result = ConditionNode(name, ancestors=parents, condition=code)
        self.nodes.append(result)
        return result

    def create_data_node(self, data: AstNode, parents: Optional[set]=None):
        if parents is None:
            parents = set()
        name = self.generate_symbol('data_')
        code = self._generate_code_for_node(data)
        result = DataNode(name, ancestors=parents, data=code)
        self.nodes.append(result)
        return result

    def create_observe_node(self, dist: AstNode, value: AstNode, parents: set):
        if isinstance(dist, AstCall):
            func = dist.function_name
            args = [self._generate_code_for_node(arg) for arg in dist.args]
            args = dist.add_keywords_to_args(args)
        else:
            func = None
            args = None
        name = self.generate_symbol('y')
        d_code = self._generate_code_for_node(dist)
        v_code = self._generate_code_for_node(value)
        obs_value = value.value if is_value(value) else None
        result = Vertex(name, ancestors=parents, distribution_code=d_code, distribution_name=_get_dist_name(dist),
                        distribution_args=args, distribution_func=func, observation=v_code,
                        observation_value=obs_value)
        self.nodes.append(result)
        return result

    def create_sample_node(self, dist: AstNode, size: int, parents: set):
        if isinstance(dist, AstCall):
            func = dist.function_name
            args = [self._generate_code_for_node(arg) for arg in dist.args]
            args = dist.add_keywords_to_args(args)
        else:
            func = None
            args = None
        name = self.generate_symbol('x')
        code = self._generate_code_for_node(dist)
        result = Vertex(name, ancestors=parents, distribution_code=code, distribution_name=_get_dist_name(dist),
                        distribution_args=args, distribution_func=func, sample_size=size)
        self.nodes.append(result)
        return result

    def generate_code(self, *, class_name: Optional[str] = None, imports: Optional[str]=None,
                      base_class: Optional[str]=None):
        code_gen = GraphCodeGenerator(self.nodes, self.code_generator.state_object,
                                      imports=imports if imports is not None else '')
        return code_gen.generate_model_code(class_name=class_name, base_class=base_class)


def _get_dist_name(dist: AstNode):
    if isinstance(dist, AstCall):
        result = dist.function_name
        if result.startswith('dist.'):
            result = result[5:]
        return result
    else:
        return None
