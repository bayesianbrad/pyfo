#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 20. Dec 2017, Tobias Kohn
# 22. Jan 2018, Tobias Kohn
#
"""
# PyFOPPL: Vertices and Graph

The graph and its vertices, as provided in this module, form the backbone of the graphical model.

## The Graphical Model

The probabilistic model is compiled into a graph, where each vertex corresponds to the sampling of a distribution, or
an observation of a stochastic value, respectively. Any dependencies between these stochastic vertices are captured by
the edges (called `arcs`). Consider the (abstract) example:
```
x1 = sample(normal(0, 1))
x2 = sample(normal(x1, 4))
x = x1 + x2 / 2
y = observe(normal(x, 1), 3)
```
Here, we have three vertices: `x1` and `x2` are sampled values, and the third vertex is the observation `y` on the last
line. Obviously, `x2` depends on `x1`, and `y` depends on both `x1` and `x2`. The value `x` is not a vertex as it does
not do any sampling or observation on its own. The graphical model then looks as follows:

  [x1] --> [x2]
    \     /
     \   /
      v v
      [y]

The variable `x` is eliminated during the compilation process and completely inlined into `y`.

## The Graph Around the Graphical Model

In addition to the vertices above, there are additional nodes, which are not part of the graphical model itself. On the
one hand, we put data items (vectors/lists/tables/tensors) into data nodes to simplify the generated code, and not
having to inline large amounts of data. On the other hand, conditional `if`-statements are also expressed as special
nodes.

In other words: we embed the graphical model with its vertices and arcs into a graph of additional nodes and edges.
Each node is a fully fledged object of its own, providing a rich set of information for further analysis.

## Computation

The actual computation associated with any node is encoded in a `evaluate(state)` function. Here, the `state` is a
dictionary holding variables, in particular the values computed by other nodes. The example above is -- in principle --
rewritten to:
```
state['x1'] = dist.Normal(0, 1).sample()
state['x2'] = dist.Normal(state['x1'], 4).sample()
state['y'] = 3   # The observed value
```
Using the `state`-dictionary allows us to carry the state across different nodes and complete the entire computation.
In addition, the model can also compute the log-pdf by a rewrite as follows (based upon a prior sampling to compute
initial values for the variables in the `state`-dictionary):
```
log_pdf = 0.0
log_pdf += dist.Normal(0, 1).log_pdf(state['x1'])
log_pdf += dist.Normal(state['x1'], 4).log_pdf(state['x2'])
log_pdf += dist.Normal((state['x1'] + state['x2'])/2, 1).log_pdf(3)
```
Both computations are facilitated by the methods `update` and `update_pdf` of the node.
"""
from . import Options, runtime
from .foppl_distributions import distributions
from .basic_imports import *

####################################################################################################

class GraphNode(object):
    """
    The base class for all nodes, including the actual graph vertices, but also conditionals, data, and possibly
    parameters.

    Each node has a name, which is usually generated automatically. The generation of the name is based on a simple
    counter. This generated name (i.e. the counter value inside the name) is used later on to impose a compute order
    on the nodes (see the method `get_ordered_list_of_all_nodes` in the `graph`). Hence, you should not change the
    naming scheme unless you know exactly what you are doing!

    The set of ancestors provides the edges for the graph and the graphical model, respectively. Note that all
    ancestors are always vertices. Conditions, parameters, data, etc. are hold in other fields. This ensures that by
    looking at the ancestors of vertices, we get the pure graphical model.

    Finally, the methods `evaluate`, `update` and `update_pdf` are used by the model to sample values and compute
    log-pdf, etc. Of course, `evaluate` is just a placeholder here so as to define a minimal interface. Usually, you
    will use `update` and `update_pdf` instead of `evaluate`. However, given a `state`-dictionary holding all the
    necessary values, it is save to call `evaluate`.
    """

    name = ""
    ancestors = set()
    line_number = -1

    __symbol_counter__ = 30000

    @classmethod
    def __gen_symbol__(cls, prefix:str):
        cls.__base__.__symbol_counter__ += 1
        return "{}{}".format(prefix, cls.__base__.__symbol_counter__)

    @property
    def display_name(self):
        return self.name[-3:]

    def evaluate(self, state):
        raise NotImplemented()

    def update(self, state: dict):
        result = self.evaluate(state)
        state[self.name] = result
        return result

    def update_pdf(self, state: dict):
        self.update(state)
        return 0.0


####################################################################################################

# This is used to generate the various `evaluate`-methods later on.
_LAMBDA_PATTERN_ = "lambda state: {}"


class ConditionNode(GraphNode):
    """
    A `ConditionNode` represents a condition that depends on stochastic variables (vertices). It is not directly
    part of the graphical model, but you can think of conditions to be attached to a specific vertex.

    Usually, we try to transform all conditions into the form `f(state) >= 0` (this is not possible for `f(X) == 0`,
    through). However, if the condition satisfies this format, the node object has an associated `function`, which
    can be evaluated on its own. In other words: you can not only check if a condition is `True` or `False`, but you
    can also gain information about the 'distance' to the 'border'.
    """

    def __init__(self, *, name:str=None, condition=None, ancestors:set=None, op:str='?', function=None,
                 line_number:int=-1):
        from .code_objects import CodeCompare, CodeValue
        if name is None:
            name = self.__class__.__gen_symbol__('cond_')
        if ancestors is None:
            ancestors = set()
        if function:
            if op == '?':
                op = '>='
            if condition is None:
                condition = CodeCompare(function, op, CodeValue(0))
        self.name = name
        self.ancestors = ancestors
        self.op = op
        self.condition = condition
        self.function = function
        self.code = _LAMBDA_PATTERN_.format(condition.to_py() if condition else "None") + Options.conditional_suffix
        self.function_code = _LAMBDA_PATTERN_.format(function.to_py() if function else "None")
        self.evaluate = eval(self.code)
        self.evaluate_function = eval(self.function_code)
        self.line_number = line_number
        for a in ancestors:
            if isinstance(a, Vertex):
                a._add_dependent_condition(self)

    def __repr__(self):
        if self.function:
            result = "{f} {o} 0\n\tFunction: {f}".format(f=repr(self.function), o=self.op)
        elif self.condition:
                result = repr(self.condition)
        else:
            result = "???"
        ancestors = ', '.join([v.name for v in self.ancestors])
        result = "{}:\n\tAncestors: {}\n\tCondition: {}".format(self.name, ancestors, result)
        if Options.debug:
            result += "\n\tRelation: {}".format(self.op)
            result += "\n\tCode:          {}".format(self.code)
            if self.function:
                result += "\n\tFunction-Code: {}".format(self.function_code)
            if self.line_number >= 0:
                result += "\n\tLine: {}".format(self.line_number)
        return result

    @property
    def has_function(self):
        return self.function is not None

    def update(self, state: dict):
        if self.function:
            f_result = self.evaluate_function(state)
            result = f_result >= 0
            state[self.name + ".function"] = f_result
        else:
            result = self.evaluate(state)
        state[self.name] = result
        return result


class DataNode(GraphNode):
    """
    Data nodes bear virtually no importance, but help keep large data sets out of the code. A data node never
    depends on anything else, but only provides a constant value.
    """

    def __init__(self, *, name:str=None, data, line_number:int=-1):
        if name is None:
            name = self.__class__.__gen_symbol__('data_')
        self.name = name
        self.data = data
        self.ancestors = set()
        self.code = _LAMBDA_PATTERN_.format(repr(self.data))
        self.evaluate = eval(self.code)
        self.line_number = line_number

    def __repr__(self):
        return "{} = {}".format(self.name, repr(self.data))


class Parameter(GraphNode):
    """
    Currently, parameter are not fully supported, yet.

    Parameter nodes are very similar to data nodes in that they just provide a simple constant value, and do not
    depend on any other nodes. The difference is, however, that parameter nodes should allow to change their values
    upon intervention from the outside.
    """

    def __init__(self, *, name:str=None, value, line_number:int=-1):
        if name is None:
            name = self.__class__.__gen_symbol__('param_')
        self.name = name
        self.ancestors = set()
        self.value = value
        self.code = _LAMBDA_PATTERN_.format(value)
        self.evaluate = eval(self.code)
        self.line_number = line_number

    def __repr__(self):
        return "{}: {}".format(self.name, self.value)


class Vertex(GraphNode):
    """
    Vertices play the crucial and central role in the graphical model. Each vertex represents either the sampling from
    a distribution, or the observation of such a sampled value.

    You can get the entire graphical model by taking the set of vertices and their `ancestors`-fields, containing all
    vertices, upon which this vertex depends. However, there is a plethora of additional fields, providing information
    about the node and its relationship and status.

    `name`:
      The generated name of the vertex. See also: `original_name`.
    `original_name`:
      In contrast to the `name`-field, this field either contains the name attributed to this value in the original
      code, or `None`.
    `ancestors`:
      The set of all parent vertices. This contains only the ancestors, which are in direct line, and not the parents
      of parents. Use the `get_all_ancestors()`-method to retrieve a full list of all ancestors (including parents of
      parents of parents of ...).
    `dist_ancestors`:
      The set of ancestors used for the distribution/sampling, without those used inside the conditions.
    `cond_ancestors`:
      The set of ancestors, which are linked through conditionals.
    `data`:
      A set of all data nodes, which provide data used in this vertex.
    `distribution`:
      The distribution is an AST/IR-structure, which is usually not used directly, but rather for internal purposes,
      such as extracting the name and type of the distribution.
    `distribution_name`:
      The name of the distribution, such as `Normal` or `Gamma`.
    `distribution_type`:
      Either `continuous` or `discrete`. You will usually query this field using one of the properties
      `is_continuous` or `is_discrete`.
    `observation`:
      The observation in an AST/IR-structure, which is usually not used directly, but rather for internal purposes.
    `conditions`:
      The set of all conditions under which this vertex is evaluated. Each item in the set is actually a tuple of
      a `ConditionNode` and a boolean value, to which the condition should evaluate. Note that the conditions are
      not owned by a vertex, but might be shared across several vertices.
    `dependent_conditions`:
      The set of all conditions that depend on this vertex. In other words, all conditions which contain this
      vertex in their `get_all_ancestors`-set.
    `sample_size`:
      The dimension of the samples drawn from this distribution.
    `support_size`:
      Used for the 'categorical' distribution; basically the length of the vector/list in the first argument.
    `code`:
      The original code for the `evaluate`-method as a string. This is mostly used for debugging.
    """

    def __init__(self, *, name:str=None, ancestors:set=None, data:set=None, distribution=None, observation=None,
                 ancestor_graph=None, conditions:list=None, line_number:int=-1):
        from . import code_types
        if name is None:
            name = self.__class__.__gen_symbol__('y' if observation else 'x')
        if ancestor_graph:
            if ancestors:
                ancestors = ancestors.union(ancestor_graph.vertices)
            else:
                ancestors = ancestor_graph.vertices
        if ancestors is None:
            ancestors = set()
        if data is None:
            data = set()
        if conditions is None:
            conditions = []
        self.name = name
        self.original_name = None
        self.dist_ancestors = ancestors
        if len(conditions) > 0:
            anc = []
            for c,_ in conditions:
                anc += list(c.ancestors)
            self.cond_ancestors = set(anc)
        else:
            self.cond_ancestors = set()
        self.ancestors = ancestors.union(self.cond_ancestors)
        self.data = data
        self.distribution = distribution
        self.observation = observation
        self.conditions = conditions
        self.dependent_conditions = set()
        self.distribution_name = distribution.name
        self.distribution_type = distributions.get(distribution.name, 'unknown')
        self.support_size = distribution.get_support_size()
        code_type = self.distribution.code_type
        self.sample_size = code_type.size if isinstance(code_type, code_types.ListType) else 1
        self.code = _LAMBDA_PATTERN_.format(self.distribution.to_py())
        self.evaluate = eval(self.code)
        if self.observation:
            obs = self.observation.to_py()
            self.evaluate_observation = eval(_LAMBDA_PATTERN_.format(obs))
            self.evaluate_observation_pdf = eval("lambda state, dist: dist.log_pdf({})".format(obs))
        else:
            self.evaluate_observation = None
            self.evaluate_observation_pdf = None
        self.line_number = line_number

    def __repr__(self):
        result = "{}:\n" \
                 "\tAncestors: {}\n" \
                 "\tDistribution: {}\n".format(self.name,
                                               ', '.join(sorted([v.name for v in self.ancestors])),
                                               repr(self.distribution))
        if len(self.conditions) > 0:
            result += "\tConditions: {}\n".format(', '.join(["{} == {}".format(c.name, v) for c, v in self.conditions]))
        if self.observation:
            result += "\tObservation: {}\n".format(repr(self.observation))
        if Options.debug:
            result += "\tDependent Conditions: {}\n".format(', '.join(sorted([c.name for c in self.dependent_conditions])))
            result += "\tDistr-Type: {}\n\tOriginal-Name: {}\n\tCode: {}\n\tSample-Size: {}\n".format(
                self.distribution_type,
                self.original_name if self.original_name else "-",
                self.code,
                self.sample_size
            )
            if self.support_size:
                result += "\tSupport-size: {}\n".format(self.support_size)
            if self.line_number >= 0:
                result += "\tLine: {}\n".format(self.line_number)
        return result

    def _add_dependent_condition(self, cond: ConditionNode):
        self.dependent_conditions.add(cond)
        for a in self.ancestors:
            a._add_dependent_condition(cond)

    @property
    def get_all_ancestors(self):
        result = []
        for a in self.ancestors:
            if a not in result:
                result.append(a)
                result += list(a.get_all_ancestors())
        return set(result)

    @property
    def is_conditional(self):
        return len(self.dependent_conditions) > 0

    @property
    def is_continuous(self):
        return self.distribution_type == 'continuous'

    @property
    def is_discrete(self):
        return self.distribution_type == 'discrete'

    @property
    def is_observed(self):
        return self.observation is not None

    @property
    def is_sampled(self):
        return self.observation is None

    def update(self, state: dict):
        if self.evaluate_observation:
            result = self.evaluate_observation(state)
        else:
            result = self.evaluate(state).sample()
        state[self.name] = result
        return result

    def update_pdf(self, state: dict):
        for cond, truth_value in self.conditions:
            if state[cond.name] != truth_value:
                return 0.0

        distr = self.evaluate(state)
        if self.evaluate_observation_pdf:
            log_pdf = self.evaluate_observation_pdf(state, distr)
        elif self.name in state:
            log_pdf = distr.log_pdf(state[self.name])
        else:
            log_pdf = 0.0

        state['log_pdf'] = state.get('log_pdf', 0.0) + log_pdf
        return log_pdf


####################################################################################################

class Graph(object):
    """
    The graph is mostly a set of vertices, and it is used in order to create the graphical model inside the compiler.
    """

    EMPTY = None

    def __init__(self, vertices:set, data:set=None):
        if data is None:
            data = set()
        arcs = []
        conditions = []
        for v in vertices:
            for a in v.ancestors:
                arcs.append((a, v))
            for c, _ in v.conditions:
                conditions.append(c)
        self.vertices = vertices
        self.data = data
        self.arcs = set(arcs)
        self.conditions = set(conditions)

    def __repr__(self):
        if len(self.vertices) == 0 and len(self.data) == 0 and len(self.conditions) == 0:
            return "<Graph.EMPTY>"
        V = '  '.join(sorted([repr(v) for v in self.vertices]))
        A = ', '.join(['({}, {})'.format(u.name, v.name) for (u, v) in self.arcs]) if len(self.arcs) > 0 else "-"
        C = '\n  '.join(sorted([repr(v) for v in self.conditions])) if len(self.conditions) > 0 else "-"
        D = '\n  '.join([repr(u) for u in self.data]) if len(self.data) > 0 else "-"
        return "Vertices V:\n  {V}\nArcs A:\n  {A}\n\nConditions C:\n  {C}\n\nData D:\n  {D}\n".format(V=V, A=A, C=C, D=D)

    @property
    def is_empty(self):
        """
        Returns `True` if the graph is empty (contains no vertices).
        """
        return len(self.vertices) == 0

    def merge(self, other):
        """
        Merges this graph with another graph and returns the result. The original graphs are not modified, but
        a new object is instead created and returned.

        :param other: The second graph to merge with the current one.
        :return:      A new graph-object.
        """
        if other:
            return Graph(set.union(self.vertices, other.vertices), set.union(self.data, other.data))
        else:
            return self

    def get_vertex_for_distribution(self, distribution):
        """
        Returns the vertex that has the specified distribution or `None`, if no such vertex exists.

        :param distribution:  The distribution as a `CodeObject`.
        :return:              Either a `Vertex` or `None`.
        """
        for v in self.vertices:
            if v.distribution == distribution:
                return v
        return None

    def get_ordered_list_of_all_nodes(self):
        """
        Returns the list of all nodes (conditionals, vertices, data, ...), sorted by the index in their generated
        name, which is to say: sorted by the order of their creation. Since each node can only depend on nodes
        created before, we get a valid order for computations.

        Used by the method `create_model`.

        :return:     A list of nodes.
        """
        def extract_number(s):
            result = 0
            for c in s:
                if '0' <= c <= '9':
                    result = result * 10 + ord(c) - ord('0')
            return result

        nodes = {}
        for v in self.vertices:
            nodes[extract_number(v.name)] = v
        for d in self.data:
            nodes[extract_number(d.name)] = d
        for c in self.conditions:
            nodes[extract_number(c.name)] = c
        result = []
        for key in sorted(nodes.keys()):
            result.append(nodes[key])
        return result

    def create_model(self, *, result_expr=None):
        """
        Creates a new model from the present graph. Note that the list of nodes is only a shallow copy. Hence, if you
        were to change any data inside a node of this graph, it will also affect all models created.

        :return:  A new instance of `Model` (`foppl_model`).
        """
        from .foppl_model import Model
        compute_nodes = self.get_ordered_list_of_all_nodes()
        if result_expr:
            if hasattr(result_expr, 'to_py'):
                result_expr = result_expr.to_py()
            result_function = eval(_LAMBDA_PATTERN_.format(result_expr))
        else:
            result_function = None
        return Model(vertices=self.vertices, arcs=self.arcs, data=self.data,
                     conditionals=self.conditions, compute_nodes=compute_nodes,
                     result_function=result_function)


Graph.EMPTY = Graph(vertices=set())


def merge(*graphs):
    result = Graph.EMPTY
    for g in graphs:
        result = result.merge(g)
    return result
