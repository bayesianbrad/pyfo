#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 20. Dec 2017, Tobias Kohn
# 16. Jan 2018, Tobias Kohn
#
from .foppl_distributions import continuous_distributions, discrete_distributions

class Graph(object):
    """
    The graph models sampling and observations as nodes/vertices, and the functional relationship between these
    stochastic variables as directed edges/arcs. The shape of the graph is therefore determined by the fields
    `vertices` and `arcs`, respectively. The vertices are names stored as strings, whereas the arcs are stored
    as tuples of names.

    For each vertex `x`, the Python code to compute the vertex' value is stored in the field `conditional_densities`.
    That is, `conditional_densities[x]` contains the Python code necessary to generate the correct value for `x`,
    possibly with dependencies on other vertices (as indicated by the arcs).

    Observed values are stored in the field `observed_values`. If the vertex `x` has been observed to have value `1`,
    this value is stored as `observed_values[x]`.

    The graph constructs three additional sets of unobserved values: `cont_vars` for values samples from a continuous
    distribution, `disc_vars` for values samples from a discrete distribution, and `cond_vars` for vertices, which
    occur as part of conditional execution (`if`-expressions/statements).

    In order to assist the compilation process, the graph records some additional information as follows:
    - `observed_conditions` indicates for observed values the condition under which it is actually observed,
    - `original_names` is a mapping from internal names to the variable names used in the original script,
    - `conditional_functions` maps conditional values (`True`/`False`) to their functions, i.e., for `c = f >= 0`
      we map `c -> f`,
    - `used_functions` is set that records all functions used inside the code, which have not been recognized by
      the compiler. These functions need to be provided by other means to the model/Python code.

    Graphs are thought to be immutable objects. Use a `GraphBuilder` to create and modify new graphs. There are some
    exceptions, though: the compiler might have to add a specific value or mapping to a newly created graph. That is
    why you will find some `add_XXX`-methods. They should, however, be used with caution and only in controlled ways.

    Example:
        ```python
        (let [x (sample (categorical 0 1))
              y (if (>= x 0)
                    (sample(normal(mu=1, sigma=0.25)))
                    (sample(normal(mu=2, sigma=0.5)))]
          (observe (normal (y 1)) 1.5))
        ```
        This code gives raise to the graph:
        ```
        Vertices:
        'x', 'y', 'y_1', 'y_2', 'y_cond', 'z'
        Arcs:
        (x, y_cond), (y_1, y), (y_2, y), (y, z)
        Conditional densities:
        x      -> categorical(0, 1)
        y_1    -> normal(mu = 1, sigma = 0.25)
        y_2    -> normal(mu = 2, sigma = 0.5)
        y_cond -> x >= 0
        y      -> y_1 if y_cond else y_2
        z      -> normal(mu = y, sigma = 1)
        Observed values:
        z      -> 1.5
        Discrete vars:
        x
        Continuous vars:
        y_1, y_2
        Conditional vars:
        y_cond
        ```
    """

    def __init__(self, vertices: set, arcs: set, cond_densities: dict = None, obs_values: dict = None):
        if cond_densities is None:
            cond_densities = {}
        if obs_values is None:
            obs_values = {}
        self.vertices = vertices
        self.arcs = arcs
        self.conditional_densities = cond_densities
        self.observed_values = obs_values
        self.observed_conditions = {}
        self.original_names = {}
        self.conditional_functions = {}
        observed = self.observed_values.keys()
        f = lambda x: (x[5:x.index('(')] if x.startswith('dist') and '(' in x else x)
        self.cont_vars = set(n for n in vertices
                                if n in cond_densities
                                if n not in observed
                                if f(cond_densities[n]) in continuous_distributions)
        self.disc_vars = set(n for n in vertices
                                if n in cond_densities
                                if n not in observed
                                if f(cond_densities[n]) in discrete_distributions)
        self.cond_vars = set(n for n in vertices
                                if n in cond_densities
                                if n.startswith('cond'))
        self.used_functions = set()
        self.EMPTY = None

    def __repr__(self):
        cond = self.conditional_densities
        obs = self.observed_values
        C_ = "\n".join(["  {} -> {}".format(v, cond[v]) for v in cond])
        O_ = "\n".join(["  {} -> {}".format(v, obs[v]) for v in obs])
        V = "Vertices V:\n  " + ', '.join(sorted(self.vertices))
        A = "Arcs A:\n  " + ', '.join(['({}, {})'.format(u, v) for (u, v) in self.arcs])
        C = "Conditional densities C:\n" + C_
        O = "Observed values O:\n" + O_
        return "\n".join([V, A, C, O])

    @property
    def is_empty(self):
        """
        Returns `True` if the graph is empty (contains no vertices).
        """
        return len(self.vertices) == 0 and len(self.arcs) == 0

    def merge(self, other):
        """
        Merges this graph with another graph and returns the result. The original graphs are not modified, but
        a new object is instead created and returned.

        :param other: The second graph to merge with the current one.
        :return:      A new graph-object.
        """
        V = set.union(self.vertices, other.vertices)
        A = set.union(self.arcs, other.arcs)
        C = {**self.conditional_densities, **other.conditional_densities}
        O = {**self.observed_values, **other.observed_values}
        G = Graph(V, A, C, O)
        G.cont_vars = set.union(self.cont_vars, other.cont_vars)
        G.disc_vars = set.union(self.disc_vars, other.disc_vars)
        G.cond_vars = set.union(self.cond_vars, other.cond_vars)
        G.observed_conditions = {**self.observed_conditions, **other.observed_conditions}
        G.original_names = {**self.original_names, **other.original_names}
        G.conditional_functions = {**self.conditional_functions, **other.conditional_functions}
        G.used_functions = set.union(self.used_functions, other.used_functions)
        return G

    def add_condition_for_observation(self, obs: str, cond: str):
        if obs in self.observed_conditions:
            self.observed_conditions[obs] += " and {}".format(cond)
        else:
            self.observed_conditions[obs] = cond

    def add_condition(self, cond):
        if cond:
            for obs in self.observed_values.keys():
                self.add_condition_for_observation(obs, cond)
        return self

    def add_original_name(self, original_name, new_name):
        self.original_names[new_name] = original_name

    def add_conditional_function(self, cond_name, function_name):
        self.conditional_functions[cond_name] = function_name

    def add_used_function(self, name):
        self.used_functions.add(name)

    def get_code_for_variable(self, var_name: str):
        if var_name in self.conditional_densities:
            source = self.conditional_densities[var_name]
        elif var_name in self.observed_values:
            source = self.observed_values[var_name]
        else:
            source = "???"
        return "{source}".format(var_name=var_name, source=source)

    def is_observed_variable(self, var_name: str):
        return var_name in self.observed_values

    @property
    def not_observed_variables(self):
        V = self.vertices
        return V.difference(set(self.observed_values.keys()))

    @property
    def sampled_variables(self):
        V = self.vertices
        V = V.difference(set(self.observed_values.keys()))
        return {v for v in V if self.get_code_for_variable(v).startswith("dist.")}

    @property
    def sorted_edges_by_parent(self):
        result = {u: [] for u in self.vertices}
        for (u, v) in self.arcs:
            if u in result:
                result[u].append(v)
            else:
                result[u] = [v]
        return {key: set(result[key]) for key in result}

    @property
    def sorted_edges_by_child(self):
        result = { u: [] for u in self.vertices }
        for (u, v) in self.arcs:
            if v in result:
                result[v].append(u)
            else:
                result[v] = [u]
        return {key: set(result[key]) for key in result}

    def get_parents_of_node(self, var_name):
        edges = self.sorted_edges_by_child
        if var_name in edges:
            return edges[var_name]
        else:
            return set()

    def get_all_parents_of_node(self, var_name):
        edges = self.sorted_edges_by_child
        if var_name in edges:
            result = list(edges[var_name])
            i = 0
            while i < len(result):
                node = result[i]
                if node in edges:
                    for e in edges[node]:
                        if e not in result:
                            result.append(e)
                i += 1
            return set(result)
        else:
            return set()

    @property
    def sorted_var_list(self):
        """
        The list of all variables, sorted so that each vertex in the sequence only depends on vertices occurring
        earlier in the sequence.
        """
        edges = self.sorted_edges_by_child.copy()
        changed = True
        while changed:
            changed = False
            for u in edges:
                w = edges[u]
                for v in edges[u]:
                    if not edges[v].issubset(w):
                        w = w.union(edges[v])
                        changed = True
                edges[u] = w

        result = []
        f = lambda s: int(''.join([x for x in s if '0' <= x <= '9']))
        while len(edges) > 0:
            batch = []
            keys = list(edges.keys())
            for u in keys:
                if len(edges[u]) == 0:
                    del edges[u]
                    batch.append(u)
            result += sorted(batch, key=f)
            batch = set(batch)
            for u in edges:
                edges[u] = edges[u].difference(batch)
        return result

    @property
    def if_vars(self):
        result = set()
        for cond in self.cond_vars:
            ancestors = self.get_all_parents_of_node(cond)
            ancestors = ancestors.difference(self.disc_vars)
            result = result.union(ancestors)
        return result

    def get_conditional_functions(self):
        result = []
        for name in self.conditional_functions:
            target = self.conditional_functions[name]
            if not target.startswith("lambda "):
                target = repr(target)
            result.append("'{}': {}".format(name, target))
        if len(result) > 0:
            return "{{\n  {}\n}}".format(',\n  '.join(result))
        else:
            return "{}"

    def get_discrete_distributions(self):
        result = []
        for name in self.disc_vars:
            code = self.get_code_for_variable(name)
            if code.startswith("dist."):
                code = code[5:]
                i = 0
                while i < len(code) and ('A' <= code[i] <= 'Z' or 'a' <= code[i] <= 'z'):
                    i += 1
                result.append("'{}': '{}'".format(name, code[:i]))
        if len(result) > 0:
            return "{{\n  {}\n}}".format(',\n  '.join(result))
        else:
            return "{}"


class GraphBuilder(object):

    def __init__(self):
        self.vertices = []
        self.arcs = []
        self.conditional_densities = {}
        self.observed_values = {}
        self.cont_vars = []
        self.disc_vars = []

    def get_graph(self):
        G = Graph(set(self.vertices), set(self.arcs), self.conditional_densities, self.observed_values)
        G.cont_vars = set(self.cont_vars)
        G.disc_vars = set(self.disc_vars)
        return G

    def add_var(self, name):
        self.vertices.append(name)

    def add_continuous_var(self, name):
        self.vertices.append(name)
        self.cont_vars.append(name)

    def add_discrete_var(self, name):
        self.vertices.append(name)
        self.disc_vars.append(name)

    def add_arc(self, arc):
        self.arcs.append(arc)

    def add_cond_densitiy(self, key, value):
        self.conditional_densities[key] = value

    def add_observed_value(self, key, value):
        self.observed_values[key] = value


def merge(*graphs):
    result = Graph.EMPTY
    for g in graphs:
        result = result.merge(g)
    return result

Graph.EMPTY = Graph(set(), set())
