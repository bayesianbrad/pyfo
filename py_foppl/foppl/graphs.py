#
# (c) 2017, Tobias Kohn
#
# 20. Dec 2017
# 28. Dec 2017
#
from .foppl_distributions import continuous_distributions, discrete_distributions

class Graph(object):

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
        f = lambda x: (x[5:x.index('(')] if x.startswith('dist') and '(' in x else x)
        self.cont_vars = set(n for n in vertices
                                if n in cond_densities
                                if f(cond_densities[n]) in continuous_distributions)
        self.disc_vars = set(n for n in vertices
                                if n in cond_densities
                                if f(cond_densities[n]) in discrete_distributions)
        self.cond_vars = set(n for n in vertices
                                if n in cond_densities
                                if n.startswith('cond'))
        self.EMPTY: Graph = None

    def __repr__(self):
        cond = self.conditional_densities
        obs = self.observed_values
        C_ = "\n".join(["  {} -> {}".format(v, cond[v]) for v in cond])
        O_ = "\n".join(["  {} -> {}".format(v, cond[v]) for v in obs])
        V = "Vertices V:\n  " + ', '.join(sorted(self.vertices))
        A = "Arcs A:\n  " + ', '.join(['({}, {})'.format(u, v) for (u, v) in self.arcs])
        C = "Conditional densities C:\n" + C_
        O = "Observed values O:\n" + O_
        return "\n".join([V, A, C, O])

    @property
    def is_empty(self):
        return len(self.vertices) == 0 and len(self.arcs) == 0

    def merge(self, other):
        V = set.union(self.vertices, other.vertices)
        A = set.union(self.arcs, other.arcs)
        C = {**self.conditional_densities, **other.conditional_densities}
        O = {**self.observed_values, **other.observed_values}
        G = Graph(V, A, C, O)
        G.cont_vars = set.union(self.cont_vars, other.cont_vars)
        G.disc_vars = set.union(self.disc_vars, other.disc_vars)
        G.observed_conditions = {**self.observed_conditions, **other.observed_conditions}
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

    @property
    def sorted_var_list(self):
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


def merge(G1: Graph, G2: Graph):
    return G1.merge(G2)

Graph.EMPTY = Graph(set(), set())
