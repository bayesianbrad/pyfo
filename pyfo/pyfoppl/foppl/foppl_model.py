#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 20. Dec 2017, Tobias Kohn
# 23. Jan 2018, Tobias Kohn
#
from . import runtime
from .basic_imports import *

# We try to import `networkx` and `matplotlib`. If present, these packages can be used to get a visual
# representation of the graph. But neither of these packages is actually needed.
try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

####################################################################################################

class Model(object):
    """
    The model serves two purposes: on the one hand, it acts as an interface to the graph, i.e. its vertices. On the
    other hand, it provides the methods to draw samples and compute the log-pdf.

    You will rarely, if ever, instantiate a `Model` instance on your own. Instead, the model is created by the graph,
    after having constructed the entire graph. While the `Graph` class is more leaned towards building the graphical
    model, the `Model` class rather provides function to retrieve different pieces of information from the model.
    """

    def __init__(self, *, vertices: set, arcs: set, data: set, conditionals: set, compute_nodes: list,
                 result_function = None):
        self.vertices = vertices
        self.arcs = arcs
        self.data = data
        self.conditionals = conditionals
        self.compute_nodes = compute_nodes
        self.result_function = result_function
        self.nodes = { v.name: v for v in self.compute_nodes }

    def __repr__(self):
        V = '  '.join(sorted([repr(v) for v in self.vertices]))
        A = ', '.join(['({}, {})'.format(u.name, v.name) for (u, v) in self.arcs]) if len(self.arcs) > 0 else "-"
        C = '\n  '.join(sorted([repr(v) for v in self.conditionals])) if len(self.conditionals) > 0 else "-"
        D = '\n  '.join([repr(u) for u in self.data]) if len(self.data) > 0 else "-"
        graph = "Vertices V:\n  {V}\nArcs A:\n  {A}\n\nConditions C:\n  {C}\n\nData D:\n  {D}\n".format(V=V, A=A, C=C, D=D)
        model = "\nContinuous:  {}\nDiscrete:    {}\nConditional: {}\nConditions: {}\n".format(
            ', '.join(sorted(self.gen_cont_vars())),
            ', '.join(sorted(self.gen_disc_vars())),
            ', '.join(sorted(self.gen_if_vars())),
            ', '.join(sorted(self.gen_cond_vars()))
        )
        return graph + model

    def create_network_graph(self):
        """
        Create a `networkx` graph. Used by the method `display_graph()`.

        :return: Either a `networkx.DiGraph` instance or `None`.
        """
        if nx:
            G = nx.DiGraph()
            for v in self.vertices:
                G.add_node(v.display_name)
                for a in v.ancestors:
                    G.add_edge(a.display_name, v.display_name)
            return G
        else:
            return None

    def display_graph(self):
        """
        Transform the graph to a `networkx.DiGraph`-structure and display it using `matplotlib` -- if the necessary
        libraries are installed.

        :return: `True` if the graph was drawn, `False` otherwise.
        """
        G = self.create_network_graph()
        if nx and plt and G:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                pos = graphviz_layout(G, prog='dot')
            except ModuleNotFoundError:
                from networkx.drawing.layout import shell_layout
                pos = shell_layout(G)
            except ImportError:
                from networkx.drawing.layout import shell_layout
                pos = shell_layout(G)
            plt.subplot(111)
            plt.axis('off')
            nx.draw_networkx_nodes(G, pos,
                                   node_color='r',
                                   node_size=1250,
                                   nodelist=[v.display_name for v in self.vertices if v.is_sampled])
            nx.draw_networkx_nodes(G, pos,
                                   node_color='b',
                                   node_size=1250,
                                   nodelist=[v.display_name for v in self.vertices if v.is_observed])
            for v in self.vertices:
                nx.draw_networkx_edges(G, pos, arrows=True,
                                       edgelist=[(a.display_name, v.display_name) for a in v.dist_ancestors])
                nx.draw_networkx_edges(G, pos, arrows=True,
                                       style='dashed',
                                       edge_color='g',
                                       edgelist=[(a.display_name, v.display_name) for a in v.cond_ancestors])
            nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold')
            plt.show()
            return True
        else:
            return False

    def index_of_node(self, node):
        return self.compute_nodes.index(node)

    def get_vertices(self):
        return self.vertices

    def get_vertices_names(self):
        return [v.name for v in self.vertices]

    def get_arcs(self):
        return self.arcs

    def get_arcs_names(self):
        return [(u.name, v.name) for (u, v) in self.arcs]

    def get_map_of_nodes(self):
        return self.nodes

    def get_continuous_distributions(self):
        return set([v.distribution_name for v in self.vertices if v.is_continuous])

    def get_discrete_distributions(self):
        return set([v.distribution_name for v in self.vertices if v.is_discrete])

    def gen_cond_vars(self):
        return [c.name for c in self.conditionals]

    def gen_if_vars(self):
        return [v.name for v in self.vertices if v.is_conditional and v.is_sampled and v.is_continuous]

    def gen_cont_vars(self):
        return [v.name for v in self.vertices if v.is_continuous and not v.is_conditional and v.is_sampled]

    def gen_disc_vars(self):
        return [v.name for v in self.vertices if v.is_discrete and v.is_sampled]

    def gen_vars(self):
        return [v.name for v in self.vertices if v.is_sampled]

    def gen_prior_samples(self):
        state = {}
        for node in self.compute_nodes:
            node.update(state)
        return state

    def gen_prior_samples_code(self):
        result = []
        for node in self.compute_nodes:
            result.append("# {}".format(node.name))
            result.append(node.full_code)
        return '\n'.join(result)

    def gen_pdf(self, state):
        for node in self.compute_nodes:
            node.update_pdf(state)
        if 'log_pdf' in state:
            return state['log_pdf']
        else:
            return 0.0

    def gen_pdf_code(self):
        result = []
        for node in self.compute_nodes:
            result.append("# {}".format(node.name))
            if hasattr(node, 'full_code_pdf'):
                result.append(node.full_code_pdf)
            else:
                result.append(node.full_code)
        return '\n'.join(result)

    def get_result(self, state):
        if self.result_function is not None:
            return self.result_function(state)
        else:
            return None