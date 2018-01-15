import torch 
import numpy as np  
from torch.autograd import Variable  
# import pyfo.distributions as dist
import torch.distributions as dist
from pyfo.utils.core import VariableCast as vc
from pyfo.utils.interface import interface
import math

class model(object):
	"""
	Vertices V:
	  x20001, y20002
	Arcs A:
	  (x20001, y20002)
	Conditional densities C:
	  x20001 -> dist.Normal(mean=1.0, std=5.0)
	  y20002 -> dist.Normal(mean=(x20001 + 1), std=2.0)
	Observed values O:
	  y20002 -> dist.Normal(mean=(x20001 + 1), std=2.0)
	"""
	vertices = {'y20002', 'x20001'}
	arcs = {('x20001', 'y20002')}
	names = {'x20001': 'x1'}

	@classmethod
	def get_vertices(self):
		return list(self.vertices)

	@classmethod
	def get_arcs(self):
		return list(self.arcs)

	@classmethod
	def gen_cond_vars(self):
		return []
	@classmethod
	def gen_if_vars(self):
		return []

	@classmethod
	def gen_cont_vars(self):
		return ['x20001']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def gen_pdf(self, state):
		dist_x20001 = dist.Normal(mean=vc(1.0), std=vc(math.sqrt(5.0)))
		x20001 = state['x20001']
		p10000 = dist_x20001.log_prob(x20001)
		dist_y20002 = dist.Normal(mean=vc((x20001 + 1)), std=vc(math.sqrt(2.0)))
		p10001 = dist_y20002.log_prob(vc(7.0))
		logp = p10000 + p10001
		return logp

	@classmethod
	def gen_prior_samples(self):
		dist_x20001 = dist.Normal(mean=vc(1.0), std=vc(math.sqrt(5.0)))
		x20001 = dist_x20001.sample()
		dist_y20002 = dist.Normal(mean=vc((x20001 + 1)), std=vc(math.sqrt(2.0)))
		y20002 = vc(7.0)
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state  # dictionary

	@classmethod
	def gen_vars(self):
		return ['x20001']

	@classmethod
	def get_parents_map(self):
		"""
        Returns a dictionary of all variable names and the corresponding
        parents for each variable.

        Consider, for instance, a graph containing the following edges/arcs:
          x1 -> x3, x2 -> x3, x1 -> x4, x3 -> x4
        Then the returned value of this function is a dictionary as follow:
          x1: ()
          x2: ()
          x3: (x1, x2)
          x4: (x1, x3)
        In other words: you get a dictionary that maps each variable name to
        its correspondings parents.

        :return child_parent_relationships type: Dict[str, Set[str]]
        """
		result = {u: [] for u in self.vertices}
		for (u, v) in self.arcs:
			if v in result:
				result[v].append(u)
			else:
				result[v] = [u]
		return {key: set(result[key]) for key in result}

	@classmethod
	def get_parents_of_node(self, var_name: str):
		"""
        Returns a set of all variable names, which are parents of the given
        node/variable/vertex. This function basically extracts a single entry
        from the dictionary given by `get_parents_map`.

        :return set_of_parents :type Set[str]
        """
		edges = self.get_parents_map()
		if var_name in edges:
			return edges[var_name]
		else:
			return set()