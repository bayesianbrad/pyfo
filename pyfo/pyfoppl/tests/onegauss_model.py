#
# Generated: 2018-01-10 15:38:45.298736
#
import math
import numpy as np
import torch
from torch.autograd import Variable
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	"""
	Vertices V:
	  x20001, y20002
	Arcs A:
	  (x20001, y20002)
	Conditional densities C:
	  x20001 -> dist.Normal(mu=1.0, sigma=2.23606797749979)
	  y20002 -> dist.Normal(mu=x20001, sigma=1.4142135623730951)
	Observed values O:
	  y20002 -> 7.0
	"""
	vertices = {'x20001', 'y20002'}
	arcs = {('x20001', 'y20002')}
	names = {'x20001': 'x'}
	cond_functions = {
	  
	}

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
	def gen_cont_vars(self):
		return ['x20001']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def gen_if_vars(self):
		return []

	@classmethod
	def gen_pdf(self, state):
		dist_x20001 = dist.Normal(mu=1.0, sigma=2.23606797749979)
		x20001 = state['x20001']
		p10000 = dist_x20001.log_pdf(x20001)
		dist_y20002 = dist.Normal(mu=x20001, sigma=1.4142135623730951)
		p10001 = dist_y20002.log_pdf(7.0)
		logp = p10000 + p10001
		return logp

	@classmethod
	def gen_prior_samples(self):
		dist_x20001 = dist.Normal(mu=1.0, sigma=2.23606797749979)
		x20001 = dist_x20001.sample()
		dist_y20002 = dist.Normal(mu=x20001, sigma=1.4142135623730951)
		y20002 = 7.0
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state  # dictionary

	@classmethod
	def gen_vars(self):
		return ['x20001']
