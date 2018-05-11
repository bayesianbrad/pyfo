#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  10:02
Date created:  10/11/2017

License: MIT
'''
import torch
import numpy as np
import torch.distributions as dists
from torch.distributions import constraints, biject_to

try:
    import networkx as _nx
except ModuleNotFoundError:
     _nx = None
try:
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches
except ModuleNotFoundError:
    _plt = None

class DualAveraging(object):
    """
    Dual Averaging is a scheme to solve convex optimization problems. It belongs
    to a class of subgradient methods which uses subgradients to update parameters
    (in primal space) of a model. Under some conditions, the averages of generated
    parameters during the scheme are guaranteed to converge to an optimal value.
    However, a counter-intuitive aspect of traditional subgradient methods is
    "new subgradients enter the model with decreasing weights" (see :math:`[1]`).
    Dual Averaging scheme solves that phenomenon by updating parameters using
    weights equally for subgradients (which lie in a dual space), hence we have
    the name "dual averaging".
    This class implements a dual averaging scheme which is adapted for Markov chain
    Monte Carlo (MCMC) algorithms. To be more precise, we will replace subgradients
    by some statistics calculated during an MCMC trajectory. In addition,
    introducing some free parameters such as ``t0`` and ``kappa``is helpful and
    still guarantees the convergence of the scheme.
    References
    [1] `Primal-dual subgradient methods for convex problems`,
    Yurii Nesterov
    [2] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, Andrew Gelman
    :param float prox_center: A "prox-center" parameter introduced in :math:`[1]`
        which pulls the primal sequence towards it.
    :param float t0: A free parameter introduced in :math:`[2]`
        that stabilizes the initial steps of the scheme.
    :param float kappa: A free parameter introduced in :math:`[2]`
        that controls the weights of steps of the scheme.
        For a small ``kappa``, the scheme will quickly forget states
        from early steps. This should be a number in :math:`(0.5, 1]`.
    :param float gamma: A free parameter which controls the speed
        of the convergence of the scheme.
    """

    def __init__(self, prox_center=0, t0=10, kappa=0.75, gamma=0.05):
        self.prox_center = prox_center
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma

        self._x_avg = 0  # average of primal sequence
        self._g_avg = 0  # average of dual sequence
        self._t = 0

    def step(self, g):
        """
        Updates states of the scheme given a new statistic/subgradient ``g``.
        :param float g: A statistic calculated during an MCMC trajectory or subgradient.
        """
        self._t += 1
        # g_avg = (g_1 + ... + g_t) / t
        self._g_avg = (1 - 1/(self._t + self.t0)) * self._g_avg + g / (self._t + self.t0)
        # According to formula (3.4) of [1], we have
        #     x_t = argmin{ g_avg . x + loc_t . |x - x0|^2 },
        # where loc_t := beta_t / t, beta_t := (gamma/2) * sqrt(t)
        self._x_t = self.prox_center - (self._t ** 0.5) / self.gamma * self._g_avg
        # weight for the new x_t
        weight_t = self._t ** (-self.kappa)
        self._x_avg = (1 - weight_t) * self._x_avg + weight_t * self._x_t

    def get_state(self):
        r"""
        Returns the latest :math:`x_t` and average of
        :math:`\left\{x_i\right\}_{i=1}^t` in primal space.
        """
        return self._x_t, self._x_avg


def create_network_graph(vertices):
        """
        Create a `networkx` graph. Used by the method `display_graph()`.
        :return: Either a `networkx.DiGraph` instance or `None`.
        """
        if _nx:
            G = _nx.DiGraph()
            for v in vertices:
                G.add_node(v.display_name)
                for a in v.ancestors:
                    G.add_edge(a.display_name, v.display_name)
            return G
        else:
            return None

def display_graph(vertices):
    """
    Transform the graph to a `networkx.DiGraph`-structure and display it using `matplotlib` -- if the necessary
    libraries are installed.
    :return: `True` if the graph was drawn, `False` otherwise.
    """
    G =create_network_graph(vertices)
    _is_conditioned = None
    if _nx and _plt and G:
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(G, prog='dot')
        except ModuleNotFoundError:
            from networkx.drawing.layout import shell_layout
            pos = shell_layout(G)
        except ImportError:
            from networkx.drawing.layout import shell_layout
            pos = shell_layout(G)
        _plt.subplot(111)
        _plt.axis('off')
        _nx.draw_networkx_nodes(G, pos,
                                node_color='r',
                                node_size=1250,
                                nodelist=[v.display_name for v in vertices if v.is_sampled])
        _nx.draw_networkx_nodes(G, pos,
                                node_color='b',
                                node_size=1250,
                                nodelist=[v.display_name for v in vertices if v.is_observed])

        for v in vertices:
            _nx.draw_networkx_edges(G, pos, arrows=True,
                                    edgelist=[(a.display_name, v.display_name) for a in v.ancestors])
            if v.condition_ancestors is not None and len(v.condition_ancestors) > 0:
                _is_conditioned = 1
                _nx.draw_networkx_edges(G, pos, arrows=True,
                                        style='dashed',
                                        edge_color='g',
                                        edgelist=[(a.display_name, v.display_name) for a in v.condition_ancestors])
        _nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold')

        # for node, _ in G.nodes():
        red_patch = mpatches.Circle((0,0), radius=2, color='r', label='Sampled Variables')
        blue_patch = mpatches.Circle((0,0), radius=2, color='b', label='Observed Variables')
        green_patch = mpatches.Circle((0,0), radius=2, color='g', label='Conditioned Variables') if _is_conditioned else 0
        if _is_conditioned:
            _plt.legend(handles=[red_patch, blue_patch, green_patch])
        else:
            _plt.legend(handles=[red_patch, blue_patch])
        _plt.show()


        return True
    else:
        return False

def VariableCast(value, grad = False):
    '''casts an input to torch.tensor object

    :param value Type: scalar, torch.Tensor object, torch.Tensor, numpy ndarray
    :param  grad Type: bool . If true then we require the gradient of that object

    output
    ------
    torch.tensor object
    '''
    dtype = torch.float
    if value is None:
        return None
    elif isinstance(value,torch.Tensor):
        return torch.tensor(value,dtype=dtype,requires_grad=grad)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
        return torch.tensor(tensor, dtype=dtype, requires_grad = grad)
    elif isinstance(value,list):
        return torch.tensor(value,dtype=dtype, requires_grad=grad).unsqueeze(-1)
    else:
        return torch.tensor([value],dtype=dtype, requires_grad = grad).unsqueeze(-1)

def tensor_to_list(self,values):
    ''' Converts a tensor to a list
    values = torch.FloatTensor or torch.tensor'''
    params = []
    for value in values:
        if isinstance(value, torch.tensor):
            temp = torch.tensor(value.data, requires_grad=True)
            params.append(temp)
        else:
            temp = VariableCast(value)
            temp = torch.tensor(value.data, requires_grad=True)
            params.append(value)
    return params

def TensorCast(value):
    if isinstance(value, torch.tensor):
        return value
    else:
        return torch.tensor([value])

def list_to_tensor(self, params):
    '''
    Unpacks the parameters list tensors and converts it to list

    returns tensor of  num_rows = len(values) and num_cols  = 1
    problem:
        if there are col dimensions greater than 1, then this will not work
    '''
    print('Warning ---- UNSTABLE FUNCTION ----')
    assert(isinstance(params, list))
    temp = torch.tensor(torch.Tensor(len(params)).unsqueeze(-1))
    for i in range(len(params)):
        temp[i,:] = params[i]
    return temp

def logical_trans(var):
    """
    Returns logical 0 or 1 for given variable.
    :param var: Is a  1-d torch.Tensor, float or np.array
    :return: Bool
    """
    print("Warning: logoical_trans() has not been tested on tensors of dimension greater than 1")
    value = VariableCast(var)
    if value.data[0]:
        return True
    else:
        return False

def get_tensor_data(t):
    """
    Returns data of torch.Tensor.autograd.Variable
    :param t: torch.tensor
    :return: torch.Tensor
    """
    if isinstance(t, torch.tensor):
        return t.data
    return t

def my_import(name):
    '''
    Helper function for extracting the whole module and not just the package.
    See answer by clint miller for details:
    htorch.tensorps://stackoverflow.com/questions/951124/dynamic-loading-of-python-modules

    :param name
    :type string
    :return module
    '''
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = torch.tensorr(mod, comp)
    return mod

def transform_latent_support(latent_vars, dist_to_latent):
    """
    Returns a new state with the required transformations for the log pdf. It checks the support of each continuous
    distribution and if that support does not encompass the whole real line, the required bijector is added to a
    transform list.
    TODO: Ensure that only continuous latent variables are beingpassed through this function for now.
    :param latent_vars: dictionary of {latent_var: distribution_name}
    :param dist_to_latent: dictionary that maps latent_variable names to distribution name

    :return: transform: dictionary of {latent_var: bijector_for_latent}
    """
    transforms = {}
    for latent in latent_vars:
        # print('Debug statement: latent vars: {0} and type: {1}'.format(dist_to_latent[latent], type(dist_to_latent[latent])))
        temp_support = getattr(dists,dist_to_latent[latent]).support
        # print('Debug statement temp_support {0}'.format(temp_support))
        if temp_support is not constraints.real:
            transforms[latent] = biject_to(temp_support).inv
        else:
            transforms[latent] = constraints.real
    return transforms

def convert_dict_vars_to_numpy(self, state, latent_vars ):
    """

    :param state: Information on the whole state. Likely to be torch objects
    :param latent_vars:  type: str descript: A list of the latent variables in the state.
    :return: the torch latent variables converted to numpy arrays

    Converts variables in stat to numpy arrays for plotting purposes
    """
    for latent in self.all_vars:
        state[latent] =  state[latent].numpy()
        # state[i] = state[i].data.numpy()
    return state

def _grad_logp(input, parameters):
    """
    Returns the gradient of the log pdf, with respect for
    each parameter. Note the double underscore, this is to ensure that if
    this method is overwritten, then no problems occur when overidded.
    :param state:
    :return: torch.autograd.Variable
    """
    # print(50 *'=')
    # print('Debug statement in _grad_logp \n '+50*'='+'\nChecking gradient flag. \n Printing input : {0} \n Printing parameters : {1} \n Checking if gradient turned on: {2} '.format(input, parameters, parameters.requires_grad))
    gradient_of_param = torch.autograd.grad(outputs=input.sum(), inputs=parameters, retain_graph=True)[0]
    # print('Debug statement in _grad_logp. Printing gradient : {}'.format(gradient_of_param))
    # print(50 * '=')
    return gradient_of_param


def _to_leaf(state, latent_vars):
    """
    Ensures that all latent parameters are reset to leaf nodes, before
    calling
    :param state:
    :return:
    """
    for key in state:
        state[key] = torch.tensor(state[key], requires_grad=True)
    return state

def _generate_log_pdf(model,  state):
    """
    The compiled pytorch function, log_pdf, should automatically
    return the pdf.
    :param keys type: list of discrete embedded discrete parameters
    :return: log_pdf

    Maybe overidden in other methods, that require dynamic pdfs.
    For example
    if you have a model called my mymodel, you could write the following:
    Model = compile_model(mymodel) # returns class
    class MyNewModel(Model):

        def gen_log_pdf(self, state):
            for vertex in self.vertices:
                pass
    return "Whatever you fancy"

    # This overrides the base method.
    # Then all you have to do is pass
    # My model into kernel of choice, i.e
    kernel = MCMC(MyNewModel,kernel=HMC)
    kernel.run_inference()

    If you require gradients, ensure that you have used the the core._to_leaf() function on the 'state'
    """

    # if set_leafs:
    #     # only sets the gradients of the latent variables.
    #     _state = _to_leaf(state=state, latent_vars=latents)
    # else:
    #     _state = state
    # print(50*'=')
    # for key in state:
    #     print('Debug statement in _generate_log_p \n',50*'='+ '\n Printing set_leafs : {0} \n Printing latents : {1} \n gradient: {2} \n key: {3} '.format(set_leafs, latents,  state[key].requires_grad, key))
    return model.gen_log_pdf(state)