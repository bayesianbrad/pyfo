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
import torch.tensor as tt
import torch.distributions as dists
from torch.distributions import constraints, biject_to
def VariableCast(value, grad = False, dist=None):
    '''casts an input to torch.tensor object

    :param value Type: scalar, torch.Tensor object, torch.Tensor, numpy ndarray
    :param  grad Type: bool . If true then we require the gradient of that object

    output
    ------
    torch.tensor object
    '''
    if value is None:
        return None
    elif isinstance(value, tt):
        return value
    elif torch.is_tensor(value):
        return tt(value, requires_grad = grad)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
        return tt(tensor, requires_grad = grad)
    elif isinstance(value,list):
        return tt(torch.FloatTensor(value), requires_grad=grad)
    else:
        return tt(torch.FloatTensor([value]), requires_grad = grad)

def tensor_to_list(self,values):
    ''' Converts a tensor to a list
    values = torch.FloatTensor or torch.tensor'''
    params = []
    for value in values:
        if isinstance(value, tt):
            temp = tt(value.data, requires_grad=True)
            params.append(temp)
        else:
            temp = VariableCast(value)
            temp = tt(value.data, requires_grad=True)
            params.append(value)
    return params

def TensorCast(value):
    if isinstance(value, tt):
        return value
    else:
        return tt([value])

def list_to_tensor(self, params):
    '''
    Unpacks the parameters list tensors and converts it to list

    returns tensor of  num_rows = len(values) and num_cols  = 1
    problem:
        if there are col dimensions greater than 1, then this will not work
    '''
    print('Warning ---- UNSTABLE FUNCTION ----')
    assert(isinstance(params, list))
    temp = tt(torch.Tensor(len(params)).unsqueeze(-1))
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
    if isinstance(t, tt):
        return t.data
    return t

def my_import(name):
    '''
    Helper function for extracting the whole module and not just the package.
    See answer by clint miller for details:
    https://stackoverflow.com/questions/951124/dynamic-loading-of-python-modules

    :param name
    :type string
    :return module
    '''
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def transform_latent_support(latent_vars, dist_to_latent):
    """
    Returns a new state with the required transformations for the log pdf. It checks the support of each continuous
    distribution and if that support does not encompass the whole real line, the required bijector is added to a
    transform list.

    :param latent_vars: dictionary of {latent_var: distribution_name}
    :param dist_to_latent: dictionary that maps latent_variable names to distribution name

    :return: transform: dictionary of {latent_var: bijector_for_latent}
    """
    transforms = {}
    for latent in latent_vars:
        temp_support = getattr(dists, dist_to_latent[latent]).support
        if temp_support is not constraints.real:
            transforms[latent_vars] = biject_to(temp_support).inv
    return transforms

def _to_leaf(self, state):
    """
    Ensures that all latent parameters are reset to leaf nodes, before
    calling
    :param state:
    :return:
    """
    for key in state:
        tmp = VariableCast(state[key])
        state[key] = VariableCast(tmp.data, grad=True)
        # state[key] = VariableCast(state[key].data, grad=True)
    return state