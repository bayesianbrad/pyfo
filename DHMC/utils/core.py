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
from torch.autograd import Variable

def VariableCast(value, grad = False):
    '''casts an input to torch Variable object
    input
    -----
    value - Type: scalar, Variable object, torch.Tensor, numpy ndarray
    grad  - Type: bool . If true then we require the gradient of that object

    output
    ------
    torch.autograd.variable.Variable object
    '''
    if isinstance(value, Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value, requires_grad = grad)
    elif isinstance(value, np.ndarray):
        return Variable(torch.from_numpy(value).type(torch.FloatTensor), requires_grad = grad)
    else:
        return Variable(torch.Tensor([value]).type(torch.FloatTensor), requires_grad = grad)

def tensor_to_list(self,values):
    ''' Converts a tensor to a list
    values = torch.FloatTensor or Variable'''
    params = []
    for value in values:
        if isinstance(value, Variable):
            temp = Variable(value.data, requires_grad=True)
            params.append(temp)
        else:
            temp = VariableCast(value)
            temp = Variable(value.data, requires_grad=True)
            params.append(value)
    return params

def TensorCast(value):
    if isinstance(value, torch.Tensor):
        return value
    else:
        return torch.Tensor([value])