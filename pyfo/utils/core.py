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
        return value # Should I check if FloatTensor?
    elif torch.is_tensor(value):
        return Variable(value, requires_grad = grad)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
        print(type(tensor))
        return Variable(tensor, requires_grad = grad)
    else:
        return Variable(torch.FloatTensor([value]), requires_grad = grad)

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

def list_to_tensor(self, params):
    '''
    Unpacks the parameters list tensors and converts it to list

    returns tensor of  num_rows = len(values) and num_cols  = 1
    problem:
        if there are col dimensions greater than 1, then this will not work
    '''
    print('Warning ---- UNSTABLE FUNCTION ----')
    assert(isinstance(params, list))
    temp = Variable(torch.Tensor(len(params)).unsqueeze(-1))
    for i in range(len(params)):
        temp[i,:] = params[i]
    return temp