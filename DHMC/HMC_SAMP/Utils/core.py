import torch
import numpy as np
from torch.autograd import Variable

class ContinuousRandomVariable():
    '''A very basic representation of what should be contained in a
       continuous random variable class'''
    def pdf(self):
        raise NotImplementedError("pdf is not implemented")
    def logpdf(self, x):
        raise NotImplementedError("logpdf is not implemented")

    def sample(self):
        raise NotImplementedError("sample is not implemented")
    def iscontinous(self):
        return True

class DiscreteRandomVariable():
    '''A very basic representation of what should be contained in a
       discrete random variable class'''
    def pmf(self):
        raise NotImplementedError("pdf is not implemented")
    def logpmf(self, x):
        raise NotImplementedError("log_pdf is not implemented")

    def sample(self):
        raise NotImplementedError("sample is not implemented")
    def isdiscrete(self):
        return True

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
        return Variable(torch.from_numpy(value), requires_grad = grad)
    else:
        return Variable(torch.Tensor([value]), requires_grad = grad)

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