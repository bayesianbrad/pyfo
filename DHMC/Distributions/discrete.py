#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:43
Date created:  09/11/2017

License: MIT
'''
import torch
from torch.autograd import Variable
from DHMC.utils.core import VariableCast

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

class Categorical(DiscreteRandomVariable):
    """categorical Normal random variable"""

    def __init__(self, p):
        """Initialize this distribution with p =[p0, p1, ..., pn].

        input:
            mean:
            std:  standard deviation

        output:
            integer in [1, ..., n]
        """
        self.p = VariableCast(p)

    def sample(self):
        onedraw = np.random.multinomial(1, self.p.data.numpy())
        index = np.argwhere(onedraw == 1)[0,0]
        var = Variable(torch.Tensor([int(index)]) +1 ,requires_grad = True)
        return var

    def logpdf(self, value):
        int_value =  int(value.data.numpy())
        index = int_value -1
        if 1 <= int_value <= self.p.data.shape[0]:
            return torch.log(self.p[index])
        else:
            return torch.Tensor([-np.inf])

class Categorical_Trans():
    """categorical Tranformed random variable"""

    def __init__(self, p, method=None):
        """Initialize this distribution with p =[p0, p1, ..., pn].

        input:
            mean:
            std:  standard deviation

        output:
            integer in [1, ..., n]
        """

        self.p = VariableCast(p)
        if method is None:
            self.method = "standard"
        else:
            self.method = method

    def logpdf(self, value):  # logp: 1*1
        if self.method == "standard":

            value =  VariableCast(value)
            if len(value.data.shape) == 1:
                value = value.unsqueeze(1)

            int_value = int(torch.floor(value.data)[0,0])
            index = int_value - 1

            #returning logp is [-0.93838], wrapped by tensor
            if 1 <= value.data[0,0] <= self.p.data.shape[0] + 1:
                logp = torch.log(self.p[index])   # grad does not survive through this embedding
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            else:
                return Variable(torch.Tensor([[-np.inf]]))
        else:
            raise ValueError("implement categorical transformed method")
            return 0


class Binomial_Trans(DiscreteRandomVariable):
    """ binomial distribution, into contnuous space
       discrete distribution does not support grad for now
    """

    def __init__(self, n, p, method=None):
        """Initialize this distribution with
        :parameter
         n; N_0, non-negative integar
         p: [0, 1]

        output:
            integer in [0, 1, ..., n]
        """

        self.p = VariableCast(p)
        self.n = VariableCast(n)
        if method is not None:
            self.method = method
        else:
            self.method = "standard"

    def logpdf(self, k):  # logp: 1*1
        k = VariableCast(k)
        n = self.n
        p = self.p
        if len(k.data.shape) == 1:
            k = k.unsqueeze(1)
        if len(self.n.data.shape) == 1:
            n = n.unsqueeze(1)
        if len(self.p.data.shape) == 1:
            p = p.unsqueeze(1)

        if self.method == "standard":
            int_k = int(torch.floor(k.data)[0,0])
            int_n = int(torch.floor(n.data)[0,0])
            np_p = p.data[0,0]

            #returning logp is [-0.93838], wrapped by tensor
            if 0 <= int_k <= int_n:
                logpmf = ss.binom.logpmf(int_k, int_n, np_p)
                logp = Variable(torch.Tensor([logpmf]))
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            else:
                return Variable(torch.Tensor([[-np.inf]]))
        else:
            raise ValueError("implement categorical transformed method")
            return 0


class Bernoulli_Trans(DiscreteRandomVariable):
    """ bernoulli distribution, into contnuous space
       discrete distribution does not support grad for now
    """

    def __init__(self, p, method=None):
        """Initialize this distribution with
        :parameter
         p: [0, 1]
        """

        self.p = VariableCast(p)
        if len(self.p.data.shape) == 1:
            self.p = self.p.unsqueeze(1)
        if method is not None:
            self.method = method
        else:
            self.method = "standard"
    def sample(self):
        """
        :return: x in [0,1], [1, 2]
        """
        x = torch.bernoulli(self.p) + Variable(torch.rand(1))
        if len(x.data.shape) == 1:
            x = x.unsqueeze(1)
        return x

    def logpdf(self, x):  # logp: 1*1
        x = VariableCast(x)
        p = self.p
        if len(x.data.shape) == 1:
            x = x.unsqueeze(1)

        if self.method == "standard":

            #returning logp is 1 by 1
            if 0 <= x.data[0,0] < 1:
                logp = torch.log(1-p)
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            elif 1 <= x.data[0, 0] < 2:
                logp = torch.log(p)
                if len(logp.data.shape) == 1:
                    logp = logp.unsqueeze(1)
                return logp
            else:
                return Variable(torch.Tensor([[-np.inf]]))
        else:
            raise ValueError("implement categorical transformed method")
            return 0

