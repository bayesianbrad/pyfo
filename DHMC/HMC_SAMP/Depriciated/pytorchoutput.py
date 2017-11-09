#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  08:59
Date created:  05/09/2017

License: MIT
'''
import torch
from Utils.core import VariableCast
from torch.autograd import Variable
import Distributions.distributions as dis

c24039= VariableCast(1.0)
c24040= VariableCast(2.0)
x24041 = dis.Normal(c24039, c24040)
x22542 = x24041.sample()   #sample
if isinstance(Variable, x22542):
    x22542 = Variable(x22542.data, requires_grad= True)
else:
    x22542 = Variable(VariableCast(x22542).data, requires_grad= True)
p24042 = x24041.logpdf( x22542)
c24043= VariableCast(3.0)
x24044 = dis.Normal(x22542, c24043)
c24045= VariableCast(7.0)
y22543 = c24045
p24046 = x24044.logpdf( y22543)
p24047 = Variable.add(p24042,p24046)

print(x22542)
print(p24047)

p24047.backward()

print("gradient: ", x22542.grad.data)
