#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:51
Date created:  01/09/2017

License: MIT
'''

import torch

from Depriciated.program import program
from core import VariableCast
from kinetic import Kinetic


# TO DO: Check how to call a class method within  a class
def test():
    prog_obj = program()
    logjointOrig, values_init, init_gradient  = prog_obj.generate()
    print(logjointOrig, values_init)
    print(init_gradient)
    ham_orig                   = fake_ham(logjointOrig)
    #
    # # in the future we would have to change this line so that
    # # if values is a dictionary then, we can generate a
    # # momentum with the right
    p0         = VariableCast(torch.randn(values_init.size()))
    kinetic_obj = Kinetic(p0)
    values     = values_init
    print('******** Before ********')
    print(p0)
    # first half step
    print(type(p0))
    print(type(init_gradient))
    p = p0 + 0.5 *  init_gradient
    print('******* Before ******')
    print(values)
    print(p)
    print()
    for i in range(10-1):
        print('Iter :', i )
        p      = p + 0.5 * prog_obj.eval(values,grad=True)
        values = values + 0.5 *  kinetic_obj.gauss_ke(p, grad = True)
        print('**** Inter ****')
        print(p.data)
        print(values.data)
    print('******** After ********')
    print(values)
    print(p)


def fake_ham(logjoint):
    return torch.exp(logjoint + 2.0)

test()