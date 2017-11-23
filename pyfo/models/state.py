#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:12
Date created:  22/11/2017

License: MIT
'''

class State(object):
    """
    Stores the state of the object in  a has map

    """

    def __init__(self, foppy_out):

        self._foppl_out = foppl_out #it should contain the latents and obvs

        # create intial state

    def intiate(self):
        """
        :param
        :return:
        """
        state = ['params', 'obs']
        state = {i: {} for i in state}
        return state
