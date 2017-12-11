#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:10
Date created:  05/12/2017

License: MIT
'''


def Firstdiscontinuitiy(x, p, stepsze,t0, potential):
    """
    This function returns x_new, the position of the intersection of the first intersection of a
    boundary plain with line segment [q, q+(stepsize-t0)p], t_x the time at which the boundary was hit,
    :param x:
    :param p:
    :param stepsze:
    :param t0:
    :param potential:
    :return: x_new, t_x, delta_U, phi(partition boundary)

    """
    t_x = 0
    x_new = 0
    delta_U = 0
    phi = 0


    return x_new, t_x, delta_U, phi