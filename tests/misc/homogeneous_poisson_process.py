#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:32
Date created:  03/04/2018

License: MIT
'''

import numpy as np
import torch
import torch.distributions as dists
import matplotlib.pyplot as plt

t = 0
event_range = 1000
E_lam = []
rates = [7,10,100]
event_times = []
event_times.append(t)
unif = dists.Uniform(0,1)
for rate in rates:
    for i in range(event_range):
        u = unif.sample()
        temp = -torch.log(u)/rate
        E_lam.append(temp)
        # print(E_lam)
        event_times.append(event_times[i] + E_lam[i])
        # print('Debug: The event times: {}'.format(event_times))
    for i in range(len(E_lam)):
        E_lam[i] = E_lam[i].numpy()
    plt.plot(event_times[1:], E_lam[:], label = str(rate))
    E_lam = []
    event_times = [t]
# plt.plot(event_times[1:], E_lam[:])
plt.legend()
plt.show()