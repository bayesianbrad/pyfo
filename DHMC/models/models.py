4#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  07:51
Date created:  09/10/2017

License: MIT
'''
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
from scipy import stats, integrate
from HMC_Yuan import HMC
from HMC_Yuan.Distributions import *
seed = 1
torch.manual_seed(seed=seed)
np.random.seed(seed)
models = ['Discrete1d', 'Discrete10d', 'Flat', 'Conif1d', 'Conif2d']
n_burnin = 10 ** 3
n_sample = 10 ** 3
dt = .3 * np.array([.01, 0.5])
nstep = [10, 20]
class Discrete1d():
    def __init__(self):
        self.p = torch.Tensor([0.1, 0.5, 0.2, 0.2])
        self.y = torch.Tensor([[2]])
        self.std = torch.Tensor([[2]])


    def f(self,x0):
        logp = 0
        x = Variable(x0.data, requires_grad = True)
        dist1 = Categorical_Trans(self.p, "standard")
        logp1 = dist1.logpdf(x)
        dist2 = MultivariateNormal(x, self.std)
        logp2 =  dist2.logpdf(self.y)
        logp = logp1 + logp2
        grad = torch.autograd.grad(logp2, x)[0]

        aux = logp
        return logp, grad, aux #logp, var; grad, var

    def f_update(self,x, dx, j, aux):

        logp_prev = aux

        x_new = x.clone()
        x_new.data[j] = x_new.data[j] + dx

        dist1 = Categorical_Trans(self.p, "standard")
        logp1 = dist1.logpdf(x_new)
        dist2 = MultivariateNormal(x_new, self.std)
        logp2 = dist2.logpdf(self.y)
        logp = logp1 + logp2

        logp_diff = logp - logp_prev
        aux_new = logp

        return logp_diff, aux_new

    def run_hmc(self):
        x0 = Variable(self.p.shape[0]*torch.rand(1,1),requires_grad = True)
        # DHMC
        n_param = 1
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept =\
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

    def run_dhmc(self):
        x0 = Variable(self.p.shape[0]*torch.rand(1,1),requires_grad = True)
        # DHMC
        n_param = 1
        n_disc = 1
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept =\
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept
class Discrete10d():

    def __init__(self):
        self.p = torch.Tensor([[0.1, 0.5, 0.2, 0.2],
                          [0.1, 0.1, 0.3, 0.5],
                          [0.1, 0.5, 0.2, 0.2],
                          [0.1, 0.1, 0.3, 0.5],
                          [0.1, 0.5, 0.2, 0.2],
                          [0.1, 0.1, 0.3, 0.5],
                          [0.1, 0.5, 0.2, 0.2],
                          [0.1, 0.1, 0.3, 0.5],
                          [0.1, 0.5, 0.2, 0.2],
                          [0.1, 0.1, 0.3, 0.5]])
        self.cov = torch.diag(torch.ones(p.shape[0]))
        self.y = torch.Tensor([[2],
                          [3],
                          [2],
                          [3],
                          [2],
                          [3],
                          [2],
                          [3],
                          [2],
                          [3]])

    def f(self, x0):
        """
        :param x: [[x1],[x2]]
        :return:
        """
        # p.shape => [k, m]  p[k] = [pk1, pk2, ..., pkm]
        x = Variable(x0.data, requires_grad=True)
        logp_prior = 0
        for i in range(self.shape[0]):
            dist = Categorical_Trans(p[i], "standard")
            logp_prior += dist.logpdf(x[i])

        dist_lik = MultivariateNormal(x, self.cov)
        logp_lik = dist_lik.logpdf(self.y)
        logp = logp_prior + logp_lik
        grad = torch.autograd.grad(logp_lik, x)[0]
        aux = logp
        return logp, grad, aux  # logp, var; grad, var

    def f_only(self, x0):
        x = Variable(x0.data, requires_grad=True)
        logp_prior = 0
        for i in range(self.p.shape[0]):
            dist = Categorical_Trans(self.p[i], "standard")
            logp_prior += dist.logpdf(x[i])

        dist_lik = MultivariateNormal(x, self.cov)
        logp_lik = dist_lik.logpdf(self.y)
        logp = logp_prior + logp_lik
        return logp

    def f_update(self, x, dx, j, aux):

        logp_prev = aux

        x_new = x.clone()
        x_new.data[j] = x_new.data[j] + dx

        logp = self.f_only(x_new)

        logp_diff = logp - logp_prev
        aux_new = logp

        return logp_diff, aux_new

    def run_dhmc(self):
        x0 = Variable(self.p.shape[1] * torch.rand(self.p.shape[0], 1), requires_grad=True)
        n_param = self.p.shape[0]
        n_disc = 10
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

    def run_hmc(self):
        x0 = Variable(self.p.shape[1] * torch.rand(self.p.shape[0], 1), requires_grad=True)
        # DHMC
        n_param = self.p.shape[0]
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

class Flat():

    def __init__(self):
        self.cat_p = torch.Tensor([0.1, 0.5, 0.2, 0.2])
        self.n = torch.Tensor([[5]])
        self.p = torch.Tensor([[0.5]])

    def f(self, x0):
        x = Variable(x0.data, requires_grad=True)
        dist1 = Categorical_Trans(self.cat_p, "standard")
        logp1 = dist1.logpdf(x)
        dist2 = Binomial_Trans(self.n, self.p, "standard")
        logp2 = dist2.logpdf(x)
        logp = logp1 + logp2
        grad = Variable(torch.zeros(x.data.shape[0]))
        aux = logp
        return logp, grad, aux  # logp, var; grad, va

    def f_update(self,x, dx, j, aux):
        logp_prev = aux

        x_new = x.clone()
        x_new.data[j] = x_new.data[j] + dx

        logp, _, _ = self.f(x_new)
        logp_diff = logp - logp_prev
        aux_new = logp
        return logp_diff, aux_new

    def run_hmc(self):
        x0 = Variable(p.shape[0] * torch.rand(1, 1) + 1, requires_grad=True)
        # DHMC
        n_param = 1
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = \
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept
    def run_dhmc(self):
        x0 = Variable(p.shape[0] * torch.rand(1, 1) + 1, requires_grad=True)
        # DHMC
        n_param = 1
        n_disc = 1
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = \
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

class Condif1d():

    def __init__(self):
        self.mu0 = torch.Tensor([[0]])
        self.mu1 = torch.Tensor([[1]])
        self.mu2 = torch.Tensor([[-1]])
        self.std = torch.Tensor([[1]])

    def f(self,x0):
        x = Variable(x0.data, requires_grad=True)
        dist1 = MultivariateNormal(self.mu0, self.std)
        # x = dist1.sample()
        y = torch.Tensor([[1]])
        if x.data[0, 0] > 0:
            dist2 = MultivariateNormal(self.mu1, self.std)
        else:
            dist2 = MultivariateNormal(self.mu2, self.std)
        logp = dist1.logpdf(x) + dist2.logpdf(y)
        grad = torch.autograd.grad(logp, x)[0]  # grad var
        aux = logp
        return logp, grad, aux

    def f_only(self,x0):
        x = Variable(x0.data, requires_grad=True)
        dist1 = MultivariateNormal(self.mu0, self.std)
        # x = dist1.sample()
        y = torch.Tensor([[1]])
        if x.data[0, 0] > 0:
            dist2 = MultivariateNormal(self.mu1, self.std)
        else:
            dist2 = MultivariateNormal(self.mu2, self.std)
        logp = dist1.logpdf(x) + dist2.logpdf(y)
        return logp

    def f_update(self,x, dx, j, aux):
        logp_prev = aux

        x_new = x.clone()
        x_new.data[j] = x_new.data[j] + dx
        logp = self.f_only(x_new)
        logp_diff = logp - logp_prev

        aux_new = logp
        return logp_diff, aux_new

    def run_hmc(self):
        x0 = MultivariateNormal(self.mu0, self.std).sample()
        # DHMC
        n_param = 1
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = \
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

    def run_dhmc(self):
        x0 = MultivariateNormal(self.mu0, self.std).sample()
        # DHMC
        n_param = 1
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = \
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

class Conif2d():

    def __init__(self):
        self.mu0 = torch.Tensor([[0], [0]])
        self.mu1 = torch.Tensor([[1], [1]])
        self.mu2 = torch.Tensor([[-1], [1]])
        self.mu3 = torch.Tensor([[1], [-1]])
        self.mu4 = torch.Tensor([[-1], [-1]])
        self.cov = torch.Tensor([[1, 0], [0, 1]])
        self.y = torch.Tensor([[1], [1]])

    def f(self,x0):
        x = Variable(x0.data, requires_grad=True)
        dist1 = MultivariateNormal(self.mu0, self.cov)

        if x.data[0, 0] > 0 and x.data[1, 0] > 0:
            dist2 = MultivariateNormal(self.mu1, self.cov)
        elif x.data[0, 0] <= 0 and x.data[1, 0] > 0:
            dist2 = MultivariateNormal(self.mu2, self.cov)
        elif x.data[0, 0] > 0 and x.data[1, 0] <= 0:
            dist2 = MultivariateNormal(self.mu3, self.cov)
        elif x.data[0, 0] <= 0 and x.data[1, 0] <= 0:
            dist2 = MultivariateNormal(self.mu4, self.cov)

        logp = dist1.logpdf(x) + dist2.logpdf(self.y)
        grad = torch.autograd.grad(logp, x)[0]  # grad var

        aux = logp
        return logp, grad, aux

    def f_only(self,x0):
        x = Variable(x0.data, requires_grad=True)
        dist1 = MultivariateNormal(self.mu0, self.cov)

        if x.data[0, 0] > 0 and x.data[1, 0] > 0:
            dist2 = MultivariateNormal(self.mu1, self.cov)
        elif x.data[0, 0] <= 0 and x.data[1, 0] > 0:
            dist2 = MultivariateNormal(self.mu2, self.cov)
        elif x.data[0, 0] > 0 and x.data[1, 0] <= 0:
            dist2 = MultivariateNormal(self.mu3, self.cov)
        elif x.data[0, 0] <= 0 and x.data[1, 0] <= 0:
            dist2 = MultivariateNormal(self.mu4, self.cov)

        logp = dist1.logpdf(x) + dist2.logpdf(self.y)
        return logp

    def f_update(self, x, dx, j, aux):
        logp_prev = aux

        x_new = x.clone()
        x_new.data[j] = x_new.data[j] + dx
        logp = self.f_only(x_new)
        logp_diff = logp - logp_prev
        aux_new = logp

        return logp_diff, aux_new

    def run_hmc(self):
        x0 = MultivariateNormal(self.mu0, self.cov).sample()
        n_param = 2
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept =\
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

    def run_hmc(self):
        x0 = MultivariateNormal(self.mu0, self.cov).sample()
        n_param = 2
        n_disc = 0
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = \
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept

    def run_dhmc(self):
        x0 = MultivariateNormal(self.mu0, self.cov).sample()
        n_param = 2
        n_disc = 2
        hmc_obj = HMC(self.f, n_disc, n_param, self.f_update)
        samples, accept = \
            hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
        return samples.numpy(), accept


