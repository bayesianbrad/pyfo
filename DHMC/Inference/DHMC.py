import math

import numpy as np
import torch


class HMC(object):
    """
    object: Is the compiled output
        method: gen_prior_samples()
        inputs: -
        returns: Xs - An ordered vector of the continous and discrete parameters
                     Will be transformed formed into a variable in this script
        method: gen_pdf(Xs, compute_grad )
        inputs: Xs - A Variable of the latent parameters
                compute_grad - boolean
        :returns log_pdf, grad, prev-log

        method: gen_ordered_vars()
        inputs: -
        returns: Xs ordered
        method:


    """
    def __init__(self, log_posterior,  n_disc, n_param ,logp_update=None, M=None, scale=None):
        # if scale is None:
        #     scale = Variable(torch.ones(n_param))
        # # Set the scale of p to be inversely proportional to the scale of theta.
        # self.M = 1 / torch.cat((scale[:-n_disc] ** 2, scale[-n_disc:]))
        self.n_param = n_param
        self.n_disc = n_disc
        self.n_cont = n_param - n_disc
        self.logp_update = logp_update
        self.log_posterior = log_posterior  # return log_p, and grad
        if M is None:
            self.M = torch.ones(n_param, 1)

    def kinetic_energy(self, velocity): # v, tensor; return float value
        # kinetic = 0.5*torch.matmul(torch.t(velocity), velocity)
        kinetic =  0.5*(velocity ** 2).sum()
        return kinetic

    def potential_energy(self, position): # pos = x, var ; return float value
        log_p, _, _ = self.log_posterior(position)
        return -log_p.data[0,0]

    def hamiltonian(self, position, velocity):
        total = self.potential_energy(position) + self.kinetic_energy(velocity)
        return total

    def sample_momentum(self, x, scale = None):
        v = torch.zeros(x.data.shape)

        if self.n_cont != 0:
            v[:self.n_cont] = torch.sqrt(self.M[:self.n_cont]) * torch.randn(self.n_cont,x.data.shape[1])
        if self.n_disc != 0:
            v[:self.n_disc] = self.M[:self.n_disc] * torch.Tensor(np.random.laplace(size=(self.n_disc, x.data.shape[1])))
        if scale is not None:
            v =  v * scale
        return v

    def leapfrog_step(self, x, v, step_size, num_steps):

        if x.data.shape[1] != 1:
            leap_sample = torch.zeros(num_steps, 2*x.data.shape[0], x.data.shape[1])
        else:
            leap_sample = torch.zeros( num_steps, 2*len(x.data))
        # l = 0,..., L-1
        for l in range(0, num_steps ):
            x, v, logp, grad = self.one_leapfrog_step(x, v, step_size)
            leap_sample[l] = torch.cat((torch.t(x.data), torch.t(v)), 0)
            if math.isinf(logp.data[0,0]):
                # return theta, p, grad, logp, n_feval, n_fupdate  # Yuan
                # print("in leapfrog: logp inf!")
                break

        return x, v

    def one_leapfrog_step(self, x, v, step_size):
        n_disc = self.n_disc
        n_cont = self.n_cont
        M = self.M

        # cont v, 0.5 step
        logp, grad, aux = self.log_posterior(x)
        if n_cont != 0:
            v[:n_cont] = v[:n_cont] + 0.5 * step_size * grad.data[:n_cont]

        if n_disc == 0: # all cont
            x.data = x.data + step_size * v
        else:  # contain disc
            # update cont part first if any for 0.5 step
            if n_cont != 0:  #double check whether x is modified
                # x[:n_cont].data = x[:n_cont].data + 0.5 * step_size * v[:n_cont] / M[:n_cont]  # x would not change this way
                x.data[:n_cont] = x.data[:n_cont] + 0.5 * step_size * v[:n_cont] / M[:n_cont]
                logp, grad, aux = self.log_posterior(x)

            # logp, _, aux = self.log_posterior(x)
            # if math.isinf(logp.data[0,0]):
            #     print("in one leapfrog: logp inf!")
            #     return x, v, logp, grad

            # update disc part for 1 step
            x, v, logp, aux = self.update_disc(x, v, step_size, M, n_disc, logp, aux)

            # update cont part if any for 0.5 step
            if n_cont != 0:
                # x[:n_cont].data = x[:n_cont].data + 0.5 * step_size * v[:n_cont] / M[:n_cont]  # x would not change this way
                x.data[:n_cont] = x.data[:n_cont] + 0.5 * step_size * v[:n_cont] / M[:n_cont]

        # cont v, 0.5 step
        if n_cont != 0:
            logp, grad, aux = self.log_posterior(x)
            v[:n_cont] = v[:n_cont] + 0.5 * step_size * grad.data[:n_cont]

        return x, v, logp, grad

    def update_disc(self, x, v, step_size, M, n_disc, logp, aux):
        coord_order = len(x.data) - n_disc + np.random.permutation(n_disc)

        for j in coord_order:
            v_sign = math.copysign(1.0, v[j][0])  # double check v[j][0]
            dx  = v_sign /M[j] * step_size  # tensor
            logp_diff, aux_new = self.logp_update(x, dx, j, aux)
            dU = - logp_diff.data[0,0]
            K_prev = (torch.abs(v[j]) / M[j])[0]
            if K_prev > dU:
                v[j] += - v_sign * M[j] * dU
                x.data[j] = x.data[j] + dx
                logp += logp_diff
                aux = aux_new
            else:
                v[j] = - v[j]

        return x, v, logp, aux

    def run_hmc(self, initial_x, step_size_range, num_steps_range,num_burnin, num_samples, seed=1):
        np.random.seed(seed)
        total_num = num_burnin + num_samples
        if initial_x.data.shape[1] != 1:
            all_sample = torch.zeros(num_burnin + num_samples, initial_x.data.shape[0], initial_x.data.shape[1])
        else:
            all_sample = torch.zeros( num_burnin+num_samples, len(initial_x.data))
        x0 = initial_x
        burnin_accept = 0
        total_accept = 0

        accept_index = np.zeros(num_burnin+num_samples)

        for i in range(num_burnin+num_samples):

            # step 1: HMC
            ## set up parameter
            step_size = torch.Tensor(1).uniform_(step_size_range[0], step_size_range[1])
            num_steps = np.random.randint(num_steps_range[0], num_steps_range[1] + 1)

            x0_orig = x0.clone()  # pass by ref, later operation would change x0 value, store the original value
            v0 = self.sample_momentum(x0_orig, 0.5)
            v0_orig = v0.clone()

            ## leapfrog
            x, v = self.leapfrog_step(x0,
                              v0,
                              step_size=step_size,
                              num_steps=num_steps)

            # step 2: accept
            orig = self.hamiltonian(x0_orig, v0_orig)
            current = self.hamiltonian(x, v)
            delta_hamiltonian = np.exp(orig - current)
            p_accept = np.min([1.0, delta_hamiltonian])

            if p_accept > np.random.uniform():
                x0 = x
                total_accept = total_accept + 1
                if i < num_burnin:
                    burnin_accept = burnin_accept + 1
                accept_index[i] = 1
            else:
                x0 = x0_orig

            all_sample[i] =  torch.t(x0.data)
            if (i + 1) % 1000 == 0:
                print('{:d} iterations have been completed.'.format(i + 1))

        return all_sample, accept_index # [total_accept, burnin_accept]
