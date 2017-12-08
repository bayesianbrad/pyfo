#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  10:48
Date created:  08/12/2017

License: MIT
'''
import numpy as np
import warnings

def test_cont_grad(self, x0, sd=1, atol=None, rtol=.01, dx=10 ** -6, n_test=10):
    """
    Wrapper function for test_grad to check the returned gradient values
    (with respect to the continuous parameters). The gradients are
    evaluated at n_test randomly generated points around x0.
    """

    if atol is None:
        atol = dx

    for i in range(n_test):
        x = x0.copy()
        x[-self.n_disc:] += sd * np.random.randn(self.n_disc)

        def f_test(x_cont):
            logp, grad, aux \
                = self.f(np.concatenate((x_cont, x0[-self.n_disc:])))
            grad = grad[:-self.n_disc]
            return logp, grad

        test_pass, x_cont, grad, grad_est \
            = self.test_grad(f_test, x[:-self.n_disc], atol, rtol, dx)

        if not test_pass:
            warnings.warn(
                'Test failed: the returned gradient value does not agree with ' +
                'the centered difference approximation within the tolerance level.',
                RuntimeWarning
            )
            break

    if test_pass:
        print('Test passed! The computed gradient seems to be correct.')

    return test_pass, x_cont, grad, grad_est


def test_grad(self, f, x, atol, rtol, dx):
    """Compare the computed gradient to a centered finite difference approximation. """
    x = np.array(x, ndmin=1)
    grad_est = np.zeros(len(x))
    for i in range(len(x)):
        x_minus = x.copy()
        x_minus[i] -= dx
        x_plus = x.copy()
        x_plus[i] += dx
        f_minus, _ = f(x_minus)
        f_plus, _ = f(x_plus)
        grad_est[i] = (f_plus - f_minus) / (2 * dx)

    _, grad = f(x)
    test_pass = np.allclose(grad, grad_est, atol=atol, rtol=rtol)

    return test_pass, x, grad, grad_est


def test_update(self, x0, sd, n_test=10, atol=10 ** -3, rtol=10 ** -3):
    """
    Check that the outputs of 'f' and 'f_update' functions are consistent
    by comparing the values logp differences computed by the both functions.
    """

    test_pass = True
    for i in range(n_test):
        index = np.random.randint(self.n_param - self.n_disc, self.n_param)
        x = x0 + .1 * sd * np.random.randn(len(x0))
        dtheta = sd * np.random.randn(1)
        x_new = x.copy()
        x_new[index] += dtheta
        logp_prev, _, aux = self.f(x)
        logp_curr, _, _ = self.f(x_new)
        logp_diff, _ = self.f_update(x, dtheta, index, aux)
        both_inf = math.isinf(logp_diff) \
                   and math.isinf(logp_curr - logp_prev)
        if not both_inf:
            abs_err = abs(logp_diff - (logp_curr - logp_prev))
            if logp_curr == logp_prev:
                rel_err = 0
            else:
                rel_err = abs_err / abs(logp_curr - logp_prev)
            if abs_err > atol or rel_err > rtol:
                test_pass = False
                break

    if test_pass:
        print('Test passed! The logp differences agree.')
    else:
        warnings.warn(
            'Test failed: the outputs of f and f_update are not consistent.' +
            'the logp differences do not agree.',
            RuntimeWarning
        )
    return test_pass, x, logp_diff, logp_curr - logp_prev
