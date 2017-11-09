import numpy as np
from torch.autograd import Variable
import torch
from Distributions_yuan import *
# import tensorflow as tf


def test_grad_backward():
    distobj = Normal(0,1)
    x = distobj.sample()
    # x.retain_grad()
    print("x: ", x)
    y = 2 * x
    # y = distobj.logpdf(x)
    y.backward()
    print("y: ", y,", x: ", x.grad)

# # disttensor = torch.Tensor([distobj])
# var = Variable(distobj.sample())
# x = Variable(torch.Tensor([1.0]))
# # dist = Variable(Normal(0,1))
# # print(distobj.logpdf(7), torch.add(distobj.logpdf(7), distobj.logpdf(7)))
#
# # print(x.data)
# test_grad_backward()

def test_grad():
    mu = 0
    sig = 1
    dist = Normal(mu, sig)
    x = dist.sample()
    print("x: ", x.data[0])
    p_x = dist.logpdf(x)
    print("p_x", p_x.data[0])
    g_x = torch.autograd.grad(outputs=[p_x], inputs=[x])[0]
    print("g_x: ", g_x.data[0])
    y = 7
    obs_dist = Normal(x, 5)
    p_y_x = obs_dist.logpdf(y)
    g_y_x = torch.autograd.grad(outputs=[p_y_x], inputs=[x])[0]
    print("g_y_x:", g_y_x.data[0])
# test_grad()

def get_logpdf(x, y):
    return Normal(0, 5).logpdf(x) + Normal(x, 2).logpdf(y)

def test_logpdf_grad():
    x = Normal(0, 5).sample()
    y = 7
    p = get_logpdf(x, y)
    p.backward()
    print(x.data, p, x.grad.data)
    x.data = VariableCast(6).data
    x.grad.data.zero_()
    p = get_logpdf(x, y)
    p.backward()
    print(x.data, p, x.grad.data)


def test_mvn():
    # mean = torch.Tensor([[0],[0], [0]])
    # cov  = torch.Tensor([[1,0,0],[0,1,0], [0,0,1]])
    mean = torch.Tensor([[0],[0]])
    cov  = torch.Tensor([[1,0.8],[0.8,1]])
    mvn = MultivariateNormal(mean,cov)
    sample = mvn.sample()
    print("some sample, ", sample)
    log_p = mvn.logpdf(sample)
    print("log pdf: ", log_p.data)
    log_p.backward()
    print("test g: ", sample.grad.data)   # need to double check the g
# test_mvn()

def test_fndef():
    mean = torch.Tensor([[0], [0]])
    cov = torch.Tensor([[1, 0.8], [0.8, 1]])
    mvn = MultivariateNormal(mean, cov)
    obs = torch.Tensor([[7], [7]])
    def get_logpdf(obs):
        return mvn.logpdf(obs)
    print(get_logpdf(obs))

def test_logjoint():
    dist1 = Normal(0, np.sqrt(5))
    initial_x = dist1.sample()

    def p_x(x):
        return Normal(0, np.sqrt(5)).logpdf(x)

    obs = Variable(torch.Tensor([7]))

    def p_y_x(x, y):
        return Normal(x, np.sqrt(2)).logpdf(y)

    def log_joint(x, y=obs):
        # return Normal(0, np.sqrt(5)).logpdf(x) + Normal(x, np.sqrt(2)).logpdf(y)
        return p_x(x) + p_y_x(x, y)

    p = log_joint(initial_x)
    g = p.backward()
    print("x, g", initial_x, initial_x.grad.data)

    initial_x.data = torch.Tensor([4.5])
    initial_x.grad.data.zero_()
    p = log_joint(initial_x)
    g = p.backward()
    print("x, g", initial_x, initial_x.grad.data)

    initial_x.data = torch.Tensor([ -1.8225173950195312])
    initial_x.grad.data.zero_()
    p = log_joint(initial_x)
    g = p.backward()
    print("x, g", initial_x, initial_x.grad.data)
# test_logjoint()

def test1():
    dist1 = Normal(0, np.sqrt(5))
    x = Variable(torch.Tensor([-1.8225]), requires_grad = True)
    y = Variable(torch.Tensor([7]))

    p = Normal(0, np.sqrt(5)).logpdf(x) + Normal(x, np.sqrt(2)).logpdf(y)

    print(torch.autograd.grad(outputs=[p], inputs=[x]).data[0])

# test1()


def test_cat():
    p = torch.Tensor([0.1, 0.5, 0.2, 0.2])
    x = Variable(torch.Tensor([0.5]), requires_grad=True)
    dist = Categorical_Trans(p, "standard")
    logp = dist.logpdf(x)
    # grad = torch.autograd.grad(logp, x)
    print(x, logp)
test_cat()

def test_cat_grad():
    p = torch.Tensor([0.1, 0.5, 0.2, 0.2])
    x = Variable(torch.Tensor([0.5]), requires_grad=True)
    if 0< x.data[0] < 1:
        index = int(torch.floor(x.data)[0])
        logp = torch.log(torch.Tensor([0.1]))
        # grad = torch.autograd.grad(logp, x)
        print(x.data, logp)

# test_cat_grad()