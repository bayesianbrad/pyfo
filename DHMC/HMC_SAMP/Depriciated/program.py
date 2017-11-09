import torch
import numpy as np
from torch.autograd import Variable
import Distributions.distributions as dis
from core import VariableCast
class program():
    ''''This needs to be a function of all free variables.
         If I provide a map of all values and computes the log density
         and assigns values rather than samples.
         If I don't provide then it samples
         For any variable values provided in the map, they are assigned

         method
         def eval

         Needs to be a class '''
    def __init__(self):
    #     '''Generating code, returns  a map of variable names / symbols
    #      store all variables of interest / latent parameters in here.
    #       Strings -  A list of all the unique numbers of the para'''
    #     # self.params = [{'x' + Strings[i] : None} for i in range(len(Strings))]
         self.params  = {'x':None}

    def calc_grad(self, logjoint, values):
        ''' Stores the gradients, grad, in a tensor, where each row corresponds to each the
            data from the Variable of the gradients '''
        # Assuming values is a dictionary we could extract the values into a list as follows
        # if isinstance(dict, values):
        #     self.params = list(values.values())
        # else:
        #     self.params = values
        if isinstance(values, list):
            grad = torch.autograd.grad([logjoint], values, grad_outputs=torch.ones(values.data.size()))
        else:
            grad = torch.autograd.grad([logjoint], values, grad_outputs=torch.ones(values.data.size()))
        # note: Having grad_outputs set to the dimensions of the first element in the list, implies that we believe all
        # other values are the same size.
        # print(grad)
        if values.size()[0] == 1:
            gradients = torch.Tensor(1,values.size()[0])
        else:
            gradients = torch.Tensor(values.size())

        for i in range(len(values)):
            gradients[i,:] = grad[i][0].data.unsqueeze(0)  # ensures that each row of the grads represents a params grad
        return gradients

class conjgauss(program):
    def __init__(self):
        super().__init__()

    def generate(self):
        ''' Generates the initial state and returns the samples and logjoint evaluated at initial samples  '''

        ################## Start FOPPL input ##########
        logp = [] # empty list to store logps of each variable
        a = VariableCast(0.0)
        b = VariableCast(2.236)
        normal_object = dis.Normal(a, b)
        x = Variable(normal_object.sample().data, requires_grad = True)

        std  = VariableCast(1.4142)
        obs2 = VariableCast(7.0)
        p_y_g_x    = dis.Normal(x, std)

        # TO DO Ask Yuan, is it either possible to have once an '.logpdf' method is initiated can we do a
        # logp.append(<'variable upon which .logpdf method used'>)
        logp.append(normal_object.logpdf(x))
        logp.append(p_y_g_x.logpdf(obs2))
        # TO DO We will need to su m all the logs here.
        # Do I have them stored in a dictionary with the value
        # or do we have a separate thing for the logs?
        ################# End FOPPL output ############

        # sum up all logs
        logp_x_y   = VariableCast(torch.zeros(1,1))
        for logprob in logp:
            logp_x_y = logp_x_y + logprob
        return logp_x_y, x, VariableCast(self.calc_grad(logp_x_y,x))
    def eval(self, values, grad= False, grad2= False):
        ''' Takes a map of variable names, to variable values . This will be continually called
            within the leapfrog step

        values      -       Type: python dict object
                            Size: len(self.params)
                            Description: dictionary of 'parameters of interest'
        grad        -       Type: bool
                            Size: -
                            Description: Flag to denote whether the gradients are needed or not
        '''
        logp = []  # empty list to store logps of each variable # In addition to foopl input
        ################## Start FOPPL input ##########
        values = Variable(values.data, requires_grad = True)
        a = VariableCast(0.0)
        b = VariableCast(2.236)
        normal_object = dis.Normal(a, b)

        std  = VariableCast(1.4142)
        obs2 = VariableCast(7.0)
        # Need a better way of dealing with values. As ideally we have a dictionary (hash map)
        # then we say if values['x']
        p_y_g_x    = dis.Normal(values, std)

        logp.append(normal_object.logpdf(values))
        logp.append(p_y_g_x.logpdf(obs2))

        ################# End FOPPL output ############
        logjoint = VariableCast(torch.zeros(1, 1))

        for logprob in logp:
            logjoint = logjoint + logprob
        # grad2 is a hack so that we can call this at the start
        if grad:
            gradients = self.calc_grad(logjoint, values)
            return VariableCast(gradients)
        elif grad2:
            gradients = self.calc_grad(logjoint, values)
            return logjoint, VariableCast(gradients)
        else:
            return logjoint, values
    # def free_vars(self):
    #     return self.params


class linearreg(program):
    def __init__(self):
        super().__init__()

    def generate(self):
        logp   = []
        parms  = []
        c23582 = VariableCast(torch.Tensor([0.0]))
        c23583 = VariableCast(torch.Tensor([10.0]))
        normal_obj1 = dis.Normal(c23582, c23583)
        x23474 = Variable(normal_obj1.sample().data, requires_grad = True)  # sample
        parms.append(x23474)
        p23585 = normal_obj1.logpdf(x23474)  # prior
        logp.append(p23585)
        c23586 = VariableCast(torch.Tensor([0.0]))
        c23587 = VariableCast(torch.Tensor([10.0]))
        normal_obj2 = dis.Normal(c23586, c23587)
        x23471 = Variable(normal_obj2.sample().data, requires_grad = True)  # sample
        parms.append(x23471)
        p23589 = normal_obj2.logpdf(x23471)  # prior
        logp.append(p23589)
        c23590 = VariableCast(torch.Tensor([1.0]))
        # Do I cast this as a variable with requires_grad = True ???
        x23591 = x23471 * c23590 + x23474 # some problem on Variable, Variable.data

        # x23592 = Variable(x23591.data + x23474.data, requires_grad = True)

        c23593 = VariableCast(torch.Tensor([1.0]))
        normal_obj2 = dis.Normal(x23591, c23593)

        c23595 = VariableCast(torch.Tensor([2.1]))
        y23481 = c23595
        p23596 = normal_obj2.logpdf(y23481)  # obs, log likelihood
        logp.append(p23596)
        c23597 = VariableCast(torch.Tensor([2.0]))

        # This is highly likely to be the next variable
        x23598 = torch.mul(x23471, c23597) + x23474
        # x23599 = torch.add(x23598, x23474)
        c23600 = torch.Tensor([1.0])
        # x23601 = dis.Normal(x23599, c23600)

        normal_obj3 = dis.Normal(x23598, c23600)
        c23602 = torch.Tensor([3.9])
        y23502 = c23602
        p23603 = normal_obj3.logpdf(y23502)  # obs, log likelihood
        logp.append(p23603)
        c23604 = Variable(torch.Tensor([3.0]))
        x23605 = Variable(torch.mul(x23471, c23604).data, requires_grad = True)
        x23606 = torch.add(x23605, x23474)
        c23607 = torch.Tensor([1.0])
        normal_obj4 = dis.Normal(x23606, c23607)
        c23609 = torch.Tensor([5.3])
        y23527 = c23609
        p23610 = normal_obj4.log_pdf(y23527)  # obs, log likelihood
        logp.append(p23610)
        p23611 = torch.add([p23585, p23589, p23596, p23603, p23610])
        # return E from the model
        # Do I want the gradients of x23471 and x23474? and nothing else.
        if grad:
            gradients = self.calc_grad(p23611, parms)
            return VariableCast(gradients)
        elif grad2:
            gradients = self.calc_grad(p23611, values)
            return p23611, VariableCast(gradients)
        else:
            return p23611, values

    def eval(self, values, grad=False, grad2=False):
        logp = []
        parms = []
        for value in values:
            if isinstance(value, Variable):
                temp = Variable(value.data, requires_grad = True)
                parms.append(temp)
            else:
                temp = VariableCast(value)
                temp = Variable(value.data, requires_grad = True)
                parms.append(value)
        c23582 = VariableCast(torch.Tensor([0.0]))
        c23583 = VariableCast(torch.Tensor([10.0]))
        normal_obj1 = dis.Normal(c23582, c23583)
        x23474 = parms[0] # sample
        parms.append(x23474)
        p23585 = normal_obj1.logpdf(x23474)  # prior
        logp.append(p23585)
        c23586 = VariableCast(torch.Tensor([0.0]))
        c23587 = VariableCast(torch.Tensor([10.0]))
        normal_obj2 = dis.Normal(c23586, c23587)
        x23471 = parms[1]  # sample
        parms.append(x23471)
        p23589 = normal_obj2.logpdf(x23471)  # prior
        logp.append(p23589)
        c23590 = VariableCast(torch.Tensor([1.0]))
        # Do I cast this as a variable with requires_grad = True ???
        x23591 = x23471 * c23590 + x23474  # some problem on Variable, Variable.data

        # x23592 = Variable(x23591.data + x23474.data, requires_grad = True)

        c23593 = VariableCast(torch.Tensor([1.0]))
        normal_obj2 = dis.Normal(x23591, c23593)

        c23595 = VariableCast(torch.Tensor([2.1]))
        y23481 = c23595
        p23596 = normal_obj2.logpdf(y23481)  # obs, log likelihood
        logp.append(p23596)
        c23597 = VariableCast(torch.Tensor([2.0]))

        # This is highly likely to be the next variable
        x23598 = torch.mul(x23471, c23597) + x23474
        # x23599 = torch.add(x23598, x23474)
        c23600 = torch.Tensor([1.0])
        # x23601 = dis.Normal(x23599, c23600)

        normal_obj3 = dis.Normal(x23598, c23600)
        c23602 = torch.Tensor([3.9])
        y23502 = c23602
        p23603 = normal_obj3.logpdf(y23502)  # obs, log likelihood
        logp.append(p23603)
        c23604 = torch.Tensor([3.0])
        x23605 = Variable(torch.mul(x23471, c23604).data, requires_grad=True)
        x23606 = torch.add(x23605, x23474)
        c23607 = torch.Tensor([1.0])
        normal_obj4 = dis.Normal(x23606, c23607)
        c23609 = torch.Tensor([5.3])
        y23527 = c23609
        p23610 = normal_obj4.log_pdf(y23527)  # obs, log likelihood
        logp.append(p23610)
        p23611 = torch.add([p23585, p23589, p23596, p23603, p23610])
        # return E from the model
        # Do I want the gradients of x23471 and x23474? and nothing else.
        if grad:
            gradients = self.calc_grad(p23611, parms)
            return VariableCast(gradients)
        elif grad2:
            gradients = self.calc_grad(p23611, values)
            return p23611, VariableCast(gradients)
        else:
            return p23611, values
class conditionalif(program):
    ''''This needs to be a function of all free variables.
         If I provide a map of all vlues and computes the log density
         and assigns values rather than samples.
         If I don't provide then it samples
         For any variable values provided in the map, they are assigned

         method
         def eval

         Needs to be a class '''
    def __init__(self):
        '''Generating code, returns  a map of variable names / symbols '''
        self.params = {'x': None}

    def generate(self):
        logp = []  # empty list to store logps of each variable
        a = VariableCast(0.0)
        b = VariableCast(1)
        c1 = VariableCast(-1)
        normal_obj1 = dis.Normal(a, b)
        x = Variable(normal_obj1.sample().data, requires_grad=True)
        logp_x = normal_obj1.logpdf(x)

        if torch.gt(x.data,torch.zeros(x.size()))[0][0]:
            y           = VariableCast(1)
            normal_obj2 = dis.Normal(b,b)
            logp_y_x = normal_obj2.logpdf(y)
        else:
            y = VariableCast(1)
            normal_obj3 = dis.Normal(c1,b)
            logp_y_x    = normal_obj3.logpdf(y)

        logp_x_y = logp_x + logp_y_x

        return logp_x_y, x, VariableCast(self.calc_grad(logp_x_y, x))

        # sum up all logs
        logp_x_y = VariableCast(torch.zeros(1, 1))
        for logprob in logp:
            logp_x_y = logp_x_y + logprob
        return logp_x_y, x, VariableCast(self.calc_grad(logp_x_y, x))
    def eval(self, values, grad= False, grad2= False):
        ''' Takes a map of variable names, to variable values '''
        a = VariableCast(0.0)
        b = VariableCast(1)
        c1 = VariableCast(-1)
        normal_obj1 =dis.Normal(a, b)
        values = Variable(values.data, requires_grad = True)
        logp_x = normal_obj1.logpdf(values)
        # else:
        #     x = normal_object.sample()
        #     x = Variable(x.data, requires_grad = True)
        if torch.gt(values.data,torch.zeros(values.size()))[0][0]:
            y           = VariableCast(1)
            normal_obj2 = dis.Normal(b,b)
            logp_y_x = normal_obj2.logpdf(y)
        else:
            y = VariableCast(1)
            normal_obj3 = dis.Normal(c1,b)
            logp_y_x    = normal_obj3.logpdf(y)

        logjoint = Variable.add(logp_x, logp_y_x)
        if grad:
            gradients = self.calc_grad(logjoint, values)
            return VariableCast(gradients)
        elif grad2:
            gradients = self.calc_grad(logjoint, values)
            return logjoint, VariableCast(gradients)
        else:
            return logjoint, values
