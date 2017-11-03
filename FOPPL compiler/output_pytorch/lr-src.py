import torch
import numpy as np
from torch.autograd import Variable
from Distributions import *

c23582= torch.Tensor([0.0])
c23583= torch.Tensor([10.0])
x23584 = Normal(c23582, c23583)
x23474 =  x23584.sample()  #sample
p23585 = x23584.logpdf( x23474) #prior
c23586= torch.Tensor([0.0])
c23587= torch.Tensor([10.0])
x23588 = Normal(c23586, c23587)
x23471 =     x23588.sample()  #sample
p23589 = x23588.logpdf( x23471)   #prior
c23590= torch.Tensor([1.0])
x23591 = torch.mul(x23471.data, c23590)  #some problem on Variable, Variable.data
x23592 = torch.add(x23591,x23474.data)
c23593= torch.Tensor([1.0])
x23594 = Normal(x23592,  c23593)
c23595= torch.Tensor([2.1])
y23481 = c23595
p23596 = x23594.logpdf( y23481)  #obs, log likelihood
c23597= torch.Tensor([2.0])
x23598 = torch.mul(x23471, c23597)
x23599 = torch.add(x23598,x23474)
c23600= torch.Tensor([1.0])
x23601 = Normal(x23599,  c23600)
c23602= torch.Tensor([3.9])
y23502 = c23602
p23603 = x23601.logpdf( y23502)  #obs, log likelihood
c23604= torch.Tensor([3.0])
x23605 = torch.mul(x23471, c23604)
x23606 = torch.add(x23605,x23474)
c23607= torch.Tensor([1.0])
x23608 = Normal(x23606,  c23607)
c23609= torch.Tensor([5.3])
y23527 = c23609
p23610 = x23608.log_pdf( y23527)  #obs, log likelihood
p23611 = torch.add([p23585,p23589,p23596,p23603,p23610])
# return E from the model
x23612 = [x23471,x23474]


print("log joint: ", p23611)
print(x23612)

### tf code
# import tensorflow as tf
# from tensorflow.contrib.distributions import *
# c23582= tf.constant(0.0)
# c23583= tf.constant(10.0)
# x23584 = Normal(mu=c23582, sigma=c23583)
# x23474 = tf.Variable( x23584.sample())   #sample
# p23585 = x23584.log_pdf( x23474) if x23584.is_continuous else x23584.log_pmf( x23474)   #prior
# c23586= tf.constant(0.0)
# c23587= tf.constant(10.0)
# x23588 = Normal(mu=c23586, sigma=c23587)
# x23471 = tf.Variable( x23588.sample())   #sample
# p23589 = x23588.log_pdf( x23471) if x23588.is_continuous else x23588.log_pmf( x23471)   #prior
# c23590= tf.constant(1.0)
# x23591 = tf.multiply(x23471, c23590)
# x23592 = tf.add_n([x23591,x23474])
# c23593= tf.constant(1.0)
# x23594 = Normal(mu=x23592, sigma=c23593)
# c23595= tf.constant(2.1)
# y23481 = c23595
# p23596 = x23594.log_pdf( y23481) if x23594.is_continuous else x23594.log_pmf( y23481) #obs, log likelihood
# c23597= tf.constant(2.0)
# x23598 = tf.multiply(x23471, c23597)
# x23599 = tf.add_n([x23598,x23474])
# c23600= tf.constant(1.0)
# x23601 = Normal(mu=x23599, sigma=c23600)
# c23602= tf.constant(3.9)
# y23502 = c23602
# p23603 = x23601.log_pdf( y23502) if x23601.is_continuous else x23601.log_pmf( y23502) #obs, log likelihood
# c23604= tf.constant(3.0)
# x23605 = tf.multiply(x23471, c23604)
# x23606 = tf.add_n([x23605,x23474])
# c23607= tf.constant(1.0)
# x23608 = Normal(mu=x23606, sigma=c23607)
# c23609= tf.constant(5.3)
# y23527 = c23609
# p23610 = x23608.log_pdf( y23527) if x23608.is_continuous else x23608.log_pmf( y23527) #obs, log likelihood
# p23611 = tf.add_n([p23585,p23589,p23596,p23603,p23610])
# # return E from the model
# x23612 = [x23471,x23474]
#
# sess = tf.Session()
# sess.run(x23474.initializer)
# sess.run(x23471.initializer)
# sess.run(p23611)
# # printing E:
# print(sess.run(x23612))
# writer = tf.summary.FileWriter( './Graph_Output/g23613', sess.graph)
# sess.close()