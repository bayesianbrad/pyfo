import torch
import numpy as np
from torch.autograd import Variable
from Distributions import *

c24039= VariableCast(1.0)
c24040= VariableCast(2.0)
x24041 = Normal(c24039, c24040)
x22542 = Variable(torch.Tensor([0.0]),  requires_grad = True)
# x22542.detach()
# x22542 = x24041.sample()   #sample
p24042 = x24041.logpdf( x22542)
c24043= VariableCast(3.0)
x24044 = Normal(x22542, c24043)
c24045= VariableCast(7.0)
y22543 = c24045
p24046 = x24044.logpdf( y22543)
p24047 = Variable.add(p24042,p24046)

print(x22542)
print(p24047)

p24047.backward()

print("gradient: ", x22542.grad.data)


### tf code
# c24039= tf.constant(1.0)
# c24040= tf.constant(1.0)
# x24041 = Normal(mu=c24039, sigma=c24040)
# x22542 = tf.Variable( x24041.sample())   #sample
# p24042 = x24041.log_pdf( x22542) if x24041.is_continuous else x24041.log_pmf( x22542)   #prior
# c24043= tf.constant(1.0)
# x24044 = Normal(mu=x22542, sigma=c24043)
# c24045= tf.constant(7.0)
# y22543 = c24045
# p24046 = x24044.log_pdf( y22543) if x24044.is_continuous else x24044.log_pmf( y22543) #obs, log likelihood
# p24047 = tf.add_n([p24042,p24046])
# # return E from the model
#
# sess = tf.Session()
# sess.run(x22542.initializer)
# sess.run(p24047)
# # printing E:
# print(sess.run(x22542))
# writer = tf.summary.FileWriter( './Graph_Output/g24048', sess.graph)
# sess.close()