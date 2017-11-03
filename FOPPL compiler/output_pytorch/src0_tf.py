import tensorflow as tf
from tensorflow.contrib.distributions import *
c24039= tf.constant(1.0)
c24040= tf.constant(2.0)
x24041 = Normal(mu=c24039, sigma=c24040)
x22542 = tf.Variable( 2.0)
# x22542 = tf.Variable( x24041.sample())   #sample
p24042 = x24041.log_pdf( x22542) if x24041.is_continuous else x24041.log_pmf( x22542)   #prior
c24043= tf.constant(3.0)
x24044 = Normal(mu=x22542, sigma=c24043)
c24045= tf.constant(7.0)
y22543 = c24045
p24046 = x24044.log_pdf( y22543) if x24044.is_continuous else x24044.log_pmf( y22543) #obs, log likelihood
p24047 = tf.add_n([p24042,p24046])
# return E from the model

x_g1 = tf.gradients(p24042, x22542)
x_g2 = tf.gradients(p24046, x22542)
x_g = tf.gradients(p24047, x22542)
sess = tf.Session()
sess.run(x22542.initializer)

x_, log_pdf, x_g1, x_g2, x_g = sess.run([x22542,p24047,x_g1, x_g2, x_g])
print(x_, log_pdf, x_g1, x_g2, x_g)
writer = tf.summary.FileWriter( './Graph_Output/g24048', sess.graph)
sess.close()