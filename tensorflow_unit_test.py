import tensorflow as tf
import numpy as np

#try to take derivatives with respect to the placeholder
graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder(dtype=tf.float32,shape=[None,10])
    b = tf.cos(a)
    
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    da = tf.gradients(b,[a])
    init = tf.global_variables_initializer()
    
    
with tf.Session(graph=graph) as sess:
    sess.run(init)
    feed_dict = {a:np.random.rand(10,10)}
    out_list  = [b, da]
    b_v,da_v = sess.run(out_list, feed_dict)
    print(da_v)
    