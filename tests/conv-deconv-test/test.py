#!/usr/bin/python
import tensorflow as tf
import numpy as np
from test2 import why
x = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))

#with tf.Session() as sess:

#	rand_array = np.random.rand(32, 30, 30, 42)
#	print(sess.run(x, feed_dict={x: why.rand_array}))
for i, j in range(2, 3):
	print(i)