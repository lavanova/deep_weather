#!/usr/bin/python
import tensorflow as tf
import numpy as np
class why:
	x = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
	rand_array = np.random.rand(32, 30, 30, 42)
	