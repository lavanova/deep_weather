#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
from config import Config as cfg

import deep500 as d5
from deep500.frameworks import tensorflow as d5tf
#from deep500.frameworks.tensorflow.tf_graph_executor import TensorflowNativeGraphExecutor as tfge

seed = random.randint(1, 1000)


def denseLayer(shape, inputs, reuse = False, name = None):
	l = tf.layers.dense(inputs = inputs, units = shape, kernel_initializer = cfg.init, bias_initializer = cfg.bias_init, reuse = reuse, name = name)
	out = tf.nn.leaky_relu(features = l)
	return tf.nn.dropout(out, keep_prob = 1-cfg.dropout , seed = seed)

#input has shape[batch, in_height, in_width, in_channels]
def convLayer(inputs, name, outchannels, stride = cfg.conv_stride_size):
	return tf.contrib.layers.conv2d(inputs, outchannels, cfg.filter_size, stride, padding = 'SAME')
	
def deconvLayer(inputs, name, outchannels, stride = cfg.deconv_stride_size):
	return tf.contrib.layers.conv2d_transpose(inputs, outchannels, cfg.filter_size, stride, padding = 'SAME')
	
	
	
class CNN(object):
	def __init__(self):
		self.data = None
		self.labels = None
		self.ip = None
		self.out = None
		self.y = None
		self.loss = None
		self.opt = None
		self.buildNet()
	
	def buildNet(self):
		#use a batch-size of 5
		self.y = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
		self.ip = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
		l1 = convLayer(self.ip, 'conv1', 32)
		l2 = convLayer(l1, 'conv2', 8, stride = [2, 2])
		print("the dimension is reduced to "+str(l2.shape))
		l3 = deconvLayer(l2, 'deconv1', 32, stride = [2, 2])
		self.out= deconvLayer(l3, 'deconv1', 42)
		self.loss = tf.losses.mean_squared_error(self.y, self.out)
		self.opt = tf.train.AdamOptimizer(learning_rate = cfg.lr).minimize(self.loss)
		pass
	
