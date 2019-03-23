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
	
def one2oneLayer(input, name):
	weights = cfg.get_ones(cfg.input_shape)
	bias = cfg.get_zeros(cfg.input_shape)
	return tf.multiply(input, weights) + bias
	
	
# use x0 and y3 to predict x3
class CorrNet(object):
	def __init__(self):
		self.data = None
		self.labels = None
		self.x0 = None
		self.x3 = None
		self.y3 = None
		self.residual = None
		self.loss = None
		self.opt = None
		self.buildNet()
	
	def buildNet(self):
		self.x0 = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
		self.x3 = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
		self.y3 = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
		l0 = tf.multiply(self.x0, cfg.get_zeros(cfg.input_shape)) + cfg.get_zeros(cfg.input_shape)
		l1 = convLayer(l0, 'conv1', 64)
		l2 = convLayer(l1, 'conv2', 128, stride = [2, 2])
		l3 = deconvLayer(l2, 'deconv1', 64, stride = [2, 2])
		l4 = tf.stack([l1, l3], axis = 3)
		self.residual = deconvLayer(l3, 'deconv1', 42)
		out = one2oneLayer(self.y3, "y3one2one") + self.residual
		self.loss = tf.losses.mean_squared_error(self.x3, out)
		self.opt = tf.train.AdamOptimizer(learning_rate = cfg.lr).minimize(self.loss)
		pass
	
