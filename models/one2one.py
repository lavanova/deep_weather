#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np

#import deep500 as d5
#from deep500.frameworks import tensorflow as d5tf
#from deep500.frameworks.tensorflow.tf_graph_executor import TensorflowNativeGraphExecutor as tfge

seed = random.randint(1, 1000)

X_SHAPE = [30, 30, 42]

class One2One(object):
	def __init__(self, saver, sess, logger, dataloader, FLAGS):
		self.data = None
		self.labels = None
		self.x = None
		self.out = None
		self.y = None
		self.loss = None
		self.train_op = None

		self.saver = saver
		self.sess = sess
		self.logger = logger
		self.dataloader = dataloader
		self.FLAGS = FLAGS
		self.buildNet()

	def get_filter(self, shape, name):
		init = tf.contrib.layers.xavier_initializer(seed = seed)
		filter = tf.get_variable(name = name + 'weights', regularizer = tf.contrib.layers.l2_regularizer(scale=1.0) ,shape = shape, initializer = init, trainable = True)
		return filter

	def one2oneLayer(self, input, name):
		shape = X_SHAPE
		weights = tf.Variable(tf.ones(shape))
		bias = tf.Variable(tf.zeros(shape))
		return tf.multiply(input, weights) + bias

	def buildNet(self):
		with tf.variable_scope('one2one'):
			self.y = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
			self.x = tf.placeholder(tf.float32, shape=(None, 30, 30, 42))
			l1 = self.one2oneLayer(self.x, "l1")
			self.out = l1
			self.loss = tf.losses.mean_squared_error(self.y, self.out)
			self.train_op = tf.train.AdamOptimizer(learning_rate = self.FLAGS.lr).minimize(self.loss)

	def run(self, dataloader = None, train = None):
		if dataloader == None:
			dataloader = self.dataloader
		if train == None:
			train = self.FLAGS.train

		data_X, data_Y = dataloader.get_next()

		if train:
			init = tf.global_variables_initializer()
			self.sess.run(init)

			for i in range(self.FLAGS.epoch_num):
			    X, Y = self.sess.run([data_X, data_Y])
			    train_dict = {
			        self.x: X,
			        self.y: Y
			    }
			    _, loss = self.sess.run([self.train_op, self.loss], feed_dict = train_dict)
			    if i % self.FLAGS.log_interval == 0:
			        print("epoch " + str(i) + ":")
			        print("current loss is: " + str(loss))

		else:
			X, Y = self.sess.run([data_X, data_Y])
			test_dict = {
			    self.x: X,
			    self.y: Y
			}
			loss = self.sess.run(self.loss, feed_dict = test_dict)
			print("Testing mode: ")
			print("The test loss is: {:.5f}".format(loss))
