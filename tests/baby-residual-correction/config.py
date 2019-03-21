from keras import initializers, optimizers
import tensorflow as tf
import random
seed = random.randint(1,1000)

class Config(object):
	lr = 0.001
	momentum = 0
	epoch = 3
	dropout = 0
	batch_size = 32
	#width, height, inchannels, outchannels
	#note: inchannels and outchannels are determined locally
	filter_size = [3, 3]
	conv_stride_size = [1, 1]
	deconv_stride_size = [1, 1]
	input_shape = [30, 30, 42]
	npy_filepath = "/Users/chenjinfan/desktop/data/npydata"
	tf_filepath = "/Users/chenjinfan/desktop/data/tfdata"
	init = tf.initializers.random_uniform()
	bias_init = tf.initializers.random_uniform()

	def get_filter(shape, name):
		init = tf.contrib.layers.xavier_initializer(seed = seed)
		filter = tf.get_variable(name = name + 'weights', regularizer = tf.contrib.layers.l2_regularizer(scale=1.0) ,shape = shape, initializer = init, trainable = True)
		return filter
		
	def get_zeros(shape):
		return  tf.Variable(tf.zeros(shape))
		
	def get_ones(shape):
		return  tf.Variable(tf.ones(shape))