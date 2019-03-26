#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from utils import variable_summaries
from utils import base_model

seed = random.randint(1, 1000)

# define model parameter as macros here:
X_SHAPE = [30, 30, 42]


class One2One(base_model):
    def __init__(self, sess, FLAGS):
        base_model.__init__(self, sess, FLAGS)

    def one2oneLayer(self, input, namespace):
        with tf.name_scope(namespace):
            shape = X_SHAPE
            weights = tf.Variable(tf.ones(shape), name = 'weights')
            bias = tf.Variable(tf.zeros(shape), name = 'bias')
            out = tf.multiply(input, weights) + bias
            # out = tf.identity(out, name="out")
            with tf.name_scope('summary'):
                variable_summaries(weights, name = 'weights')
                variable_summaries(bias, name = 'bias')
        return out

    def _buildnet(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 30, 30, 42), name='X')
        self.y = tf.placeholder(tf.float32, shape=(None, 30, 30, 42), name='Y')
        with tf.name_scope('one2one'):
            l1 = self.one2oneLayer(self.x, "l1")
            self.loss = tf.losses.mean_squared_error(self.y, l1)
            with tf.name_scope('summary'):
                tf.summary.scalar('MSE loss', self.loss)
        with tf.name_scope('train'):
            self.global_step = tf.Variable(1, name='global_step', trainable=False)
            self.train_op = tf.train.AdamOptimizer(learning_rate = self.FLAGS.lr).minimize(self.loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()
