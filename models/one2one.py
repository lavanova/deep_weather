#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from utils import variable_summaries

seed = random.randint(1, 1000)

X_SHAPE = [30, 30, 42]

class One2One(object):
    def __init__(self, sess, FLAGS):
        self.data = None
        self.labels = None
        self.x = None
        self.out = None
        self.y = None
        self.loss = None
        self.global_step = None
        self.train_op = None
        self.summary_op = None

        self.latest_model = None
        self.saver = None
        self.sess = sess
        self.logger = None
        self.FLAGS = FLAGS
        self._buildNet()

    def get_filter(self, shape, name):
        init = tf.contrib.layers.xavier_initializer(seed = seed)
        filter = tf.get_variable(name = name + 'weights', regularizer = tf.contrib.layers.l2_regularizer(scale=1.0) ,shape = shape, initializer = init, trainable = True)
        return filter

    def one2oneLayer(self, input, namespace):
        with tf.variable_scope(namespace):
            shape = X_SHAPE
            weights = tf.Variable(tf.ones(shape), name = 'weights')
            variable_summaries(weights)
            assert weights.graph is tf.get_default_graph()
            bias = tf.Variable(tf.zeros(shape), name = 'bias')
            variable_summaries(bias)
        return tf.multiply(input, weights) + bias

    def _buildNet(self):
        with tf.variable_scope('one2one'):
            self.y = tf.placeholder(tf.float32, shape=(None, 30, 30, 42), name='Y')
            self.x = tf.placeholder(tf.float32, shape=(None, 30, 30, 42), name='X')
            l1 = self.one2oneLayer(self.x, "l1")
            self.out = l1
            self.loss = tf.losses.mean_squared_error(self.y, self.out)
            tf.summary.scalar('MSE loss', self.loss)
            self.global_step = tf.Variable(1, name='global_step', trainable=False)
            self.train_op = tf.train.AdamOptimizer(learning_rate = self.FLAGS.lr).minimize(self.loss, global_step=self.global_step)
            self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=6)
        self.logger = tf.summary.FileWriter( osp.join(self.FLAGS.logdir, self.FLAGS.exp), self.sess.graph)

    def _get_loss(self, iter_X, iter_Y):
        X, Y = self.sess.run([iter_X, iter_Y])
        dict = {
            self.x: X,
            self.y: Y
        }
        return self.sess.run(self.loss, feed_dict = dict)


    def run(self, iter_data, iter_val = None, train = None, load_path = None):
        if train == None:
            train = self.FLAGS.train
        if load_path == True:
            self.saver.restore(self.sess, self.latest_model)
        elif load_path != None:
            self.saver.restore(self.sess, load_path)

        data_X, data_Y = iter_data.get_next()
        if iter_val != None:
            val_X, val_Y = iter_val.get_next()


        if train:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            for i in range(self.FLAGS.epoch_num):
                step = tf.train.global_step(self.sess, self.global_step)
                X, Y = self.sess.run([data_X, data_Y])
                train_dict = {
                    self.x: X,
                    self.y: Y
                }
                summary, _, loss = self.sess.run([self.summary_op, self.train_op, self.loss], feed_dict = train_dict)
                if i % self.FLAGS.save_interval == 0:
                    self.latest_model = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(i))
                    self.saver.save(self.sess, self.latest_model)
                if i % self.FLAGS.log_interval == 0:
                    self.logger.add_summary(summary, step)
                    print("epoch " + str(i) + ":")
                    print("current training loss is: {:.6f}".format(loss))
                    if iter_val != None:
                        loss = self._get_loss(val_X, val_Y)
                        print("current validation loss is: {:.6f}".format(loss))

        else:
            loss = self._get_loss(data_X, data_Y)
            print("Testing mode: ")
            print("The test loss is: {:.6f}".format(loss))
