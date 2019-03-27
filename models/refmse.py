import tensorflow as tf
import TFdata
from utils import *


def n_trainable_parameters():
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))


class mse_ref(base_model):
    def __init__(self, sess, FLAGS):
        base_model.__init__(self, sess, FLAGS)

    def _buildnet(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 6, 7, 30, 30), name='X')
        self.y = tf.placeholder(tf.float32, shape=(None, 6, 7, 30, 30), name='Y')
        with tf.name_scope('MSE_loss'):
            self.loss = tf.losses.mean_squared_error(self.y, self.x)
            with tf.name_scope('summary'):
                tf.summary.scalar('MSE loss', self.loss)
        with tf.name_scope('train'):
            self.global_step = tf.Variable(1, name='global_step', trainable=False)
            self.train_op = self.loss
            #tf.train.AdamOptimizer(learning_rate = self.FLAGS.lr).minimize(self.loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()

if __name__ == "__main__":
    identity_baseline()
    #linear_baseline()
