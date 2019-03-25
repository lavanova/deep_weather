import tensorflow as tf
import numpy as np
import os.path as osp

SHAPEX = (12,7,30,30)
SHAPEY = (6,7,30,30)

def _parse_(example):
    features = {
            'X': tf.FixedLenFeature(SHAPEX, tf.float32), #FixedLenSequenceFeature, allow_missing=True
            'Y': tf.FixedLenFeature(SHAPEY, tf.float32)
        }
    data = tf.parse_single_example(example, features)
    return data['X'], data['Y']


def TF2FLRD(filenames, batchsize=30, buffersize=730, parse=_parse_, oneshot=False):
    train_dataset = tf.data.TFRecordDataset(filenames=filenames)
    train_dataset = train_dataset.map(parse)
    train_dataset = train_dataset.shuffle(buffersize)
    train_dataset = train_dataset.batch(batchsize)
    train_dataset = train_dataset.repeat()
    if oneshot:
        return train_dataset.make_one_shot_iterator()
    else:
        return train_dataset.make_initializable_iterator()

def variable_summaries(var, name=''):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

class base_model():
    def __init__(self, sess, FLAGS):
        self.x = None
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
        self._buildnet()

    '''
    building the network in the graph
    called inside of the class constructor
    strongly virtual function, requires child class implementation
    '''
    def _buildnet(self):
        raise NotImplementedError()

    '''
    private function for calculating loss giving iterators
    '''
    def _get_loss(self, iter_X, iter_Y):
        X, Y = self.sess.run([iter_X, iter_Y])
        dict = {
            self.x: X,
            self.y: Y
        }
        return self.sess.run(self.loss, feed_dict = dict)


    '''
    run function serves as both training and testing
    controlled by FLAG.train boolean by default, but changeable by passing boolean into train argument
    load_path : path to load the ckpt file, if True, then load the latest model recorded
    '''
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
            print("Initiating Training mode: ")
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
                    print("Training loss is: {:.6f}".format(loss))
                    if iter_val != None:
                        loss = self._get_loss(val_X, val_Y)
                        print("Validation loss is: {:.6f}".format(loss))

        else:
            loss = self._get_loss(data_X, data_Y)
            print("Initiating Testing mode: ")
            print("The test loss is: {:.6f}".format(loss))
