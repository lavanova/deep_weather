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

'''
Add summaries, by default only adds histogram,
if verbose=1, add mean, stddev, min, max
The default namespace is summary
'''
def variable_summaries(var, name, verbose=0):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        if verbose:
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def print_flag(FLAGS):
    for key in FLAGS.flag_values_dict():
        print("{:<22}: {}".format(key.upper(), FLAGS[key].value))

'''
Get the number of trainable parameters in the current graph
'''
def get_n_trainable_parameters():
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))


class base_model():
    def __init__(self, sess, FLAGS):
        self.x = None             # input placeholder
        self.y = None             # output placeholder
        self.loss = None          # loss operator
        self.global_step = None   # global step counter
        self.train_op = None      # train operator

        self.summary_op = None    # summary operator
        self.latest_model = None  # string of the path of the latest model
        self.sess = sess          # session object
        self.FLAGS = FLAGS        # flag object
        self._buildnet()

        # define saver and logger objects
        self.saver = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=6)
        self.logger = tf.summary.FileWriter( osp.join(self.FLAGS.logdir, self.FLAGS.exp), self.sess.graph)
        self._check_graph()

    '''
    check if the all the mandatory operators are defined
    '''
    def _check_graph(self):
        assert(self.x != None)
        assert(self.y != None)
        assert(self.loss != None)
        assert(self.global_step != None)
        assert(self.train_op != None)

    '''
    building the network in the graph
    called inside of the class constructor
    strongly virtual function, requires child class implementation
    ALL graph definition must be inside this function
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
    train (bool) : for training or test
    load (string or bool) : to load the ckpt file, if True, then load the latest model recorded
    '''
    def run(self, iter_data, iter_val = None, train = None, load = False):
        if train == None:
            train = self.FLAGS.train
        if load == True:
            self.saver.restore(self.sess, self.latest_model)
        elif load != False:
            self.saver.restore(self.sess, load)

        data_X, data_Y = iter_data.get_next()
        if iter_val != None:
            val_X, val_Y = iter_val.get_next()

        if train:
            if (self.FLAGS.resume_iter != -1):
                print("\nResume Training mode: ")
                model_file = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(self.FLAGS.resume_iter))
                self.saver.restore(self.sess, model_file)
            else:
                print("\nInitiating Training mode: ")
                self.sess.run(tf.global_variables_initializer())
            for i in range(self.FLAGS.epoch_num):
                step = tf.train.global_step(self.sess, self.global_step)
                # get data
                X, Y = self.sess.run([data_X, data_Y])
                train_dict = {
                    self.x: X,
                    self.y: Y
                }
                # calculate loss
                if(self.summary_op == None):
                    _, loss = self.sess.run([self.train_op, self.loss], feed_dict = train_dict)
                else:
                    summary, _, loss = self.sess.run([self.summary_op, self.train_op, self.loss], feed_dict = train_dict)
                # save model
                if step % self.FLAGS.save_interval == 0:
                    self.latest_model = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(step))
                    self.saver.save(self.sess, self.latest_model)
                # logging
                if step % self.FLAGS.log_interval == 0:
                    print("epoch " + str(step) + ":")
                    print("Training loss is: {:.6f}".format(loss))
                    if(self.summary_op != None):
                        self.logger.add_summary(summary, step)
                    if iter_val != None:
                        loss = self._get_loss(val_X, val_Y)
                        print("Validation loss is: {:.6f}".format(loss))

        else:
            print("\nInitiating Testing mode: ")
            loss = self._get_loss(data_X, data_Y)
            print("The test loss is: {:.6f}".format(loss))
