import sys
sys.path.append('../')
import os.path as osp
import global_macros
from config import Config as cfg
import tensorflow as tf
import numpy as np
from one2one import One2One
from tensorflow.python.platform import flags
from utils import TF2FLRD

FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_integer('batch_size', 64, 'Size of a batch')
#flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
# flags.DEFINE_integer('data_workers', 4,
#     'Number of different data workers to load data in parallel')

# General Experiment Settings
flags.DEFINE_string('ckptdir', global_macros.CKPT_ROOT + "/one2one",
    'location where models will be stored')
flags.DEFINE_string('logdir', global_macros.LOGGER_ROOT + "/one2one",
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'test', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 200,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 500, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 1e-3, 'Learning for training')
# flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')

# EBM Specific Experiments Settings
# flags.DEFINE_float('ml_coeff', 1.0, 'Maximum Likelihood Coefficients')
# flags.DEFINE_float('l2_coeff', 1.0, 'L2 Penalty training')
# flags.DEFINE_bool('cclass', False, 'Whether to conditional training in models')


def parse(example):
    shapeX = (12*7,30,30)
    shapeY = (6*7,30,30)
    features = {
            'X': tf.FixedLenFeature(shapeX, tf.float32), #FixedLenSequenceFeature, allow_missing=True
            'Y': tf.FixedLenFeature(shapeY, tf.float32)
        }
    data = tf.parse_single_example(example, features)

    data['X'] = tf.transpose(data['X'][6*7:12*7], perm=[1,2,0])
    data['Y'] = tf.transpose(data['Y'], perm=[1,2,0])
    return data['X'], data['Y']


def main():
    file_comment = "_small_grid"
    years_train = [2000, 2001, 2002]
    years_val = [2010]
    years_test = [2011]

    ipaths_train = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in years_train]
    ipaths_val = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in years_val]
    ipaths_test = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in years_test]

    iter_train = TF2FLRD(ipaths_train, batchsize=FLAGS.batch_size, buffersize=730, parse=parse)
    iter_val = TF2FLRD(ipaths_val, batchsize=730, buffersize=730, parse=parse)
    iter_test = TF2FLRD(ipaths_test, batchsize=730, buffersize=730, parse=parse)

    with tf.Session() as sess:
        sess.run(iter_train.initializer)
        sess.run(iter_val.initializer)
        sess.run(iter_test.initializer)

        train_writer = tf.summary.FileWriter( osp.join(FLAGS.logdir, FLAGS.exp), sess.graph)

        model = One2One(sess=sess, logger=train_writer, FLAGS=FLAGS)
        model.run(iter_data=iter_train, iter_val=iter_val)
        model.run(iter_data=iter_test, train=False, load_path=True)

if __name__ == "__main__":
    main()
