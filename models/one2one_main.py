import sys
sys.path.append('../')
import global_macros
from config import Config as cfg
import tensorflow as tf
import numpy as np
from one2one import One2One
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_integer('batch_size', 30, 'Size of inputs')
#flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
# flags.DEFINE_integer('data_workers', 4,
#     'Number of different data workers to load data in parallel')

# General Experiment Settings
flags.DEFINE_string('logdir', global_macros.CACHE_ROOT + "/one2one",
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
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


def TF2FLRD(filenames, batchsize=30, buffersize=730, oneshot=False):
    train_dataset = tf.data.TFRecordDataset(filenames=filenames)
    train_dataset = train_dataset.map(parse)
    train_dataset = train_dataset.shuffle(buffersize)
    train_dataset = train_dataset.batch(batchsize)
    train_dataset = train_dataset.repeat()
    if oneshot:
        return train_dataset.make_one_shot_iterator()
    else:
        return train_dataset.make_initializable_iterator()


def main():
    file_comment = "_small_grid"
    train_years = [2000, 2001]
    test_years = [2011]
    train_files = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in train_years]
    test_files = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in test_years]
    train_iter = TF2FLRD(train_files, batchsize=FLAGS.batch_size, buffersize=730)
    test_iter = TF2FLRD(test_files, batchsize=730, buffersize=730)

    with tf.Session() as sess:
        sess.run(train_iter.initializer)
        sess.run(test_iter.initializer)
        
        model = One2One(saver=None, sess=sess, logger=None, dataloader=train_iter, FLAGS=FLAGS)
        model.run()
        model.run(dataloader=test_iter, train=False)

if __name__ == "__main__":
    main()
