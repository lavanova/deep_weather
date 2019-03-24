import tensorflow as tf
import numpy as np

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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
