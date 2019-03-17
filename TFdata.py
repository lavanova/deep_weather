import tensorflow as tf

def parse(example):
    shapeX = (12,7,30,30)
    shapeY = (6,7,30,30)
    features = {
            'X': tf.FixedLenFeature(shapeX, tf.float32), #FixedLenSequenceFeature, allow_missing=True
            'Y': tf.FixedLenFeature(shapeY, tf.float32)
        }
    data = tf.parse_single_example(example, features)
    return data['X'], data['Y']

def TF2FLRD(filenames, buffersize=730, batchsize=30): # 730 to shuffle a year at least/ 4000 too much
    train_dataset = tf.data.TFRecordDataset(filenames=filenames)
    train_dataset = train_dataset.map(parse)
    train_dataset = train_dataset.shuffle(buffersize)
    train_dataset = train_dataset.batch(batchsize)
    #return train_dataset.make_initializable_iterator()
    return train_dataset.make_one_shot_iterator()
