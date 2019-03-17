import tensorflow as tf
import TFdata

def identity_loss():
    shape = (None,6,7,30,30)
    X = tf.placeholder(tf.float32, shape)
    Y = tf.placeholder(tf.float32, shape)
    pred = X
    loss = tf.losses.mean_squared_error( labels = Y, predictions = pred)


def identity_baseline():
    #identity_loss()
    shape = (None,6,7,30,30)
    X = tf.placeholder(tf.float32, shape)
    Y = tf.placeholder(tf.float32, shape)
    pred = X
    loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    filenames = "./tfdata/tf_2000_small_grid"
    iter = TFdata.TF2FLRD(filenames, batchsize=730)
    with tf.Session() as sess:
        xdata, ydata = sess.run(iter.get_next())
        xdata0 = xdata[:,0:6,:]
        xdata1 = xdata[:,6:12,:]
        print(xdata.shape)
        result = sess.run(loss, feed_dict={X:xdata0 , Y:ydata})
        print("The mse wrt previous time is: {:.5f}".format(result) )
        result = sess.run(loss, feed_dict={X:xdata1 , Y:ydata})
        print("The mse wrt forcast is: {:.5f}".format(result) )


def linear_baseline():
    shape = (None,6,7,30,30)
    Nunits = 6*7*30*30
    learning_rate = 0.0001
    nepochs = 1000
    display_step = 1
    # X0 = tf.placeholder(tf.float32, shape)
    # X1 = tf.placeholder(tf.float32, shape)
    # Y = tf.placeholder(tf.float32, shape)
    X0 = tf.placeholder(tf.float32, (None, Nunits) )
    X1 = tf.placeholder(tf.float32, (None, Nunits) )
    Y = tf.placeholder(tf.float32, (None, Nunits) )
    H1 = tf.layers.dense(
        inputs=X0,
        units=Nunits,
        use_bias=True,
        name="dense_layer"
    )
    pred = X1 + H1
    loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimize_op = optimizer.minimize(loss)
    #Tfiles = ["./tfdata/tf_2000_small_grid",  "./tfdata/tf_2001_small_grid"]
    Tfiles = "./tfdata/tf_2000_small_grid"
    Vfiles = "./tfdata/tf_2005_small_grid"
    Titer = TFdata.TF2FLRD(Tfiles, batchsize=100)
    Viter = TFdata.TF2FLRD(Vfiles, batchsize=730)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(nepochs):
            xdata, ydata = sess.run(Titer.get_next())
            ydata = ydata.flatten()
            xdata0 = xdata[:,0:6,:].flatten()
            xdata1 = xdata[:,6:12,:].flatten()
            training_loss, _ = sess.run([loss, optimize_op], feed_dict={X0:xdata0, X1:xdata1, Y:ydata})
            if(i%display_step):
                print("The training loss is: {:.5f}".format(training_loss))
        xdata, ydata = sess.run(Viter.get_next())
        xdata0 = xdata[:,0:6,:]
        xdata1 = xdata[:,6:12,:]
        val_loss = sess.run(loss, feed_dict={X0:xdata0, X1:xdata1, Y:ydata})
        print("The validation loss is: {:.5f}".format(val_loss))

if __name__ == "__main__":
    identity_baseline()
    #linear_baseline()
