#!/usr/bin/env python3
# model
import numpy as np
import tensorflow as tf
import parameters


# TODO: not accounted yet: different strides for different dimensions, alright at the moment?
#      make sure loss is properly changed, output/input is correct


class UNetwork():
    #    def mult_chan_conv(self, tensor, filters, kernel = [3,3,3], stride = [1,1,1], data_format='channels_last',
    #           is_training = True):
    #        # Multichannel first step conv of concat data
    #        padding = 'valid'
    #        if self.should_pad: padding = 'same'
    #        #default channels_last corresponds to inputs with shape (batch, depth, height, width, channels)
    #        #for i in range(NR_CHANNELS): #iterate over channels

    #TODO: adapt kernel and stride, previously were 3,3,3 1,1,1 // 2,2,2 2,2,2
    def conv_batch_relu(self, tensor, filters, kernel=[3, 3, 3], stride=[1, 1, 1], is_training=True):
        # Produces the conv_batch_relu combination as in the paper
        padding = 'valid'
        if self.should_pad: padding = 'same'

        conv = tf.layers.conv3d(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")  # , data_format = "NCDHW"
        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
        conv = tf.nn.relu(conv)
        return conv

    def upconvolve(self, tensor, filters, kernel=[2, 2, 2], stride=[1, 2, 2], scale=4, activation=None):
        # Upconvolution - two different implementations: the first is as suggested in the original Unet paper and
        # the second is a more recent version
        # Needs to be determined if these do the same thing
        padding = 'valid'
        if self.should_pad: padding = 'same'
        # upsample_routine = tf.keras.layers.UpSampling3D(size = (scale,scale,scale)) # Uses tf.resize_images
        # tensor = upsample_routine(tensor)
        # conv = tf.layers.conv3d(tensor, filters, kernel, stride, padding = 'same',
        #                                 kernel_initializer = self.base_init, kernel_regularizer = self.reg_init)
        # use_bias = False is a tensorflow bug
        conv = tf.layers.conv3d_transpose(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                          use_bias=False,
                                          kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                          data_format="channels_first")
        return conv

    def centre_crop_and_concat(self, prev_conv, up_conv):
        # If concatenating two different sized Tensors, centre crop the first Tensor to the right size and concat
        # Needed if you don't have padding
        p_c_s = prev_conv.get_shape()
        u_c_s = up_conv.get_shape()
        offsets = np.array([0, 0, (p_c_s[2] - u_c_s[2]) // 2, (p_c_s[3] - u_c_s[3]) // 2,
                            (p_c_s[4] - u_c_s[4]) // 2], dtype=np.int32)
        size = np.array([-1, p_c_s[1], u_c_s[2], u_c_s[3], u_c_s[4]], np.int32)
        prev_conv_crop = tf.slice(prev_conv, offsets, size)
        up_concat = tf.concat((prev_conv_crop, up_conv), 1)
        return up_concat

    def __init__(self, base_filt=parameters.BASEFILTER_SIZE, in_depth=parameters.INPUT_DEPTH,
                 out_depth=parameters.OUTPUT_DEPTH, in_sizel=parameters.INPUT_SIZEL, in_sizew=parameters.INPUT_SIZEW,
                 out_sizel=parameters.OUTPUT_SIZEL, out_sizew=parameters.OUTPUT_SIZEW,
                 learning_rate=parameters.LEARNING_RATE, print_shapes=True, drop=parameters.DROPOUT_RATE, should_pad=True,
                 nr_chan=parameters.NR_CHANNELS, nr_concat=parameters.NR_CONCAT):#, bsize=parameters.BATCH_SIZE):
        # Initialise your model with the parameters defined above
        # Print-shape is a debug shape printer for convenience
        # Should_pad controls whether the model has padding or not
        # Base_filt controls the number of base conv filters the model has. Note deeper analysis paths have filters
        # that are scaled by this value
        # Drop specifies the proportion of dropped activations

        self.base_init = tf.truncated_normal_initializer(stddev=0.1)  # Initialise weights
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)  # Initialise regularisation (was useful)

        self.should_pad = should_pad  # To pad or not to pad, that is the question
        self.drop = drop  # Set dropout rate

        with tf.variable_scope('3DuNet'):
            self.training = tf.placeholder(tf.bool)
            self.do_print = print_shapes
            self.model_input = tf.placeholder(tf.float32, shape=(
            None, nr_chan, in_depth, in_sizew, in_sizel))  # batch none to input it later
            # Define placeholders for feed_dict #TODO: nr channels first or later?

            #self.model_cout = tf.placeholder(tf.float32, shape=(None, 1, out_depth, out_sizew, out_sizel))


            self.model_cout = tf.placeholder(tf.float32, shape=(None, 1, out_depth, out_sizew, out_sizel))


            if self.do_print:
                print('Input features shape', self.model_input.get_shape())
                print('Output shape', self.model_cout.get_shape())

            # Level zero
            self.inp = tf.slice(self.model_input, [0, 0, 0, 0, 0],[-1,(nr_chan-nr_concat),out_depth, out_sizew, out_sizel])
            #self.inp = self.model_input
            conv_0_1 = self.conv_batch_relu(self.inp, base_filt, is_training=self.training)#self.model_input
            conv_0_2 = self.conv_batch_relu(conv_0_1, base_filt * 2, is_training=self.training)
            # Level one
            max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2],
                                              data_format="channels_first")  # Stride, Kernel previously [2,2,2]
            # pool_size:An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width)

            conv_1_1 = self.conv_batch_relu(max_1_1, base_filt * 2, is_training=self.training)
            conv_1_2 = self.conv_batch_relu(conv_1_1, base_filt * 4, is_training=self.training)
            #conv_1_2 = tf.layers.dropout(conv_1_2, rate=self.drop, training=self.training)
            # Level two
            max_2_1 = tf.layers.max_pooling3d(conv_1_2, [1, 2, 2], [1, 2, 2],
                                              data_format="channels_first")  # Stride, Kernel previously [2,2,2]
            conv_2_1 = self.conv_batch_relu(max_2_1, base_filt * 4, is_training=self.training)
            conv_2_2 = self.conv_batch_relu(conv_2_1, base_filt * 8, is_training=self.training)
            #conv_2_2 = tf.layers.dropout(conv_2_2, rate=self.drop, training=self.training)
            # Level three
            max_3_1 = tf.layers.max_pooling3d(conv_2_2, [1, 2, 2], [1, 2, 2],
                                              data_format="channels_first")  # Stride, Kernel previously [2,2,2]
            conv_3_1 = self.conv_batch_relu(max_3_1, base_filt * 8, is_training=self.training)
            conv_3_2 = self.conv_batch_relu(conv_3_1, base_filt * 16, is_training=self.training)
            #conv_3_2 = tf.layers.dropout(conv_3_2, rate=self.drop, training=self.training)
            # Level two
            up_conv_3_2 = self.upconvolve(conv_3_2, base_filt * 16)#, kernel=2,
                                          #stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_2_1 = self.centre_crop_and_concat(conv_2_2, up_conv_3_2)
            conv_2_3 = self.conv_batch_relu(concat_2_1, base_filt * 8, is_training=self.training)
            conv_2_4 = self.conv_batch_relu(conv_2_3, base_filt * 8, is_training=self.training)
            #conv_2_4 = tf.layers.dropout(conv_2_4, rate=self.drop, training=self.training)
            # Level one
            up_conv_2_1 = self.upconvolve(conv_2_4, base_filt * 8)#, kernel=2,
                                          #stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_1_1 = self.centre_crop_and_concat(conv_1_2, up_conv_2_1)
            conv_1_3 = self.conv_batch_relu(concat_1_1, base_filt * 4, is_training=self.training)
            conv_1_4 = self.conv_batch_relu(conv_1_3, base_filt * 4, is_training=self.training)
            #conv_1_4 = tf.layers.dropout(conv_1_4, rate=self.drop, training=self.training)
            # Level zero
            up_conv_1_0 = self.upconvolve(conv_1_4, base_filt * 4)#, kernel=2,
                                          #stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_0_1 = self.centre_crop_and_concat(conv_0_2, up_conv_1_0)
            conv_0_3 = self.conv_batch_relu(concat_0_1, base_filt * 2, is_training=self.training)
            conv_0_4 = self.conv_batch_relu(conv_0_3, base_filt * 2, is_training=self.training)
            # TODO: layer to add mean directly to end

            #conv_0_4 = tf.layers.dropout(conv_0_4, rate=self.drop, training=self.training)


            #single traj mean
            #concat_0_5 = tf.concat((tf.slice(self.model_input,[0, 8, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]), conv_0_4), 1)
            #conv_0_5 = tf.layers.conv3d(concat_0_5, 1, [1, 1, 1], [1, 1, 1], padding='same',
            #                            data_format="channels_first")  # 1 instead of OUTPUT_CLASSES


            #single traj spread
            conv_0_5 = tf.layers.conv3d(conv_0_4, 1, [1, 1, 1], [1, 1, 1], padding='same',
                                        data_format="channels_first")  # 1 instead of OUTPUT_CLASSES
            self.predictions = conv_0_5







            #self.predictions, vari = tf.nn.moments(conv_0_5, [1], keep_dims=True)
            #self.spread = tf.sqrt(vari)




            # End of singular trajectory hybrid model



            #Start of hybrid model

            #concat_0_5 = tf.concat((tf.slice(self.model_input,[0, (nr_chan-nr_concat), 0, 0, 0],[-1,nr_concat,out_depth,out_sizew,out_sizel]), conv_0_4), 1)
            #conv_0_5 = tf.layers.conv3d(concat_0_5, 10, [1, 1, 1], [1, 1, 1], padding='same',
            #                                 data_format="channels_first")  # 1 instead of OUTPUT_CLASSES
            #self.predictions, vari = tf.nn.moments(conv_0_5, [1], keep_dims=True)
            #self.spread = tf.sqrt(vari)

            #end of hybrid model







            #conv_0_5 = tf.layers.conv3d(conv_0_4, 5, [1, 1, 1], [1, 1, 1], padding='same',
            #                                         data_format="channels_first")  # 1 instead of OUTPUT_CLASSES

            #18ch concat

            #stddev calc
            #self.mean, vari =  tf.nn.moments(concat_0_5, [1], keep_dims=True)

            #self.predictions = self.spread

            #self.predictions = tf.layers.conv3d(concat_0_5, 1, [1, 1, 1], [1, 1, 1], padding='same',
            #                                    data_format="channels_first")  # 1 instead of OUTPUT_CLASSES






            #flat = tf.layers.Flatten()(concat_0_5)
            #dense_0_5 = tf.layers.dense(flat, 38080)
            #dense_0_5 = tf.layers.dropout(dense_0_5, rate=0.2, training=self.training)
            #self.predictions = tf.reshape(dense_0_5, [-1,1,7,40,136])







            #TODO: BATCH NORMALIZATION!!!! not directly conv3d
            #13channel concat
            #concat_0_5 = tf.concat((tf.slice(self.model_input,[0, 8, 0, 0, 0],[-1,5,out_depth,out_sizew,out_sizel]), conv_0_4), 1)
            #self.predictions = tf.layers.conv3d(concat_0_5, 1, [1, 1, 1], [1, 1, 1], padding='same',
            #                                    data_format="channels_first")  # 1 instead of OUTPUT_CLASSES





            # conv_out
            #self.predictions = tf.layers.conv3d(conv_0_4, 1, [1, 1, 1], [1, 1, 1], padding='same',
            #                                    data_format="channels_first")  # 1 instead of OUTPUT_CLASSES





            # self.predictions = tf.expand_dims(tf.argmax(conv_out, axis = -1), -1)


            if self.do_print:
                print('Model output shape', self.predictions.get_shape())
                # print('Model Argmax output shape', self.predictions.get_shape())

            self.linf_loss = tf.reduce_max(
                tf.abs(tf.subtract(self.predictions, self.model_cout)))  # faster calculation than RMSE


            #self.linf_loss = tf.reduce_max(
            #    tf.abs(tf.subtract(self.spread, tf.slice(self.model_cout,[0, 1, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))))  # faster calculation than RMSE



            #difer = tf.subtract(tf.slice(self.model_cout,[0, 0, 0, 0, 0],[1,1,out_depth,out_sizew,out_sizel]), tf.reduce_mean(tf.slice(self.model_input,[0, 8, 0, 0, 0],[1,5,out_depth,out_sizew,out_sizel]),1))
            #i = tf.constant(1)
            #c = lambda difer, i: tf.less(i, bsize)
            #b = lambda difer, i: [tf.concat([difer, tf.subtract(tf.slice(self.model_cout,[i, 0, 0, 0, 0],[1,1,out_depth,out_sizew,out_sizel]), tf.reduce_mean(tf.slice(self.model_input,[i, 8, 0, 0, 0],[1,5,out_depth,out_sizew,out_sizel]),1))],0), tf.add(i, 1)]
            #tsr, tsi = tf.while_loop(c, b, [difer, i], shape_invariants=[tf.TensorShape([None, 1, out_depth, out_sizew, out_sizel]), i.get_shape()])


            #difer = tf.subtract(self.model_cout, tf.reduce_mean(tf.slice(self.model_input,[0, 8, 0, 0, 0],[-1,5,out_depth,out_sizew,out_sizel]),1))
            #self.loss = tf.losses.mean_squared_error(difer, self.predictions)
            #self.linf_loss = tf.reduce_max(self.predictions)  # faster calculation than RMSE


            #meanloss = tf.losses.mean_squared_error(self.model_cout, self.predictions)


            #self.lossm = tf.losses.mean_squared_error(tf.slice(self.model_cout,[0, 0, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]), self.mean)
            #self.losss = tf.losses.mean_squared_error(tf.slice(self.model_cout,[0, 1, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]), self.spread)


            #mout = tf.slice(self.model_cout,[0, 0, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel])
            #spout = tf.slice(self.model_cout,[0, 1, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel])

            #loss_0 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 0, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_1 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 1, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_2 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 2, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_3 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 3, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_4 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 4, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_5 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 5, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_6 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 6, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_7 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 7, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_8 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 8, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #loss_9 = tf.losses.mean_squared_error(mout, tf.slice(conv_0_5,[0, 9, 0, 0, 0],[-1,1,out_depth,out_sizew,out_sizel]))
            #self.loss = (loss_0+loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8+loss_9)/10
            #self.loss = tf.losses.mean_squared_error(spout, self.spread)


                # TODO: Not sure about sideffects of tf.losses.absolute_difference here
            #self.loss = self.lossm + self.losss#tf.losses.mean_squared_error(self.model_cout, self.predictions)

            self.loss = tf.losses.mean_squared_error(self.model_cout, self.predictions)
            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            #self.sumloss = tf.summary.histogram('training_loss', self.loss)
            #self.valloss = tf.summary.histogram('validation_loss', self.loss)
            #if self.training == True:
            #    tf.summary.scalar('training_loss', self.loss)
            #else:
            #    tf.summary.scalar('validation_loss', self.loss)

            self.extra_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            with tf.control_dependencies(self.extra_update_ops):
                self.train_op = self.trainer.minimize(
                    self.loss)  # = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
