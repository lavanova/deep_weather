from config import Config as cfg
import tensorflow as tf
import numpy as np
from model import One2One
def parse_one_string_record(string_record):
	example = tf.train.Example()
	example.ParseFromString(string_record)
	X = example.features.feature['X'].float_list.value
	Y = example.features.feature['Y'].float_list.value
	x = np.empty(75600)
	y = np.empty(37800)
	for i in range(37800):
		x[i*2] = X[i*2]
		x[i*2+1] = X[i*2+1]
		y[i] = Y[i]
	x, y = x.reshape(2, 6, 7, 30, 30), y.reshape(1, 6, 7, 30, 30)
	return x, y

	
def trainForOneYear(i, sess, model):
	print("traning for year "+str(j+2000))
	reader = tf.TFRecordReader()
	tfrecords_filename = cfg.tf_filepath+"/tf_"+str(2000+i)+"_small_grid"
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	
	counter = 0
	batchBuffery3 = np.empty(shape = (cfg.batch_size, 42, 30, 30))
	batchBufferx3 = np.empty(shape = (cfg.batch_size, 42, 30, 30))
		
	for string_record in record_iterator:
		x, y = parse_one_string_record(string_record)
		y = y.reshape(42, 30, 30)
		x = x[1].reshape(42, 30, 30)
		
		batchBuffery3[counter%cfg.batch_size] = y
		batchBufferx3[counter%cfg.batch_size] = x
		
		if counter == cfg.batch_size:
			counter = 0
			feedx = batchBufferx3.transpose(0, 2, 3, 1)
			feedy = batchBuffery3.transpose(0, 2, 3, 1)
			_, loss = sess.run([model.opt, model.loss], feed_dict ={model.ip: feedx, model.y: feedy}) 
			print("current loss is " + str(loss))
		
		counter += 1

def testOnOneYear(i, sess, model):
	print("start testing on year "+str(2000+i))
	reader = tf.TFRecordReader()
	tfrecords_filename = cfg.tf_filepath+"/tf_"+str(2000+i)+"_small_grid"
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
		
	counter = 0
	total_loss = 0
	num_batches = 0
	batchBuffery3 = np.empty(shape = (cfg.batch_size, 42, 30, 30))
	batchBufferx3 = np.empty(shape = (cfg.batch_size, 42, 30, 30))
			
	for string_record in record_iterator:
		x, y = parse_one_string_record(string_record)
		y = y.reshape(42, 30, 30)
		x = x[1].reshape(42, 30, 30)
			
		batchBuffery3[counter%cfg.batch_size] = y
		batchBufferx3[counter%cfg.batch_size] = x
			
		if counter == cfg.batch_size:
			counter = 0
			num_batches +=1
			feedx = batchBufferx3.transpose(0, 2, 3, 1)
			feedy = batchBuffery3.transpose(0, 2, 3, 1)
			loss = sess.run(model.loss, feed_dict ={model.ip: feedx, model.y:
				feedy}) 
			total_loss += loss
			
		counter += 1
		
	return total_loss/num_batches

def main():	
	with tf.Session() as sess:
		model = One2One()
		sess.run(tf.global_variables_initializer())	
		
		test_error_before_training = testOnOneYear(11, sess, model)
		for i in range(cfg.epoch):
				print("epoch " + str(i))
				for j in range(0, 10):
					trainForOneYear(j, sess, model)	
				
		print("Training finished")
		test_error_after_training = testOnOneYear(11, sess, model)
		improvement = test_error_before_training - test_error_after_training
		print("The error before training is " +str(test_error_before_training))
		print("The error after training is " +str(test_error_after_training))
		print("We got an improvement of "+str(improvement))
		print("which is "+str(100*improvement/test_error_before_training)+"%")


main()