from config import Config as cfg
import tensorflow as tf
import numpy as np
from model import CNN
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
	reader = tf.TFRecordReader()
	tfrecords_filename = cfg.tf_filepath+"/tf_"+str(2000+i)+"_small_grid"
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	
	counter = 0
	batchBuffer = np.empty(shape = (cfg.batch_size, 42, 30, 30))
		
	for string_record in record_iterator:
		_, y = parse_one_string_record(string_record)
		y = y.reshape(42, 30, 30)
		batchBuffer[counter%cfg.batch_size] = y
#		batchBuffer.transpose(1, 2, 3, 0)
		if counter == cfg.batch_size:
			counter = 0
			feed = batchBuffer.transpose(0, 2, 3, 1)
			_, loss = sess.run([model.opt, model.loss], feed_dict ={model.ip: feed, model.y: feed}) 
			print("current loss is " + str(loss))
		
		counter += 1
		
		
def testForOneYear(i, sess, model):
	reader = tf.TFRecordReader()
	tfrecords_filename = cfg.tf_filepath+"/tf_"+str(2000+i)+"_small_grid"
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	
	counter = 0
	batchBuffer = np.empty(shape = (cfg.batch_size, 42, 30, 30))
	total_loss = 0
	num_batches = 0
	for string_record in record_iterator:
		_, y = parse_one_string_record(string_record)
		y = y.reshape(42, 30, 30)
		batchBuffer[counter%cfg.batch_size] = y
#		batchBuffer.transpose(1, 2, 3, 0)
		if counter == cfg.batch_size:
			num_batches += 1
			counter = 0
			feed = batchBuffer.transpose(0, 2, 3, 1)
			loss = sess.run( model.loss, feed_dict ={model.ip: feed, model.y: feed}) 
			total_loss += loss
			print("current loss is " + str(loss))
		
		counter += 1
		
	return total_loss/num_batches


def main():				
	#train conv-deconv-autodecoder on 14 years
	with tf.Session() as sess:
		model = CNN()
		sess.run(tf.global_variables_initializer())	
		for i in range(cfg.epoch):
			print("epoch " + str(i))
			for j in range(0, 5):
				print(str(j+2000)+" year:")
				trainForOneYear(j, sess, model)	
	
		#test on year 2000
		year = 0		
		loss = testForOneYear(year, sess, model)
		print("the validation loss on year " +str(2000+i)+" is "+str(loss))
					

main()
