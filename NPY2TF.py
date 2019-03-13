#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import parameters
import random
from sklearn import preprocessing

'''
Return 2D list of parameters, with the first axis year (according to the defined order), second axis type of data
'''
def loadNPY(yrs = [2000, 2001], IPATH = parameters.NPY_DATA_DIRECTORY, loadlist=['X0', 'X3', 'X6', 'Y3', 'Y6']):
    data = [[None for i in range(len(loadlist))] for j in range(len(yrs))]
    for i in range( len(yrs) ):
        for j in range( len(loadlist) ):
            datapath = IPATH + "/" + loadlist[j] + "_" + str(yrs[i]) + ".npy"
            print("Loading data from: " + loadlist[j] + "_" + str(yrs[i]) + ".npy")
            data[i][j] = np.load(datapath)
    for item in data:
        assert(item)
    return data

'''
2D list of nparrays -> reduced 2D list of nparraays
The operation is in place to save space
Dimensions of data:
index, measure type (param), height levels, latitude(41), longitude(141)
inputs are list of indices of interest
if allflag is 1, it will read all all dimensions and all dimensions are kept
'''
def dimSelect(data, allflag = 0, types = np.arange(parameters.Nparamens), heights = np.arange(parameters.Nheight),
 latitudes = np.arange(parameters.Nlatitude), longitudes = np.arange(parameters.Nlongitude)):
    _,Ntype,Nheight,Nlatitude,Nlongitude = data[0][0].shape
    if allflag:
        types = np.arange(Ntype)
        heights = np.arange(Nheight)
        latitudes = np.arange(Nlatitude)
        longitudes = np.arange(Nlongitude)
    d1 = len(data)
    d2 = len(data[0])
    for i in range(d1):
        for j in range(d2):
            data[i][j] = data[i][j][:,types,:,:,:]
            data[i][j] = data[i][j][:,:,heights,:,:]
            data[i][j] = data[i][j][:,:,:,latitudes,:]
            data[i][j] = data[i][j][:,:,:,:,longitudes]
    print("dim selection finished")
    return data
'''
2D list of nparrays -> normalized 2D list of nparrays
The operation is in place to save space
This implementation normalizes all the dimensions to mean 0 std 1
All the data type dims are used,
only refdims (year dimensions) are used to calculate the mean and std,
Other years will be be transformed by the mean and std
If refdim is empty, then all the dimensions will be used
'''
def dimNormalize(data, refdims=[], normdim=(0,)):
    d1 = len(data)
    d2 = len(data[0])
    assert(0 in normdim), "dimNormalizeAll: does not support index to be kept"
    if refdims == []:
        refdims = np.arange(d1)
    datacat = np.concatenate([data[i][j] for i in refdims for j in range(d2) ],axis=0)
    meanvec = np.mean(datacat, axis=normdim)
    stdvec = np.std(datacat, axis=normdim) + 0.00001 #epsilon
    orishape = data[0][0].shape
    newshape = []
    for dim in range (len(orishape)):
        if dim in normdim:
            newshape.append(1)
        else:
            newshape.append(orishape[dim])
    meanvec = meanvec.reshape(newshape)
    stdvec = stdvec.reshape(newshape)
    for i in range(d1):
        for j in range(d2):
            data[i][j] = (data[i][j] - meanvec) / stdvec
    print("dim normalize finished")
    return data

'''
dataX, dataY, str -> None
Write size X and Y to the specified OPath
Implementation is flattened, which requires resizing when loading the file
'''
def NPY2TF(X, Y, OPath):
    writer = tf.python_io.TFRecordWriter(OPath)
    size = X.shape[0]
    assert(size == Y.shape[0])
    for h in range(0, size):
        features = {
            'X': tf.train.Feature(float_list=tf.train.FloatList(value=X[h,:].flatten() )),
            'Y': tf.train.Feature(float_list=tf.train.FloatList(value=Y[h,:].flatten() ))
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()

'''
quickly calling NPY2TF for all different years 
'''
def quickNPY2TF(data, yrs, xinds, yinds, axis=1, comment=''):
    for i in range( len(yrs) ):
        opath = parameters.TF_DATA_DIRECTORY + "/tf_" + str(yrs[i]) + comment
        print("NPY2TF saving to: " + "/tf_" + str(yrs[i]) + comment)
        NPY2TF(np.concatenate([data[i][j] for j in xinds], axis=axis), np.concatenate([data[i][j] for j in yinds] ), opath)


def main():
    yrs = [2000]
    loadlist = ['X0','X3','Y3']
    data = loadNPY(yrs = yrs, loadlist=loadlist)
    data = dimSelect(data)
    data = dimNormalize(data, normdim=(0,2,3,4))
    quickNPY2TF(data,yrs,[0,2],[1],comment='_test') #


def test_main():
    data = loadNPY(loadlist=['X0','X3'])
    data = dimSelect(data, allflag=0, types=[0,1,3], heights=[1,5], latitudes=np.arange(20),longitudes=np.arange(20))
    print(data[0][0].shape)
    data = dimNormalizeAll(data, normdim=(0,2,3,4))
    #data = dimSelect(data, allflag=0, types=[0,1,3])
    print(data[0][0].shape)
    #print(np.mean(data[0][0], axis=0))


if __name__ == "__main__":
    #test_main()
    main()
