import os
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = ROOT_DIRECTORY + "/data"
GRIB_DATA_DIRECTORY = DATA_DIRECTORY + "/initdata"
NPY_DATA_DIRECTORY = DATA_DIRECTORY + "/npydata"
TF_DATA_DIRECTORY = DATA_DIRECTORY + "/tfdata"

CACHE_ROOT = ROOT_DIRECTORY + "/cache"