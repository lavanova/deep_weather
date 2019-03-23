#from GRIB2NPY import GRIB2NPY
from NPY2TF import *

def main():
    Nyrs = 1
    yrs = np.arange(Nyrs) + 2000
    loadlist = ['X0','X3','Y3']
    #data = loadNPY(yrs = yrs, loadlist=loadlist)
    #data = dimSelect(data, latitudes = np.arange(40), longitudes = np.arange(40))
    data = loadSelect(yrs = yrs, loadlist=loadlist, latitudes = np.arange(30), longitudes = np.arange(30))
    data = dimNormalize(data, normdim=(0,2,3,4))
    quickNPY2TF(data,yrs,[0,2],[1],comment='_small_grid') #

if __name__ == "__main__":
    main()
