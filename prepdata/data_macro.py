ROOT_DIRECTORY = "/media/chengyuan/2CC869A3C8696BD0/deep_weather"
DATA_DIRECTORY = ROOT_DIRECTORY + "/data"
GRIB_DATA_DIRECTORY = DATA_DIRECTORY + "/initdata"
NPY_DATA_DIRECTORY = DATA_DIRECTORY + "/npydata"
TF_DATA_DIRECTORY = DATA_DIRECTORY + "/tfdata"

Nheight = 7  # 7 hPa levels
Nparamens = 6
Nlatitude = 41
Nlongitude = 141


type2index = {
't' : 0,  # temperature
'u' : 1,  # horizontal wind
'v' : 2,  # vertical wind
'r' : 3,  # relative humidity
'cc' : 4, # cloud coverage
'z' : 5   # geopotential
}

index2type = {
0 : 't',
1 : 'u',
2 : 'v',
3 : 'r',
4 : 'cc',
5 : 'z'
}

index2pressure = {
0 : 150,
1 : 200,
2 : 250,
3 : 400,
4 : 500,
5 : 700,
6 : 850
}
