#!/usr/bin/env python2

import numpy as np
import data_macro
from eccodes import *


def GRIB2NPY(years = 2000 + np.arange(1)):
    print( years, "  of data will be processed...")
    for i in years:
        print("Start year: " + str(i))

        PATHI = data_macro.GRIB_DATA_DIRECTORY  # '$DIR/initdata'
        PATHO = data_macro.NPY_DATA_DIRECTORY   # '$DIR/npydata'
        INPUT_MEAN = PATHI + '/data_era5_mean_' + str(i)
        INPUT_TRAJ = PATHI + '/data_era5_traj_' + str(i)
        OUTPUTX0 = PATHO + '/X0_' + str(i)
        OUTPUTX3 = PATHO + '/X3_' + str(i)
        OUTPUTX6 = PATHO + '/X6_' + str(i)
        OUTPUTY3 = PATHO + '/Y3_' + str(i)
        OUTPUTY6 = PATHO + '/Y6_' + str(i)

        print("Loading files...")
        fmean = open(INPUT_MEAN)
        ftraj = open(INPUT_TRAJ)
        x = codes_count_in_file(fmean)
        print("Loading finished")

        Nheight = 7  # 7 hPa levels
        Nparamens = 6  #geopotential, u, v wind direction, fraction of cloud cover, relative humidity, temperature
        Nlatitude = 41
        Nlongitude = 141
        Nhours = x / (Nparamens * Nheight * 3)

        npx0 = np.full([Nhours, Nparamens, Nheight, Nlatitude, Nlongitude], np.nan, dtype=np.float32)
        npx3 = np.full([Nhours, Nparamens, Nheight, Nlatitude, Nlongitude], np.nan, dtype=np.float32)
        npx6 = np.full([Nhours, Nparamens, Nheight, Nlatitude, Nlongitude], np.nan, dtype=np.float32)
        npy3 = np.full([Nhours, Nparamens, Nheight, Nlatitude, Nlongitude], np.nan, dtype=np.float32)
        npy6 = np.full([Nhours, Nparamens, Nheight, Nlatitude, Nlongitude], np.nan, dtype=np.float32)

        print("Start extracting x0, x3, x6")
        for ise in xrange(0, x):
            hour_count = ise // (Nheight * Nparamens * 3)
            height_count = (ise // Nparamens) % Nheight  # (i-(i%Nparam))%Nheight

            gidm = codes_grib_new_from_file(fmean)
            type = codes_get(gidm, "shortName")
            typeind = data_macro.type2index[type]
            sr = int(codes_get(gidm, "stepRange"))
            values = codes_get_values(gidm)
            if sr == 0:
                for a in xrange(0, Nlatitude):
                    for b in xrange(0, Nlongitude):
                        npx0[hour_count, typeind, height_count, a, b] = values[Nlongitude*a + b]
            elif sr == 3:
                for a in xrange(0, Nlatitude):
                    for b in xrange(0, Nlongitude):
                        npx3[hour_count, typeind, height_count, a, b] = values[Nlongitude*a + b]
            elif sr == 6:
                for a in xrange(0, Nlatitude):
                    for b in xrange(0, Nlongitude):
                        npx6[hour_count, typeind, height_count, a, b] = values[Nlongitude*a + b]
            else:
                assert(0), "GRIB2NPY: not supporting the step size!"
            codes_release(gidm)

            if(ise%10000==0):
                print(str(ise) + "/" + str(x))

        print("Finished x0, x3, x6 extraction")

        assert(not np.isnan(npx0).any())
        assert(not np.isnan(npx3).any())
        assert(not np.isnan(npx6).any())

        np.save(OUTPUTX0,npx0)
        np.save(OUTPUTX3,npx3)
        np.save(OUTPUTX6,npx6)


        for ise in xrange(0, x):  # traj0
            hour_count = ise // (Nheight * Nparamens * 3)
            height_count = (ise // Nparamens) % Nheight
            param_iterator = ise % Nparamens

            gidt = codes_grib_new_from_file(ftraj)
            type = codes_get(gidt, "shortName")
            typeind = data_macro.type2index[type]
            sr = int(codes_get(gidt, "stepRange"))
            values = codes_get_values(gidt)

            if sr == 0:
                pass
            elif sr == 3:
                for a in xrange(0, Nlatitude):
                    for b in xrange(0, Nlongitude):
                        npy3[hour_count, typeind, height_count, a, b] = values[Nlongitude*a + b]
            elif sr == 6:
                for a in xrange(0, Nlatitude):
                    for b in xrange(0, Nlongitude):
                        npy6[hour_count, typeind, height_count, a, b] = values[Nlongitude*a + b]
            else:
                assert(0), "GRIB2NPY: not supporting the step size!"

            codes_release(gidt)
            if(ise%10000==0):
                print(str(ise) + "/" + str(x))

        print("Finished y3, y6 extraction")

        assert(not np.isnan(npy3).any())
        assert(not np.isnan(npy6).any())

        np.save(OUTPUTY3,npy3)
        np.save(OUTPUTY6,npy6)
        print("Year " + str(i) + "finished! ")

if __name__ == "__main__":
    # years entry is a list of years
    example = 2000 + np.arange(15)
    GRIB2NPY(years = [2000, 2001])
