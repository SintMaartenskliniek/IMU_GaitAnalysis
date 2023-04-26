# -*- coding: utf-8 -*-
"""
@author: C.J. Ensink

Function to read h5 files of APDM sensors
Dependencies: h5py, numpy

Sint Maartenskliniek configuration "HAN measurements":
Sternum = '687'
Lumbar = '1784'
Right foot = '1366'
Left foot = '1264'

"""
# # Example running file
# from h5_reader import h5_reader

# filename = '20191024-103654_Walk.h5'
# sensordata = h5_reader(filename)


# Function
import h5py
import numpy as np

def h5_reader(filename):
    # Create empy dict to fill with sensordata
    sensordata = {}
   
    f = h5py.File(filename, 'r')
    for keys in f:
            # print(keys)
            if keys == 'Sensors':
                for sensornums in f[keys]:
                    # print(sensornums)
                    try:
                        for sensors in f[keys][sensornums]:
                            # print(sensors)
                            if sensors == 'Accelerometer':
                                sensordata[sensornums]['Accelerometer'] = np.array(f[keys][sensornums][sensors])
                            elif sensors == 'Gyroscope':
                                sensordata[sensornums]['Gyroscope'] = np.array(f[keys][sensornums][sensors])
                            elif sensors == 'Magnetometer':
                                sensordata[sensornums]['Magnetometer'] = np.array(f[keys][sensornums][sensors])
                            elif sensors == 'Time':
                                sensordata[sensornums]['Time'] = np.array(f[keys][sensornums][sensors])
                    except:
                        sensordata[sensornums] = {}
                        for sensors in f[keys][sensornums]:
                            # print(sensors)
                            if sensors == 'Accelerometer':
                                sensordata[sensornums]['Accelerometer'] = np.array(f[keys][sensornums][sensors])
                            elif sensors == 'Gyroscope':
                                sensordata[sensornums]['Gyroscope'] = np.array(f[keys][sensornums][sensors])
                            elif sensors == 'Magnetometer':
                                sensordata[sensornums]['Magnetometer'] = np.array(f[keys][sensornums][sensors])
                            elif sensors == 'Time':
                                sensordata[sensornums]['Time'] = np.array(f[keys][sensornums][sensors])
                            # elif sensors == 'Configuration':
                                # sensor1264['Configuration'] = np.array(f[keys][sensornums][sensors])
                        
                                
            elif keys == 'Processed':
                for sensornums in f[keys]:
                    # print(sensornums)
                    try:
                        for sensors in f[keys][sensornums]:
                            # print(sensors)
                            if sensors == 'Orientation':
                                    sensordata[sensornums]['Orientation'] = np.array(f[keys][sensornums][sensors])
                    except:
                        sensordata[sensornums] = {}
                        for sensors in f[keys][sensornums]:
                            if sensors == 'Orientation':
                                    sensordata[sensornums]['Orientation'] = np.array(f[keys][sensornums][sensors])
    
    try:
        # sensordata['Sternum'] = sensordata.pop('687')
        # sensordata['Lumbar'] = sensordata.pop('10242')
        # sensordata['Right foot'] = sensordata.pop('1366')
        # sensordata['Left foot'] = sensordata.pop('1264')
        
        sensordata['Sternum'] = sensordata.pop('687')
        sensordata['Lumbar'] = sensordata.pop('1784')
        sensordata['Right foot'] = sensordata.pop('1366')
        sensordata['Left foot'] = sensordata.pop('1264')
        
        # sensordata['Lumbar'] = sensordata['XI-006141']
        # sensordata['Right foot'] = sensordata['XI-006311']
        # sensordata['Left foot'] = sensordata['XI-006346']
    
    finally:
        return sensordata
    return sensordata







