# -*- coding: utf-8 -*-
"""

@author: ensinkc

--------------------------------------
def data_filelist(datafolder):
    return filepaths

Use either the Mobility_Lab_Study_Export_... folder (including Walk_trials.csv info-file),
or the folder with exported XSens files as an input for this function.
The output will contain a list (filepaths) with the filepaths to the Xsens files of the 
left foot, right foot, and lumbar sensor, or the APDM file with all sensordata


--------------------------------------
def data_preprocessor(filepaths):
    return data_dict with:
        sample_frequency,
        ACCEFleft,
        quaternionorientationleft,
        eulerorientationleft,
        ACCSFleft, 
        GYRleft,
        MAGleft,
        ACCEFright,
        quaternionorientationright,
        eulerorientationright,
        ACCSFright,
        GYRright,
        MAGright,
        GYRlumbar,
        eulerorientationlumbar,
        timestamp

Pre-processing of the sensordata to a format that is suitable for the feet_processor pipeline.



"""
import math
import pandas as pd
import numpy as np
import os
import json
import time
from scipy.spatial.transform import Rotation as R
import pyquaternion
from pyquaternion import Quaternion
from scipy import signal
from .apdm_helpers import h5_reader
from .xsens_dot_helpers import calcCalGyr, calcCalAcc, eulFromQuat, calcFreeAcc

def data_filelist(datafolder):
    
    # Check files in datafolder
    for (dirpath, dirnames, filenames) in os.walk(datafolder):
        break
    
    # Check if there is a file called sensorspec.json
    sensorspecfile = False # default
    sample_frequency = False # default
    
    for i in range(0,len(filenames)):
        if 'sensorspec.json' in filenames[i]:
            with open(dirpath+'/'+filenames[i]) as f:
                sensorspec = json.load(f)
                f.close()
            sensorspecfile = True # sensorspec file was found
       
    if sensorspecfile == False:
        print('Sensor specification files is missing, looking for recognizable sensor files to assign default sensor specifications...')
        for i in range(0,len(filenames)):
            # Check if there is a file that ends with '.h5' if no sensorspec.json file is found
            if '.h5' in filenames[i] or datafolder.startswith('Mobility_Lab') or 'StudyMetadata' in filenames[i] or 'Walk_trials.csv' in filenames[i]:
                
                sensortype = 'apdm'
                print('Sensor files recognized as "APDM", assign default sensor specifications...')
                
                if '833' in filenames[i]: # data is from iGait study
                    continue
                
                else:
                    for j in range(0, len(dirnames)):
                        if 'raw' in dirnames[j]:
                            rawdir = dirnames[j]
                            
                    for j in range(0, len(filenames)):
                        if filenames[j].startswith('Walk_trials') and filenames[j].endswith('.csv'):
                            walkinfo = pd.read_csv((datafolder + '/' + filenames[j]), delimiter='\;', engine='python')
                            if len(walkinfo.columns) < 2:
                                walkinfo = pd.read_csv((datafolder + '/' + filenames[j]), delimiter='","', engine='python')
        
                    findcondition = walkinfo[walkinfo['Condition'] == '2-minute']
                    if len(findcondition) > 0:
                        file = (findcondition['File Name']).to_string(index = False)[:]
                    else:
                        file = 'remove'
            
            # if no sensorspec.json file available, and data is not APDM format ('.h5')
            else:
                # Check if first file contains 'MT_' format and if file is .txt file > Assumed Xsens Awinda
                if filenames[0].startswith('MT_') and filenames[0].endswith('.txt'):
                    print('Sensor files recognized as "Awinda", assign default sensor specifications...')
                    sensorspec = dict()
                    sensorspec['leftfoot'] = '00B40AC5.txt'
                    sensorspec['leftshank'] = '00B40ACF.txt' #leftlateralankle
                    sensorspec['lumbar'] = '00B40A8D.txt'
                    sensorspec['rightfoot'] = '00B40A23.txt'
                    sensorspec['rightshank'] ='00B40AC7.txt' #rightlateralankle
                    sensorspec['sternum'] = '00B40A40.txt'
                # Check if first file is .csv file > Assumed Xsens DOT, find filenames that contain either 'Linkervoet' or 'Rechtervoet'
                elif filenames[0].endswith('.csv'):
                    sensorspec = dict()
                    for i in range(0,len(filenames)):
                        if 'Linkervoet' in filenames[i] or 'Xsens4' in filenames[i]:
                            sensorspec['leftfoot'] = filenames[i]
                            print('Sensor files recognized as "DOT", assign default sensor specifications...')
                        elif 'Rechtervoet' in filenames[i] or 'Xsens 3' in filenames[i]:
                            sensorspec['rightfoot']= filenames[i]
                        elif 'D4CA6' in filenames[i]:
                            sensorspec['lumbar']= filenames[i]
                        elif 'Xsens 2' in filenames[i]:
                            sensorspec['sternum'] = filenames[i]
                # If none of the above, print that a sensorspec file is needed.
                else: 
                    sensorspec = dict()
                    print('Sensor specification files is missing, no sensor files recognized, please upload a "sensorspec.json.txt" file to the measurement folder.')
                
                
                
                
    # Assign full sensorfilename to the location
    for loc in sensorspec:
        for i in range(0,len(filenames)):
            if isinstance(sensorspec[loc], str) and sensorspec[loc] in filenames[i]:
                # Assign sensortype
                if filenames[i].endswith('.txt'):
                    sensortype = 'xsens_awinda'
                    
                elif filenames[i].startswith('gaitup') and filenames[i].endswith('.csv'):
                    # gaitup_...csv (gaitup header file excluded)
                    sensortype = 'gaitup'
                    
                elif filenames[i].endswith('.csv'):
                    sensortype = 'xsens_dot'
                    
                elif filenames[i].startswith('apdm') and filenames[i].endswith('.h5'):
                    # apdm....h5
                    sensortype = 'apdm_homemonitoring'
                else:
                    continue
                               
                sensorspec[loc] = filenames[i]
        if loc == 'sample_frequency' or loc == 'sample frequency':
            sample_frequency = sensorspec[loc]
    
    
    # Define filepaths
    if sensortype == 'xsens_awinda' or sensortype == 'xsens_dot' or sensortype == 'apdm_homemonitoring' or sensortype == 'gaitup':
        filepaths = getFilePathsFromSensorspec(datafolder, sensorspec)
    
    elif sensortype == 'apdm':
        if '833' in filenames[0]: # data is from iGait study
            filepaths=list()
            for k in range(0, len(filenames)):
                filepaths.append(datafolder + '/' + filenames[k])
        
        else:
            filepaths = [datafolder + '/' + rawdir + '/' + file]
            filepaths = [i for i in filepaths if 'remove' not in i]
    else:
        print('Something went wrong in recognizing the filetypes')
        
    return filepaths, sensortype, sample_frequency
    

def getFilePathsFromSensorspec(datafolder, sensorspec):
    filepaths = {}
    try:
        filepaths['LeftFoot'] = datafolder + '/' + sensorspec['leftfoot']
    except:
        filepaths['LeftFoot'] = []

    try:
        filepaths['RightFoot'] = datafolder + '/' + sensorspec['rightfoot']
    except:
        filepaths['RightFoot'] = []

    try:
        filepaths['LeftShank'] = datafolder + '/' + sensorspec['leftshank']
    except:
        filepaths['LeftShank'] = []

    try:
        filepaths['RightShank'] = datafolder + '/' + sensorspec['rightshank']
    except:
        filepaths['RightShank'] = []

    try:
        filepaths['Lumbar'] = datafolder + '/' + sensorspec['lumbar']
    except:
        filepaths['Lumbar'] = []

    try:
        filepaths['Sternum'] = datafolder + '/' + sensorspec['sternum']
    except:
        filepaths['Sternum'] = []
        
    return filepaths

def data_preprocessor(filepaths, sensortype, **kwargs):
  
    if sensortype == 'xsens_dot':
        # Define Sample Frequency (Hz)
        # Default
        sample_frequency = 60
    
        for key, value in kwargs.items():
            if key == 'sample_frequency':
                sample_frequency = value
    
        try:
            dataleft = pd.read_csv(filepaths['LeftFoot'], delimiter=',', engine='python', skiprows = 11)
        except ValueError:
            dataleft = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_y', 'Mag_Z', 'Status']))
        
        try:
            dataright = pd.read_csv(filepaths['RightFoot'], delimiter=',', engine='python', skiprows = 11)
        except ValueError:
            dataright = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_y', 'Mag_Z', 'Status']))
        
        try:
            dataleftshank = pd.read_csv(filepaths['LeftShank'], delimiter=',', engine='python', skiprows = 11)
        except ValueError:
            dataleftshank = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_y', 'Mag_Z', 'Status']))

        try:
            datarightshank = pd.read_csv(filepaths['RightShank'], delimiter=',', engine='python', skiprows = 11)
        except ValueError:
            datarightshank = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_y', 'Mag_Z', 'Status']))

        try:
            datalumbar = pd.read_csv(filepaths['Lumbar'], delimiter=',', engine='python', skiprows = 11)
        except ValueError:
            datalumbar = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_y', 'Mag_Z', 'Status']))
            
        try:
            datasternum = pd.read_csv(filepaths['Sternum'], delimiter=',', engine='python', skiprows = 11)
        except ValueError:
            datasternum = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_y', 'Mag_Z', 'Status']))
                
        
        # Check if all sensors contain data
        missingsensors = []
        if len(dataleft) == 0:
            print('No left foot data available')
            missingsensors.append('LeftFoot')
        if len(dataright) == 0:
            print('No right foot data available')
            missingsensors.append('RightFoot')
        if len(dataleftshank) == 0:
            print('No left shank data available')
            missingsensors.append('LeftShank')
        if len(datarightshank) == 0:
            print('No right shank data available')
            missingsensors.append('RightShank')
        if len(datalumbar) == 0:
            print('No lumbar data available')
            missingsensors.append('Lumbar')
        if len(datasternum) == 0:
            print('No sternum data available')
            missingsensors.append('Sternum')
        
        # Make all data frames the same length
        lenleft = max(dataleft['PacketCounter'])
        lenright = max(dataright['PacketCounter'])
        if 'LeftShank' not in missingsensors:
            lenleftshank = max(dataleftshank['PacketCounter'])
        else:
            lenleftshank = max(lenleft,lenright)*2
        if 'RightShank' not in missingsensors:
            lenrightshank = max(datarightshank['PacketCounter'])
        else:
            lenrightshank = max(lenleft, lenright) * 2
        if 'Lumbar' not in missingsensors:
            lenlumbar = max(datalumbar['PacketCounter'])
        else:
            lenlumbar = max(lenleft,lenright)*2
        if 'Sternum' not in missingsensors:
            lensternum = max(datasternum['PacketCounter'])
        else:
            lensternum = max(lenleft,lenright)*2
        
        datalens = [lenleft,lenright,lenleftshank,lenrightshank,lenlumbar,lensternum]
        minlen = min(datalens)
        index_minlen = min(range(len(datalens)), key=datalens.__getitem__)
        
        if index_minlen == 0:
            lastsample = np.where(dataleft['PacketCounter']==lenleft)[0][0]
        elif index_minlen == 1:
            lastsample = np.where(dataright['PacketCounter']==lenright)[0][0]
        elif index_minlen == 2:
            lastsample = np.where(dataleftshank['PacketCounter'] == lenleftshank)[0][0]
        elif index_minlen == 3:
            lastsample = np.where(datarightshank['PacketCounter'] == lenrightshank)[0][0]
        elif index_minlen == 4:
            lastsample = np.where(datalumbar['PacketCounter']==lenlumbar)[0][0]
        elif index_minlen == 5:
            lastsample = np.where(datasternum['PacketCounter']==lensternum)[0][0]
        
        if (len(dataleft)-1)-lastsample == 1:
            dataleft = dataleft.drop(lastsample+1)
        else:
            dataleft = dataleft.drop(range(lastsample+1, len(dataleft)-1))
        if (len(dataright)-1)-lastsample == 1:
            dataright = dataright.drop(lastsample+1)
        else:
            dataright = dataright.drop(range(lastsample+1, len(dataright)-1))
        if (len(dataleftshank)-1)-lastsample == 1:
            dataleftshank = dataleftshank.drop(lastsample+1)
        else:
            dataleftshank = dataleftshank.drop(range(lastsample+1, len(dataleftshank)-1))
        if (len(datarightshank)-1)-lastsample == 1:
            datarightshank = datarightshank.drop(lastsample+1)
        else:
            datarightshank = datarightshank.drop(range(lastsample+1, len(datarightshank)-1))
        if (len(datalumbar)-1)-lastsample == 1:
            datalumbar = datalumbar.drop(lastsample+1)
        else:
            datalumbar = datalumbar.drop(range(lastsample+1, len(datalumbar)-1))
        if (len(datasternum)-1)-lastsample == 1:
            datasternum = datasternum.drop(lastsample+1)
        else:
            datasternum = datasternum.drop(range(lastsample+1, len(datasternum)-1))
        
        
        # Find missing packets
        packets = pd.DataFrame()
        if 'LeftFoot' not in missingsensors:
            packets = pd.concat([packets, dataleft['PacketCounter']])
        if 'RightFoot' not in missingsensors:
            packets = pd.concat([packets, dataright['PacketCounter']])
        if 'LeftShank' not in missingsensors:
            packets = pd.concat([packets, dataleftshank['PacketCounter']])
        if 'RightShank' not in missingsensors:
            packets = pd.concat([packets, datarightshank['PacketCounter']])
        if 'Lumbar' not in missingsensors:
            packets = pd.concat([packets, datalumbar['PacketCounter']])
        if 'Sternum' not in missingsensors:
            packets = pd.concat([packets, datasternum['PacketCounter']])
        allpacketslist = list(packets[0])
        
        # Sort packets
        allpacketslist.sort()
        
        # Constants Declaration
        missingpackets = np.array([])
        prev = -1
        count = 0
        
        # Iterating
        unique, counts = np.unique(allpacketslist, return_counts=True)
        count = dict(zip(unique, counts))
        amount_double=0
        for key in count:
            if count[key] < 4-len(missingsensors):
                missingpackets = np.append(missingpackets, key)
                if count[key]==0:
                    print('0!')
            if count[key] > 4-len(missingsensors):
                amount_double=amount_double+1
        if amount_double > 0:
            print('WARNING: There are ' + str(amount_double) + ' double values in the PacketCounter!')
    
        missingpackets = missingpackets[np.where(~np.isnan(missingpackets))]
        
        print("Measurement " + str(filepaths['LeftFoot'][13:-32]) +" contains " + str(len(missingpackets)) + " missing packets in available sensordata (=" + str(round(len(missingpackets)/(lenleft+lenright)*100,2)) + '%)')
        
        # Remove missing packets
        dataleft = dataleft[~dataleft['PacketCounter'].isin(missingpackets)]
        dataright = dataright[~dataright['PacketCounter'].isin(missingpackets)]
        dataleftshank = dataleftshank[~dataleftshank['PacketCounter'].isin(missingpackets)]
        datarightshank = datarightshank[~datarightshank['PacketCounter'].isin(missingpackets)]
        datalumbar = datalumbar[~datalumbar['PacketCounter'].isin(missingpackets)]
        datasternum = datasternum[~datasternum['PacketCounter'].isin(missingpackets)]
        
        # Define sensordata
        quaternionorientationleft = (dataleft.filter(['Quat_W','Quat_X','Quat_Y','Quat_Z'])).to_numpy()
        eulerorientationleft = eulFromQuat(quaternionorientationleft)
        GYRleft = calcCalGyr((dataleft.filter(['dq_W','dq_X','dq_Y','dq_Z'])).to_numpy(), sample_frequency)
        ACCSFleft = calcCalAcc((dataleft.filter(['dv[1]','dv[2]','dv[3]'])).to_numpy(), sample_frequency)
        MAGleft = (dataleft.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        ACCEFleft = calcFreeAcc(quaternionorientationleft, ACCSFleft)
        
        quaternionorientationright = (dataright.filter(['Quat_W','Quat_X','Quat_Y','Quat_Z'])).to_numpy()
        eulerorientationright = eulFromQuat(quaternionorientationright)
        GYRright = calcCalGyr((dataright.filter(['dq_W','dq_X','dq_Y','dq_Z'])).to_numpy(), sample_frequency)
        ACCSFright = calcCalAcc((dataright.filter(['dv[1]','dv[2]','dv[3]'])).to_numpy(), sample_frequency)
        MAGright = (dataright.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        ACCEFright = calcFreeAcc(quaternionorientationright, ACCSFright)

        quaternionorientationleftshank = (dataleftshank.filter(['Quat_W','Quat_X','Quat_Y','Quat_Z'])).to_numpy()
        eulerorientationleftshank = eulFromQuat(quaternionorientationleftshank)
        GYRleftshank = calcCalGyr((dataleftshank.filter(['dq_W','dq_X','dq_Y','dq_Z'])).to_numpy(), sample_frequency)
        ACCSFleftshank = calcCalAcc((dataleftshank.filter(['dv[1]','dv[2]','dv[3]'])).to_numpy(), sample_frequency)
        MAGleftshank = (dataleftshank.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        ACCEFleftshank = calcFreeAcc(quaternionorientationleftshank, ACCSFleft)

        quaternionorientationrightshank = (datarightshank.filter(['Quat_W','Quat_X','Quat_Y','Quat_Z'])).to_numpy()
        eulerorientationrightshank = eulFromQuat(quaternionorientationrightshank)
        GYRrightshank = calcCalGyr((datarightshank.filter(['dq_W','dq_X','dq_Y','dq_Z'])).to_numpy(), sample_frequency)
        ACCSFrightshank = calcCalAcc((datarightshank.filter(['dv[1]','dv[2]','dv[3]'])).to_numpy(), sample_frequency)
        MAGrightshank = (datarightshank.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        ACCEFrightshank = calcFreeAcc(quaternionorientationrightshank, ACCSFright)
        
        quaternionorientationlumbar = (datalumbar.filter(['Quat_W','Quat_X','Quat_Y','Quat_Z'])).to_numpy()
        eulerorientationlumbar = eulFromQuat(quaternionorientationlumbar)
        GYRlumbar = calcCalGyr((datalumbar.filter(['dq_W','dq_X','dq_Y','dq_Z'])).to_numpy(), sample_frequency)
        ACCSFlumbar = calcCalAcc((datalumbar.filter(['dv[1]','dv[2]','dv[3]'])).to_numpy(), sample_frequency)
        MAGlumbar = (datalumbar.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        ACCEFlumbar = calcFreeAcc(quaternionorientationlumbar, ACCSFlumbar)
        
        quaternionorientationsternum = (datasternum.filter(['Quat_W','Quat_X','Quat_Y','Quat_Z'])).to_numpy()
        eulerorientationsternum = eulFromQuat(quaternionorientationsternum)
        GYRsternum = calcCalGyr((datasternum.filter(['dq_W','dq_X','dq_Y','dq_Z'])).to_numpy(), sample_frequency)
        ACCSFsternum = calcCalAcc((datasternum.filter(['dv[1]','dv[2]','dv[3]'])).to_numpy(), sample_frequency)
        MAGsternum = (datasternum.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        ACCEFsternum = calcFreeAcc(quaternionorientationsternum, ACCSFsternum)
        
        # Define timestamp
        timestamp = np.linspace(start=0,stop=(len(dataleft)-1)/sample_frequency,num=len(dataleft))
        
    elif sensortype == 'xsens_awinda':
        
        # Define Sample Frequency (Hz)
        # Default
        sample_frequency = 100
        
        for key, value in kwargs.items():
            if key == 'sample_frequency':
                sample_frequency = value
        
        
        try:
            dataleft = pd.read_csv(filepaths['LeftFoot'], delimiter='\t', engine='python', skiprows = 12)
        except ValueError:
            dataleft = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        try:
            dataright = pd.read_csv(filepaths['RightFoot'], delimiter='\t', engine='python', skiprows = 12)
        except ValueError:
            dataright = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        try:
            dataleftshank = pd.read_csv(filepaths['LeftShank'], delimiter='\t', engine='python', skiprows = 12)
        except ValueError:
            dataleftshank = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        try:
            datarightshank = pd.read_csv(filepaths['RightShank'], delimiter='\t', engine='python', skiprows = 12)
        except ValueError:
            datarightshank = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        try:
            datalumbar = pd.read_csv(filepaths['Lumbar'], delimiter='\t', engine='python', skiprows = 12)
        except ValueError:
            datalumbar = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        except KeyError:
            datalumbar = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        try:
            datasternum = pd.read_csv(filepaths['Sternum'], delimiter='\t', engine='python', skiprows = 12)
        except ValueError:
            datasternum = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
        except KeyError:
            datasternum = pd.DataFrame(columns=(['PacketCounter', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                                              'Mag_X', 'Mag_Y', 'Mag_Z', 'VelInc_X', 'VelInc_Y', 'VelInc_Z', 'Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3', 'Roll', 'Pitch', 'Yaw']))
            
        # Check if all sensors contain data
        missingsensors = []
        if len(dataleft) == 0:
            print('No left foot data available')
            missingsensors.append('LeftFoot')
        if len(dataright) == 0:
            print('No right foot data available')
            missingsensors.append('RightFoot')
        if len(dataleftshank) == 0:
            print('No left shank data available')
            missingsensors.append('LeftShank')
        if len(datarightshank) == 0:
            print('No right shank data available')
            missingsensors.append('RightShank')
        if len(datalumbar) == 0:
            print('No lumbar data available')
            missingsensors.append('Lumbar')
        if len(datasternum) == 0:
            print('No sternum data available')
            missingsensors.append('Sternum')
        
        
        # Find missing packets
        # allpackets = np.hstack((np.hstack((dataleft['PacketCounter'].to_numpy(), dataright['PacketCounter'].to_numpy())), np.hstack((datalumbar['PacketCounter'].to_numpy(), datasternum['PacketCounter'].to_numpy() )) ))
        packets = pd.DataFrame()
        if 'LeftFoot' not in missingsensors:
            packets = pd.concat([packets, dataleft['PacketCounter']])
        if 'RightFoot' not in missingsensors:
            packets = pd.concat([packets, dataright['PacketCounter']])
        if 'LeftShank' not in missingsensors:
            packets = pd.concat([packets, dataleftshank['PacketCounter']])
        if 'RightShank' not in missingsensors:
            packets = pd.concat([packets, datarightshank['PacketCounter']])
        if 'Lumbar' not in missingsensors:
            packets = pd.concat([packets, datalumbar['PacketCounter']])
        if 'Sternum' not in missingsensors:
            packets = pd.concat([packets, datasternum['PacketCounter']])
        allpacketslist = list(packets[0])
        
        # allpackets = pd.concat([dataleft['PacketCounter'], dataright['PacketCounter'], datalumbar['PacketCounter'], datasternum['PacketCounter']]).to_numpy()
        
        # allpacketslist = list(allpackets)
        # Sort packets
        allpacketslist.sort()
        # Constants Declaration
        missingpackets = np.array([])
        prev = -1
        count = 0
        
        # Iterating
        unique, counts = np.unique(allpacketslist, return_counts=True)
        count = dict(zip(unique, counts))
        for key in count:
            if count[key] < 4-len(missingsensors):
                missingpackets = np.append(missingpackets, key)
        # print("Measurement " + str(filepaths['LeftFoot'][132:151]) +" contains " + str(len(missingpackets)) + " missing packets in available sensordata")
        
        # Remove missing packets
        dataleft = dataleft[~dataleft['PacketCounter'].isin(missingpackets)]
        dataright = dataright[~dataright['PacketCounter'].isin(missingpackets)]
        dataleftshank = dataleftshank[~dataleftshank['PacketCounter'].isin(missingpackets)]
        datarightshank = datarightshank[~datarightshank['PacketCounter'].isin(missingpackets)]
        datalumbar = datalumbar[~datalumbar['PacketCounter'].isin(missingpackets)]
        datasternum = datasternum[~datasternum['PacketCounter'].isin(missingpackets)]
        
        # Define sensordata
        ACCEFleft = (dataleft.filter(['FreeAcc_E','FreeAcc_N','FreeAcc_U'])).to_numpy()
        quaternionorientationleft = (dataleft.filter(['Quat_q0','Quat_q1','Quat_q2','Quat_q3'])).to_numpy()
        eulerorientationleft = (dataleft.filter(['Roll','Pitch','Yaw'])).to_numpy()
        ACCSFleft = (dataleft.filter(['Acc_X','Acc_Y','Acc_Z'])).to_numpy()
        GYRleft = (dataleft.filter(['Gyr_X','Gyr_Y','Gyr_Z'])).to_numpy()
        MAGleft = (dataleft.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        
        ACCEFright = (dataright.filter(['FreeAcc_E','FreeAcc_N','FreeAcc_U'])).to_numpy()
        quaternionorientationright = (dataright.filter(['Quat_q0','Quat_q1','Quat_q2','Quat_q3'])).to_numpy()
        eulerorientationright = (dataright.filter(['Roll','Pitch','Yaw'])).to_numpy()
        ACCSFright = (dataright.filter(['Acc_X','Acc_Y','Acc_Z'])).to_numpy()
        GYRright = (dataright.filter(['Gyr_X','Gyr_Y','Gyr_Z'])).to_numpy()
        MAGright = (dataright.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()

        ACCEFleftshank = (dataleftshank.filter(['FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U'])).to_numpy()
        quaternionorientationleftshank = (dataleftshank.filter(['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3'])).to_numpy()
        eulerorientationleftshank = (dataleftshank.filter(['Roll', 'Pitch', 'Yaw'])).to_numpy()
        ACCSFleftshank = (dataleftshank.filter(['Acc_X', 'Acc_Y', 'Acc_Z'])).to_numpy()
        GYRleftshank = (dataleftshank.filter(['Gyr_X', 'Gyr_Y', 'Gyr_Z'])).to_numpy()
        MAGleftshank = (dataleftshank.filter(['Mag_X', 'Mag_Y', 'Mag_Z'])).to_numpy()

        ACCEFrightshank = (datarightshank.filter(['FreeAcc_E', 'FreeAcc_N', 'FreeAcc_U'])).to_numpy()
        quaternionorientationrightshank = (datarightshank.filter(['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3'])).to_numpy()
        eulerorientationrightshank = (datarightshank.filter(['Roll', 'Pitch', 'Yaw'])).to_numpy()
        ACCSFrightshank = (datarightshank.filter(['Acc_X', 'Acc_Y', 'Acc_Z'])).to_numpy()
        GYRrightshank = (datarightshank.filter(['Gyr_X', 'Gyr_Y', 'Gyr_Z'])).to_numpy()
        MAGrightshank = (datarightshank.filter(['Mag_X', 'Mag_Y', 'Mag_Z'])).to_numpy()
        
        ACCEFlumbar = (datalumbar.filter(['FreeAcc_E','FreeAcc_N','FreeAcc_U'])).to_numpy()
        quaternionorientationlumbar = (datalumbar.filter(['Quat_q0','Quat_q1','Quat_q2','Quat_q3'])).to_numpy()
        eulerorientationlumbar = (datalumbar.filter(['Roll','Pitch','Yaw'])).to_numpy()
        ACCSFlumbar = (datalumbar.filter(['Acc_X','Acc_Y','Acc_Z'])).to_numpy()
        GYRlumbar = (datalumbar.filter(['Gyr_X','Gyr_Y','Gyr_Z'])).to_numpy()
        MAGlumbar = (datalumbar.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()
        
        ACCEFsternum = (datasternum.filter(['FreeAcc_E','FreeAcc_N','FreeAcc_U'])).to_numpy()
        quaternionorientationsternum = (datasternum.filter(['Quat_q0','Quat_q1','Quat_q2','Quat_q3'])).to_numpy()
        eulerorientationsternum = (datasternum.filter(['Roll','Pitch','Yaw'])).to_numpy()
        ACCSFsternum = (datasternum.filter(['Acc_X','Acc_Y','Acc_Z'])).to_numpy()
        GYRsternum = (datasternum.filter(['Gyr_X','Gyr_Y','Gyr_Z'])).to_numpy()
        MAGsternum = (datasternum.filter(['Mag_X','Mag_Y','Mag_Z'])).to_numpy()

        # Define timestamp
        timestamp = np.zeros((len(dataleft),1))
        for i in range(1,len(dataleft)):
            timestamp[i] = timestamp[i-1]+1/sample_frequency
        timestamp=timestamp.reshape(len(timestamp))
        
    elif sensortype == 'apdm' or sensortype == 'apdm_homemonitoring':
        
        if sensortype == 'apdm':
            missingpackets = np.array([0], dtype=float)
            # Define Sample Frequency (Hz)
            sample_frequency = 128
            
            data = h5_reader(filepaths[0])
            
            dataleft = data['Left foot']
            dataright = data['Right foot']
            dataleftshank = data['Left shank']
            datarightshank = data['Right shank']
            datalumbar = data['Lumbar']
            datasternum = data['Sternum']
        elif sensortype == 'apdm_homemonitoring':
            missingpackets = np.array([0], dtype=float)
            # we expect a single sensor per h5 file.
            try:
                dataSensor = h5_reader(filepaths['LeftFoot'])
                dataleft = dataSensor[list(dataSensor.keys())[0]]
            except:
                dataleft = []
            
            try:
                dataSensor = h5_reader(filepaths['RightFoot'])
                dataright = dataSensor[list(dataSensor.keys())[0]]
            except:
                dataright = []

            try:
                dataSensor = h5_reader(filepaths['LeftShank'])
                dataleftshank = dataSensor[list(dataSensor.keys())[0]]
            except:
                dataleftshank = []

            try:
                dataSensor = h5_reader(filepaths['RightShank'])
                datarightshank = dataSensor[list(dataSensor.keys())[0]]
            except:
                datarightshank = []
            
            try:
                dataSensor = h5_reader(filepaths['Lumbar'])
                datalumbar = dataSensor[list(dataSensor.keys())[0]]
            except:
                datalumbar = []
            
            try:
                dataSensor = h5_reader(filepaths['Sternum'])
                datasternum = dataSensor[list(dataSensor.keys())[0]]
            except:
                datasternum = []
            
        else:
            print('Unexpected type for apdm in preprocessor') # error('Unexpected type for apdm in preprocessor')
        
        
        # Check if all sensors contain data
        missingsensors = []
        if len(dataleft) == 0:
            print('No left foot data available')
            missingsensors.append('LeftFoot')
        if len(dataright) == 0:
            print('No right foot data available')
            missingsensors.append('RightFoot')
        if len(dataleftshank) == 0:
            print('No left shank data available')
            missingsensors.append('LeftShank')
        if len(datarightshank) == 0:
            print('No right shank data available')
            missingsensors.append('RightShank')
        if len(datalumbar) == 0:
            print('No lumbar data available')
            missingsensors.append('Lumbar')
        if len(datasternum) == 0:
            print('No sternum data available')
            missingsensors.append('Sternum')
            
            
        # sync data (based on offset in Time!)
        syncFailed, intLeft, intRight, intLeftShank, intRightShank, intLumbar, intSternum, timestamp, sample_frequency = syncData('apdm', dataleft, dataright, dataleftshank, datarightshank, datalumbar, datasternum)
        # Define sensordata
        ACCSFleft, GYRleft, MAGleft, ACCEFleft, quaternionorientationleft, eulerorientationleft = extractApdmData(dataleft, intLeft)
        ACCSFright, GYRright, MAGright, ACCEFright, quaternionorientationright, eulerorientationright = extractApdmData(dataright, intRight)
        ACCSFleftshank, GYRleftshank, MAGleftshank, ACCEFleftshank, quaternionorientationleftshank, eulerorientationleftshank = extractApdmData(dataleftshank, intLeftShank)
        ACCSFrightshank, GYRrightshank, MAGrightshank, ACCEFrightshank, quaternionorientationrightshank, eulerorientationrightshank = extractApdmData(datarightshank, intRightShank)
        ACCSFlumbar, GYRlumbar, MAGlumbar, ACCEFlumbar, quaternionorientationlumbar, eulerorientationlumbar = extractApdmData(datalumbar, intLumbar)
        ACCSFsternum, GYRsternum, MAGsternum, ACCEFsternum, quaternionorientationsternum, eulerorientationsternum = extractApdmData(datasternum, intSternum)
        
        # static rotation about z-axis of acceleration and gyroscope in sensor frame
        ACCSFleft = staticRotation(ACCSFleft)
        GYRleft = staticRotation(GYRleft)
        MAGleft = staticRotation(MAGleft)
        ACCSFright = staticRotation(ACCSFright)
        GYRright = staticRotation(GYRright)
        MAGright = staticRotation(MAGright)
        ACCSFleftshank = staticRotation(ACCSFleftshank)
        GYRleftshank = staticRotation(GYRleftshank)
        MAGleftshank = staticRotation(MAGleftshank)
        ACCSFrightshank = staticRotation(ACCSFrightshank)
        GYRrightshank = staticRotation(GYRrightshank)
        MAGrightshank = staticRotation(MAGrightshank)
        ACCSFlumbar = staticRotation(ACCSFlumbar)
        GYRlumbar = staticRotation(GYRlumbar)
        MAGlumbar = staticRotation(MAGlumbar)
        ACCSFsternum = staticRotation(ACCSFsternum)
        GYRsternum = staticRotation(GYRsternum)
        MAGsternum = staticRotation(MAGsternum)

    elif sensortype == 'gaitup':
        #skiprows = [1] if 4Daagse and home monitoring

        missingpackets = np.array([0], dtype=float)
        #changed by Jean Ormiston 27/05 to csv from RTK
        try:
            dataleft = pd.read_csv(filepaths['LeftFoot'], low_memory=False,skiprows=[1],dtype = {('Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z',
                  'Mag X', 'Mag Y', 'Mag Z', 'Quat W', 'Quat X', 'Quat Y', 'Quat Z'): np.float64}) #whole file = skiprows=[*range(0,5),6]
        except:
            dataleft = []
            
        try:
            dataright = pd.read_csv(filepaths['RightFoot'],low_memory=False,skiprows=[1],dtype = {('Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z',
                  'Mag X', 'Mag Y', 'Mag Z', 'Quat W', 'Quat X', 'Quat Y', 'Quat Z'): np.float64})#whole file = skiprows=[*range(0,5),6]
        except:
            dataright = []
            
        try:
            dataleftshank = pd.read_csv(filepaths['LeftShank'], low_memory=False, skiprows=[1],
                                   dtype={('Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z',
                                           'Mag X', 'Mag Y', 'Mag Z', 'Quat W', 'Quat X', 'Quat Y',
                                           'Quat Z'): np.float64})  # whole file = skiprows=[*range(0,5),6]
        except:
            dataleftshank = []

        try:
            datarightshank = pd.read_csv(filepaths['RightShank'], low_memory=False, skiprows=[1],
                                    dtype={('Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z',
                                            'Mag X', 'Mag Y', 'Mag Z', 'Quat W', 'Quat X', 'Quat Y',
                                            'Quat Z'): np.float64})  # whole file = skiprows=[*range(0,5),6]
        except:
            datarightshank = []

        try:
            datalumbar = pd.read_csv(filepaths['Lumbar'],low_memory=False,skiprows=[1],dtype = {('Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z',
                  'Mag X', 'Mag Y', 'Mag Z', 'Quat W', 'Quat X', 'Quat Y', 'Quat Z'): np.float64})#whole file = skiprows=[*range(0,5),6]
        except:
            datalumbar = []
            
        try:
            datasternum = pd.read_csv(filepaths['Sternum'],low_memory=False,skiprows=[1],dtype = {('Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z',
                  'Mag X', 'Mag Y', 'Mag Z', 'Quat W', 'Quat X', 'Quat Y', 'Quat Z'): np.float64}) #whole file = skiprows=[*range(0,5),6]
        except:
            datasternum = []
            
        # Check if all sensors contain data
        missingsensors = []
        if len(dataleft) == 0:
            print('No left foot data available')
            missingsensors.append('LeftFoot')
        if len(dataright) == 0:
            print('No right foot data available')
            missingsensors.append('RightFoot')
        if len(dataleftshank) == 0:
            print('No left shank data available')
            missingsensors.append('LeftShank')
        if len(datarightshank) == 0:
            print('No right shank data available')
            missingsensors.append('RightShank')
        if len(datalumbar) == 0:
            print('No lumbar data available')
            missingsensors.append('Lumbar')
        if len(datasternum) == 0:
            print('No sternum data available')
            missingsensors.append('Sternum')

        # Find missing packets
        # allpackets = np.hstack((np.hstack((dataleft['PacketCounter'].to_numpy(), dataright['PacketCounter'].to_numpy())), np.hstack((datalumbar['PacketCounter'].to_numpy(), datasternum['PacketCounter'].to_numpy() )) ))
        # packets = pd.DataFrame()
        # if 'LeftFoot' not in missingsensors:
        #     packets = pd.concat([packets, dataleft['PacketCounter']])
        # if 'RightFoot' not in missingsensors:
        #     packets = pd.concat([packets, dataright['PacketCounter']])
        # if 'Lumbar' not in missingsensors:
        #     packets = pd.concat([packets, datalumbar['PacketCounter']])
        # if 'Sternum' not in missingsensors:
        #     packets = pd.concat([packets, datasternum['PacketCounter']])
        # allpacketslist = list(packets[0])
        #
        # # allpackets = pd.concat([dataleft['PacketCounter'], dataright['PacketCounter'], datalumbar['PacketCounter'], datasternum['PacketCounter']]).to_numpy()
        #
        # # allpacketslist = list(allpackets)
        # # Sort packets
        # allpacketslist.sort()
        # # Constants Declaration
        # missingpackets = np.array([])
        # prev = -1
        # count = 0
        #
        # # Iterating
        # unique, counts = np.unique(allpacketslist, return_counts=True)
        # count = dict(zip(unique, counts))
        # for key in count:
        #     if count[key] < 4 - len(missingsensors):
        #         missingpackets = np.append(missingpackets, key)
        # print("Measurement " + str(filepaths['LeftFoot'][132:151]) + " contains " + str(
        #     len(missingpackets)) + " missing packets in available sensordata")
        #
        # # Remove missing packets
        # dataleft = dataleft[~dataleft['PacketCounter'].isin(missingpackets)]
        # dataright = dataright[~dataright['PacketCounter'].isin(missingpackets)]
        # datalumbar = datalumbar[~datalumbar['PacketCounter'].isin(missingpackets)]
        # datasternum = datasternum[~datasternum['PacketCounter'].isin(missingpackets)]
        # sync data (based on offset in Time!)
        syncFailed, intLeft, intRight, intLeftShank, intRightShank, intLumbar, intSternum, timestamp, sample_frequency = syncData('gaitup', dataleft, dataright, dataleftshank, datarightshank, datalumbar, datasternum)
        # print('Elapsed for preprocessing (sync indices sensors) : %s' % (time.time() - t))

        # Define sensordata
        ACCSFleft, GYRleft, MAGleft, ACCEFleft, quaternionorientationleft, eulerorientationleft = extractGaitupData(dataleft, intLeft)
        ACCSFright, GYRright, MAGright, ACCEFright, quaternionorientationright, eulerorientationright = extractGaitupData(dataright, intRight)
        ACCSFleftshank, GYRleftshank, MAGleftshank, ACCEFleftshank, quaternionorientationleftshank, eulerorientationleftshank = extractGaitupData(dataleftshank, intLeftShank)
        ACCSFrightshank, GYRrightshank, MAGrightshank, ACCEFrightshank, quaternionorientationrightshank, eulerorientationrightshank = extractGaitupData(datarightshank, intRightShank)
        ACCSFlumbar, GYRlumbar, MAGlumbar, ACCEFlumbar, quaternionorientationlumbar, eulerorientationlumbar = extractGaitupData(datalumbar, intLumbar)
        ACCSFsternum, GYRsternum, MAGsternum, ACCEFsternum, quaternionorientationsternum, eulerorientationsternum = extractGaitupData(datasternum, intSternum)

        # static rotation about z-axis of acceleration and gyroscope (and euler orientation?) in sensor frame
        ACCSFleft = staticRotation(ACCSFleft)
        eulerorientationleft = staticRotation(eulerorientationleft)
        ACCEFleft = staticRotation(ACCEFleft)
        GYRleft = staticRotation(GYRleft)
        MAGleft = staticRotation(MAGleft)
        ACCSFright = staticRotation(ACCSFright)
        eulerorientationright = staticRotation(eulerorientationright)
        ACCEFright = staticRotation(ACCEFright)
        GYRright = staticRotation(GYRright)
        MAGright = staticRotation(MAGright)
        ACCSFleftshank = staticRotation(ACCSFleftshank)
        eulerorientationleftshank = staticRotation(eulerorientationleftshank)
        ACCEFleftshank = staticRotation(ACCEFleftshank)
        GYRleftshank = staticRotation(GYRleftshank)
        MAGleftshank = staticRotation(MAGleftshank)
        ACCSFrightshank = staticRotation(ACCSFrightshank)
        eulerorientationrightshank = staticRotation(eulerorientationrightshank)
        ACCEFrightshank = staticRotation(ACCEFrightshank)
        GYRrightshank = staticRotation(GYRrightshank)
        MAGrightshank = staticRotation(MAGrightshank)
        ACCSFlumbar = staticRotation(ACCSFlumbar)
        eulerorientationlumbar = staticRotation(eulerorientationlumbar)
        ACCEFlumbar = staticRotation(ACCEFlumbar)
        GYRlumbar = staticRotation(GYRlumbar)
        MAGlumbar = staticRotation(MAGlumbar)
        ACCSFsternum = staticRotation(ACCSFsternum)
        eulerorientationsternum = staticRotation(eulerorientationsternum)
        ACCEFsternum = staticRotation(ACCEFsternum)
        GYRsternum = staticRotation(GYRsternum)
        MAGsternum = staticRotation(MAGsternum)

    else:
        print('Filetype not recognized')
      
    # Export data in a structured dictionary
    data = {}
    data['Missing Sensors'] = missingsensors
    data['Timestamp'] = timestamp
    data['Sample Frequency (Hz)'] = sample_frequency

    # Left Foot
    data['Left foot'] = dict()
    data['Left foot']['raw'] = dict()
    data['Left foot']['raw']['Accelerometer Earth Frame'] = ACCEFleft
    data['Left foot']['raw']['Orientation Quaternion'] = quaternionorientationleft
    data['Left foot']['raw']['Orientation Euler'] = eulerorientationleft
    data['Left foot']['raw']['Accelerometer Sensor Frame'] = ACCSFleft
    data['Left foot']['raw']['Gyroscope'] = GYRleft
    data['Left foot']['raw']['Magnetometer'] = MAGleft
    data['Left foot']['derived'] = dict()
    # if sensortype == 'gaitup':
    #     data['Left foot']['derived']['Gyroscope Earth frame'] = gyrearthframeGaitup(GYRleft, sample_frequency, datalumbar,intLeft)
    # else:
    data['Left foot']['derived']['Gyroscope Earth frame'] = gyrearthframe(GYRleft, quaternionorientationleft,
                                                                              sample_frequency)
    # Right Foot
    data['Right foot'] = dict()
    data['Right foot']['raw'] = dict()
    data['Right foot']['raw']['Accelerometer Earth Frame'] = ACCEFright
    data['Right foot']['raw']['Orientation Quaternion'] = quaternionorientationright
    data['Right foot']['raw']['Orientation Euler'] = eulerorientationright
    data['Right foot']['raw']['Accelerometer Sensor Frame'] = ACCSFright
    data['Right foot']['raw']['Gyroscope'] = GYRright
    data['Right foot']['raw']['Magnetometer'] = MAGright
    data['Right foot']['derived'] = dict()
    # if sensortype == 'gaitup':
        # data['Right foot']['derived']['Gyroscope Earth frame'] = gyrearthframeGaitup(GYRright, sample_frequency, datalumbar,intRight)
    # else:
    data['Right foot']['derived']['Gyroscope Earth frame'] = gyrearthframe(GYRright, quaternionorientationright,
                                                                               sample_frequency)
    # Left Shank
    data['Left shank'] = dict()
    data['Left shank']['raw'] = dict()
    data['Left shank']['raw']['Accelerometer Earth Frame'] = ACCEFleftshank
    data['Left shank']['raw']['Orientation Quaternion'] = quaternionorientationleftshank
    data['Left shank']['raw']['Orientation Euler'] = eulerorientationleftshank
    data['Left shank']['raw']['Accelerometer Sensor Frame'] = ACCSFleftshank
    data['Left shank']['raw']['Gyroscope'] = GYRleftshank
    data['Left shank']['raw']['Magnetometer'] = MAGleftshank
    data['Left shank']['derived'] = dict()
    data['Left shank']['derived']['Gyroscope Earth frame'] = gyrearthframe(GYRleftshank, quaternionorientationleftshank,
                                                                          sample_frequency)
    # Right Shank
    data['Right shank'] = dict()
    data['Right shank']['raw'] = dict()
    data['Right shank']['raw']['Accelerometer Earth Frame'] = ACCEFrightshank
    data['Right shank']['raw']['Orientation Quaternion'] = quaternionorientationrightshank
    data['Right shank']['raw']['Orientation Euler'] = eulerorientationrightshank
    data['Right shank']['raw']['Accelerometer Sensor Frame'] = ACCSFrightshank
    data['Right shank']['raw']['Gyroscope'] = GYRrightshank
    data['Right shank']['raw']['Magnetometer'] = MAGrightshank
    data['Right shank']['derived'] = dict()
    data['Right shank']['derived']['Gyroscope Earth frame'] = gyrearthframe(GYRrightshank, quaternionorientationrightshank,
                                                                           sample_frequency)
    # Lumbar
    data['Lumbar'] = dict()
    data['Lumbar']['raw'] = dict()
    data['Lumbar']['raw']['Accelerometer Earth Frame'] = ACCEFlumbar
    data['Lumbar']['raw']['Orientation Quaternion'] = quaternionorientationlumbar
    data['Lumbar']['raw']['Orientation Euler'] = eulerorientationlumbar
    data['Lumbar']['raw']['Accelerometer Sensor Frame'] = ACCSFlumbar
    data['Lumbar']['raw']['Gyroscope'] = GYRlumbar
    data['Lumbar']['raw']['Magnetometer'] = MAGlumbar
    data['Lumbar']['derived'] = dict()
    # if sensortype == 'gaitup':
    #     data['Lumbar']['derived']['Gyroscope Earth frame'] = gyrearthframeGaitup(GYRlumbar, sample_frequency, datalumbar,intLumbar)
    # else:
    data['Lumbar']['derived']['Gyroscope Earth frame'] = gyrearthframe(GYRlumbar, quaternionorientationlumbar,
                                                                           sample_frequency)
    # Sternum
    data['Sternum'] = dict()
    data['Sternum']['raw'] = dict()
    data['Sternum']['raw']['Accelerometer Earth Frame'] = ACCEFsternum
    data['Sternum']['raw']['Orientation Quaternion'] = quaternionorientationsternum
    data['Sternum']['raw']['Orientation Euler'] = eulerorientationsternum
    data['Sternum']['raw']['Accelerometer Sensor Frame'] = ACCSFsternum
    data['Sternum']['raw']['Gyroscope'] = GYRsternum
    data['Sternum']['raw']['Magnetometer'] = MAGsternum
    data['Sternum']['derived'] = dict()
    data['Sternum']['derived']['Gyroscope Earth frame'] = gyrearthframe(GYRsternum, quaternionorientationsternum, sample_frequency)
    # if sensortype == 'gaitup':
    #     data['Sternum']['derived']['Gyroscope Earth frame'] = gyrearthframeGaitup(GYRsternum, sample_frequency, datasternum,intSternum)
    return data

# static rotation about z-axis of acceleration and gyroscope in sensor frame
def staticRotation(data_uncalibrated):
    stat_rot = np.array(((np.cos(math.pi), -np.sin(math.pi), 0),
                         (np.sin(math.pi), np.cos(math.pi), 0),
                         (0, 0, 1)))
    
    if len(data_uncalibrated) == 0:
        data_calibrated = data_uncalibrated
    else:
        data_calibrated = np.transpose(stat_rot.dot(np.transpose(data_uncalibrated)))

    return data_calibrated


def gyrearthframeGaitup(gyr, sample_frequency, gaitupData, intSync):
    # Rotate angular velocity to earth frame and extract vertical component
    q = np.array(gaitupData[['Quat W', 'Quat X', 'Quat Y', 'Quat Z']])[intSync]
    # print(np.shape(q))
    if any(np.isnan(q[0, :])):
        q[0, :] = q[1, :]
    zeroesFound = np.all(q == 0, axis=1)
    q = q[~zeroesFound]

    q = q[~(np.isnan(q)).any(axis=1)]

    gyrEF = np.zeros((len(q), 3))
    for i in range(0, len(q)-1):

        qSample = Quaternion(q[i, :])
        gyrEF[i, :] = qSample.rotate(gyr[i, :])

    # Low pass filter
    gyrEF_filt = np.zeros((len(gyrEF), 3))
    fcgyr = 1.5  # Cut-off frequency of the filter for gyroscope data
    wgyr = fcgyr / (sample_frequency / 2)  # Normalize the frequency
    N = 2  # Order of the butterworth filter
    filter_type = 'lowpass'  # Type of the filter
    bgyr, agyr = signal.butter(N, wgyr, filter_type)
    fGYRx = signal.filtfilt(bgyr, agyr, gyrEF[:, 0])
    fGYRy = signal.filtfilt(bgyr, agyr, gyrEF[:, 1])
    fGYRz = signal.filtfilt(bgyr, agyr, gyrEF[:, 2])
    gyrEF_filt[:, 0] = fGYRx
    gyrEF_filt[:, 1] = fGYRy
    gyrEF_filt[:, 2] = fGYRz

    return gyrEF_filt
# Determine Gyroscope data in earth frame
# from pyquaternion import Quaternion
def gyrearthframe(gyr, quat, sample_frequency):
    # Rotate angular velocity to earth frame and extract vertical component
    if len(quat) == 0:
        gyrEF_filt = []
    else:
        gyrEF = np.zeros((len(quat),3))
        for i in range(0, len(quat)):
            q= Quaternion(quat[i,:])
            gyrEF[i,:] = q.rotate(gyr[i,:])
        
        # Low pass filter
        gyrEF_filt = np.zeros((len(gyr),3))
        fcgyr = 1.5  # Cut-off frequency of the filter for gyroscope data
        wgyr = fcgyr / (sample_frequency / 2) # Normalize the frequency
        N = 2 # Order of the butterworth filter
        filter_type = 'lowpass' # Type of the filter
        bgyr, agyr = signal.butter(N, wgyr, filter_type)
        fGYRx = signal.filtfilt(bgyr, agyr, gyrEF[:,0])
        fGYRy = signal.filtfilt(bgyr, agyr, gyrEF[:,1])
        fGYRz = signal.filtfilt(bgyr, agyr, gyrEF[:,2])
        gyrEF_filt[:,0] = fGYRx
        gyrEF_filt[:,1] = fGYRy
        gyrEF_filt[:,2] = fGYRz
    
    return gyrEF_filt

def extractApdmData(apdmData, intSync):
    if len(apdmData) > 0:
        q = apdmData['Orientation'][intSync]
        if any(np.isnan(q[0,:])):
            q[0,:] = q[1,:]
                
        acc = apdmData['Accelerometer'][intSync]
        gyr = apdmData['Gyroscope'][intSync]
        # Check whether this is needed for feet??
        # gyr[:,1] = gyr[:,1]*(-1)
        # gyr[:,2] = gyr[:,2]*(-1)
        mag = apdmData['Magnetometer'][intSync]
        
        nSamples = len(q)
        e = np.zeros((nSamples,3))
        for i in range(0, nSamples):
            r = R.from_quat(q[i,:])
            e[i,:] = r.as_euler('zyx', degrees=True)
        
        acc_ef = np.zeros((nSamples,3))
        for i in range(0, len(q)):
            qSample = Quaternion(q[i,:])
            acc_ef[i,:] = qSample.rotate(acc[i,:])
            
        acc_ef [:,2] = acc_ef[:,2]-9.81
    else:
        acc = np.array([])
        gyr = np.array([])
        mag = np.array([])
        acc_ef = np.array([])
        q = np.array([])
        e = np.array([])
            
    return acc, gyr, mag, acc_ef, q, e
    
def extractGaitupData(gaitupData, intSync):
    if len(gaitupData) > 0:
        q = np.array(gaitupData[['Quat W','Quat X','Quat Y','Quat Z']])[intSync]
        # q[:,0] = -q[:,0]
        # q[:,1] = -q[:,1]
        # q[:, 0] = q[:, 0]
        # q[:, 0] = -q[:, 0]
        # print(np.shape(q))
        if any(np.isnan(q[0,:])):
            q[0,:] = q[1,:]
        zeroesFound = np.all(q == 0, axis=1)
        q = q[~zeroesFound]

        q = q[~(np.isnan(q)).any(axis=1)]
        # print(np.shape(q))
        acc = np.array(gaitupData[['Accel X','Accel Y','Accel Z']])[intSync]
        acc = acc[~zeroesFound]
        acc = acc[~(np.isnan(acc)).any(axis=1)]
        acc = acc*9.81
        gyr = np.array(gaitupData[['Gyro X','Gyro Y','Gyro Z']])[intSync]
        gyr = gyr[~zeroesFound]
        gyr = gyr[~(np.isnan(gyr)).any(axis=1)]
        # Convert to rad/s instead of deg/s
        gyr = gyr/180*np.pi
        # print(np.shape(gyr))
        
        # Check whether this is needed for feet??
        # gyr[:,1] = gyr[:,1]*(-1)
        # gyr[:,2] = gyr[:,2]*(-1)
        
        mag = np.array(gaitupData[['Mag X','Mag Y','Mag Z']])[intSync]
        mag = mag[~(np.isnan(mag)).any(axis=1)]

        nSamples = len(q)
        e = np.zeros((nSamples,3))
        for i in range(0, nSamples):
            r = R.from_quat(q[i,:])

            e[i,:] = r.as_euler('xyz', degrees=True) #was zyx, changed JO 21/07
        
        acc_ef = np.zeros((nSamples,3))
        for i in range(0, len(q)):
            qSample = Quaternion(q[i,:])
            acc_ef[i,:] = qSample.rotate(acc[i,:])

        acc_ef [:,2] = acc_ef[:,2]-9.81
    else:
        acc = np.array([])
        gyr = np.array([])
        mag = np.array([])
        acc_ef = np.array([])
        q = np.array([])
        e = np.array([])
            
    return acc, gyr, mag, acc_ef, q, e
    
def syncData(sensor_type, dataleft, dataright, datalumbar, datasternum):
    
    syncFailed = False
    intLeft = intRight = intLeftShank = intRightShank = intLumbar = intSternum = tVec = []
    sample_frequency = 0

    tStart = []
    tEnd = []
    
    if len(dataleft) > 0:
        if sensor_type == 'gaitup':
            tLeft = np.array(np.squeeze(dataleft[['Time']]))
            # tLeft = tLeft.astype(np.float)
        elif sensor_type == 'apdm':
            tLeft = dataleft['Time']
        tStart.append(tLeft[0])
        tEnd.append(tLeft[-1])
    else:
        # We assume to always have left, return error
        tLeft = []
        syncFailed = True
        
    if len(dataright) > 0:
        if sensor_type == 'gaitup':
            tRight = np.array(np.squeeze(dataright[['Time']]))
        elif sensor_type == 'apdm':
            tRight = dataright['Time']
        tStart.append(tRight[0])
        tEnd.append(tRight[-1])
    else:
        # Copy left for dummy data
        tRight = tLeft

    if len(dataleftshank) > 0:
        if sensor_type == 'gaitup':
            tLeftShank = np.array(np.squeeze(dataleftshank[['Time']]))
        elif sensor_type == 'apdm':
            tLeftShank = dataleftshank['Time']
        tStart.append(tLeftShank[0])
        tEnd.append(tLeftShank[-1])
    else:
        # Copy left for dummy data
        tLeftShank = tLeft

    if len(datarightshank) > 0:
        if sensor_type == 'gaitup':
            tRightShank = np.array(np.squeeze(datarightshank[['Time']]))
        elif sensor_type == 'apdm':
            tRightShank = datarightshank['Time']
        tStart.append(tRightShank[0])
        tEnd.append(tRightShank[-1])
    else:
        # Copy left for dummy data
        tRightShank = tLeft
        
    if len(datalumbar) > 0:
        if sensor_type == 'gaitup':
            tLumbar = np.array(datalumbar['Time'].astype(float))
        elif sensor_type == 'apdm':
            tLumbar = datalumbar['Time']
        
        tStart.append(tLumbar[0])
        tEnd.append(tLumbar[-1])
    else:
        # Copy left for dummy data
        tLumbar = tLeft
        
    if len(datasternum) > 0:
        if sensor_type == 'gaitup':
            tSternum = np.array(datasternum['Time'].astype(float))
        elif sensor_type == 'apdm':
            tSternum = datasternum['Time']
        
        tStart.append(tSternum[0])
        tEnd.append(tSternum[-1])
    else:
        # Copy left for dummy data
        tSternum = tLeft

    if syncFailed == False:
    
        tStartMax = max(tStart)
        tEndMin = min(tEnd)

        iLeftStart = np.flatnonzero(tLeft >= tStartMax)[0]
        iLeftEnd = np.flatnonzero(tLeft <= tEndMin)[-1]
        iRightStart = np.flatnonzero(tRight >= tStartMax)[0]
        iRightEnd = np.flatnonzero(tRight <= tEndMin)[-1]
        iLeftShankStart = np.flatnonzero(tLeftShank >= tStartMax)[0]
        iLeftShankEnd = np.flatnonzero(tLeftShank <= tEndMin)[-1]
        iRightShankStart = np.flatnonzero(tRightShank >= tStartMax)[0]
        iRightShankEnd = np.flatnonzero(tRightShank <= tEndMin)[-1]
        iLumbarStart = np.flatnonzero(tLumbar >= tStartMax)[0]
        iLumbarEnd = np.flatnonzero(tLumbar <= tEndMin)[-1]
        iSternumStart = np.flatnonzero(tSternum >= tStartMax)[0]
        iSternumEnd = np.flatnonzero(tSternum <= tEndMin)[-1]

        intLeft = range(iLeftStart, iLeftEnd)
        intRight = range(iRightStart, iRightEnd)
        intLeftShank = range(iLeftShankStart, iLeftShankEnd)
        intRightShank = range(iRightShankStart, iRightShankEnd)
        intLumbar = range(iLumbarStart, iLumbarEnd)
        intSternum = range(iSternumStart, iSternumEnd)
        
        # Cut to same length
        nSamples = min([len(intLeft), len(intRight), len(intLeftShank), len(intRightShank), len(intLumbar), len(intSternum)])
        intLeft = intLeft[0:nSamples]
        intRight = intRight[0:nSamples]
        intLeftShank = intLeftShank[0:nSamples]
        intRightShank = intRightShank[0:nSamples]
        intLumbar = intLumbar[0:nSamples]
        intSternum = intSternum[0:nSamples]
        
        # sync
        if len(dataleft) == 0:
            intLeft = []
        if len(dataright) == 0:
            intRight = []
        if len(dataleftshank) == 0:
            intLeftShank = []
        if len(datarightshank) == 0:
            intRightShank = []
        if len(datalumbar) == 0:
            intLumbar = []
        if len(datasternum) == 0:
            intSternum = []
            
        tVec = tLeft[intLeft]
        tVec = tVec - tVec[0]
        if sensor_type == 'apdm':
            # convert us to s
            tVec = tVec/1e6
            
        sample_frequency = np.int(1/np.nanmean(np.diff(tVec)))
        if sensor_type == 'gaitup':
            sample_frequency = 128
        
    return syncFailed, intLeft, intRight, intLeftShank, intRightShank, intLumbar, intSternum, tVec, sample_frequency
