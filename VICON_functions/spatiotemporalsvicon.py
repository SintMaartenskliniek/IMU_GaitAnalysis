# -*- coding: utf-8 -*-
"""
Spatiotemporals from VICON data

@author: ensinkc
"""

import numpy as np
from scipy import signal

def spatiotemporalsGRAIL(markerdatavicon, gait_events, videoframerate):
    HSL = gait_events['Index numbers heel strike left']
    TOL = gait_events['Index numbers toe off left']
    HSR = gait_events['Index numbers heel strike right']
    TOR = gait_events['Index numbers toe off right']
    
    # Stance time = time from heel strike of one foot till toe off of the same foot
    StTL = np.array([])
    for i in range(0,len(HSL)):
        firstTO = np.argwhere(TOL > HSL[i])
        if len(firstTO) > 0:
            firstTO = np.argwhere(TOL > HSL[i])[0]
            if TOL[firstTO] - HSL[i] > 1.8*videoframerate:
                StTL = np.append(StTL, np.nan)
            else:
                StTL = np.append(StTL, TOL[firstTO] - HSL[i])
    StTR = np.array([])
    for i in range(0,len(HSR)):
        firstTO = np.argwhere(TOR > HSR[i])
        if len(firstTO) > 0:
            firstTO = np.argwhere(TOR > HSR[i])[0]
            if TOR[firstTO] - HSR[i] > 1.8*videoframerate:
                StTR = np.append(StTR, np.nan)
            else:
                StTR = np.append(StTR, TOR[firstTO] - HSR[i])
    
    # Swing time = time from toe off of one foot till heel strike of the same foot
    SwTL = np.array([])
    for i in range(0,len(TOL)):
        firstHS = np.argwhere(HSL > TOL[i])
        if len(firstHS) > 0:
            firstHS = np.argwhere(HSL > TOL[i])[0]
            if HSL[firstHS] - TOL[i] > 1.8*videoframerate:
                SwTL = np.append(SwTL, np.nan)
            else:
                SwTL = np.append(SwTL, HSL[firstHS] - TOL[i])
    SwTR = np.array([])
    for i in range(0,len(TOR)):
        firstHS = np.argwhere(HSR > TOR[i])
        if len(firstHS) > 0:
            firstHS = np.argwhere(HSR > TOR[i])[0]
            if HSR[firstHS] - TOR[i] > 1.8*videoframerate:
                SwTR = np.append(SwTR, np.nan)
            else:
                SwTR = np.append(SwTR, HSR[firstHS] - TOR[i])
    
    
    # Time interval for foot flat phase
    # valsleft = np.array(range(np.round(0.20*np.nanmean(StTL)).astype(int) , np.round(0.50*np.nanmean(StTL)).astype(int)))
    # valsright = np.array(range(np.round(0.20*np.nanmean(StTR)).astype(int) , np.round(0.50*np.nanmean(StTR)).astype(int)))
    valsleft = np.array(range(np.round(0.30*np.nanmean(StTL)).astype(int) , np.round(0.50*np.nanmean(StTL)).astype(int)))
    valsright = np.array(range(np.round(0.30*np.nanmean(StTR)).astype(int) , np.round(0.50*np.nanmean(StTR)).astype(int)))
    
    # velocity = differentiated markerdatavicon['LANK'], at assumed foot flat phase.
    # this is assumed to be the velocity of the treadmill and per definition the walkingspeed
    velocity_left = np.zeros(len(markerdatavicon['LANK']))*np.nan
    for i in range(0, len(HSL)-1):
        velocity_left[HSL[i] + valsleft[0:-1]] = np.diff(markerdatavicon['LANK'][HSL[i]+valsleft,1])*videoframerate
    velocity_right = np.zeros(len(markerdatavicon['RANK']))*np.nan
    for i in range(0, len(HSR)-1):
        velocity_right[HSR[i] + valsright[0:-1]] = np.diff(markerdatavicon['RANK'][HSR[i]+valsright,1])*videoframerate
    
    # Convert mm/s to m/s
    velocity_left = velocity_left/1000
    velocity_right = velocity_right/1000
    
    # Steplength
    # Position difference between heel strike of one and the other foot + time difference*velocity of the treadmill
    # treadmill 25-50% stancephase
    
    # Set second-order low-pass butterworth filter;
    fc1 = 5  # Cut-off frequency of the first low pass filter
    wn1 = fc1 / (videoframerate / 2) # Normalize the frequency
    fc2 = 8 # Cut-off frequency of the second low pass filter
    wn2 = fc2 / (videoframerate / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    B1, A1 = signal.butter(N, wn1, filter_type) # First low-pass filterdesign
    B2, A2 = signal.butter(N, wn2, filter_type) # Second low-pass filterdesign
    
    vlank = {}
    vrank = {}
    # Apply low pass filter on data and calculate velocity
    vlank['vlankx'] = signal.filtfilt(B1, A1, markerdatavicon['LANK'][:,0])
    vlank['vlanky'] = signal.filtfilt(B1, A1, markerdatavicon['LANK'][:,1])
    vlank['vlankz'] = signal.filtfilt(B1, A1, markerdatavicon['LANK'][:,2])
    
    vrank['vrankx'] = signal.filtfilt(B1, A1, markerdatavicon['RANK'][:,0])
    vrank['vranky'] = signal.filtfilt(B1, A1, markerdatavicon['RANK'][:,1])
    vrank['vrankz'] = signal.filtfilt(B1, A1, markerdatavicon['RANK'][:,2])
    
    vlank['vlankx2'] = signal.filtfilt(B2, A2, np.diff(vlank['vlankx']))*videoframerate
    vlank['vlanky2'] = signal.filtfilt(B2, A2, np.diff(vlank['vlanky']))*videoframerate
    vlank['vlankz2'] = signal.filtfilt(B2, A2, np.diff(vlank['vlankz']))*videoframerate
    
    vrank['vrankx2'] = signal.filtfilt(B2, A2, np.diff(vrank['vrankx']))*videoframerate
    vrank['vranky2'] = signal.filtfilt(B2, A2, np.diff(vrank['vranky']))*videoframerate
    vrank['vrankz2'] = signal.filtfilt(B2, A2, np.diff(vrank['vrankz']))*videoframerate
    
    # Back to one matrix
    vlank['vlank'] = np.swapaxes(np.vstack((vlank['vlankx2'],vlank['vlanky2'], vlank['vlankz2'])), 1, 0)
    vrank['vrank'] = np.swapaxes(np.vstack((vrank['vrankx2'],vrank['vranky2'], vrank['vrankz2'])), 1, 0)
    vlank = vlank['vlank']
    vrank = vrank['vrank']
    
    
    # # Stride lengths
    # # Position difference between two heel strikes of one foot + time difference*velocity of the treadmill
    # stridelengths_left = np.zeros((len(HSL),3))*np.nan
    # treadmill_left = np.zeros(len(HSL))*np.nan
    # foot_left = np.zeros((len(HSL),3))*np.nan
    # for i in range(0, len(HSL)):
    #     start = TOL[TOL<HSL[i]][-1]
    #     stop = HSL[i]
    #     duration = stop-start
    #     start_later = int(start+0.1*duration) #0.1
    #     stop_before = int(start+0.6*duration) #0.3
    #     duration_ff = stop_before-start_later
    #     treadmill_left[i] = (duration/videoframerate) * np.nanmean(vrank[start_later : stop_before,1]) # assumed during left swing the right foot is at the treadmill
    #     foot_left[i,:] = np.abs(markerdatavicon['LTOE'][HSL[i]] - markerdatavicon['LTOE'][TOL[TOL<HSL[i]][-1]])
    #     # foot_left[i,:] = np.abs(markerdatavicon['LTOE'][HSL[i]] - markerdatavicon['LTOE'][TOL[TOL<HSL[i]][-1]]) #LHEE
    #     stridelengths_left[i,0] = start #TOL[TOL<HSL[i]][-1]
    #     stridelengths_left[i,1] = stop #HSL[i]
    #     stridelengths_left[i,2] = treadmill_left[i] + foot_left[i,1] #np.sqrt(foot_left[i,0]**2 + foot_left[i,1]**2)
    
    # for i in range(0, len(stridelengths_left)):
    #     if stridelengths_left[i,2] > 2000 or stridelengths_left[i,2]<10 or stridelengths_left[i,1]-stridelengths_left[i,0] > 1.5*np.median(SwTL):
    #         stridelengths_left[i,:] = np.nan
    # stridelengths_left[~np.isnan(stridelengths_left).any(axis=1), :]
    
    # stridelengths_right = np.zeros((len(HSR),3))*np.nan
    # treadmill_right = np.zeros(len(HSR))*np.nan
    # foot_right = np.zeros((len(HSR),3))*np.nan
    # for i in range(0, len(HSR)):
    #     start = TOR[TOR<HSR[i]][-1]
    #     stop = HSR[i]
    #     duration = stop-start
    #     start_later = int(start+0.1*duration)
    #     stop_before = int(start+0.6*duration)
    #     treadmill_right[i] = (duration/videoframerate) * np.nanmean(vlank[start_later : stop_before,1]) # assumed during left swing the right foot is at the treadmill
    #     foot_right[i,:] = np.abs(markerdatavicon['RTOE'][HSR[i]] - markerdatavicon['RTOE'][TOR[TOR<HSR[i]][-1]])
    #     # foot_right[i,:] = np.abs(markerdatavicon['RTOE'][HSR[i]] - markerdatavicon['RTOE'][TOR[TOR<HSR[i]][-1]]) #RHEE
    #     stridelengths_right[i,0] = start #TOR[TOR<HSR[i]][-1]
    #     stridelengths_right[i,1] = stop #HSR[i]
    #     stridelengths_right[i,2] = treadmill_right[i] + foot_right[i,1] #np.sqrt(foot_right[i,0]**2 + foot_right[i,1]**2)
    
    # for i in range(0, len(stridelengths_right)):
    #     if stridelengths_right[i,2] > 2000 or stridelengths_right[i,2] <10  or stridelengths_right[i,1]-stridelengths_right[i,0] > 1.5*np.median(SwTR):
    #         stridelengths_right[i,:] = np.nan
    # stridelengths_right[~np.isnan(stridelengths_right).any(axis=1), :]
    
    # Stride lengths
    # Position difference between two heel strikes of one foot + time difference*velocity of the treadmill
    stridelengths_left = np.zeros((len(HSL),3))*np.nan
    treadmill_left = np.zeros(len(HSL))*np.nan
    foot_left = np.zeros((len(HSL),3))*np.nan
    for i in range(1, len(HSL)):
        start_stride = HSL[HSL<HSL[i]][-1]
        stop_stride = HSL[i]
        duration_stride = stop_stride-start_stride
        
        start_swing = TOL[TOL<HSL[i]][-1]
        duration_swing = stop_stride-start_swing
        start_ff = int(start_swing+0.1*duration_swing)
        stop_ff = int(start_swing+0.6*duration_swing)
        duration_ff = stop_ff-start_ff
        treadmill_left[i] = (duration_stride/videoframerate) * np.nanmean(vrank[start_ff : stop_ff,1]) # assumed during left swing the right foot is at the treadmill
        # treadmill_left[i] = (markerdatavicon['RTOE'][stop_ff,1]-markerdatavicon['RTOE'][start_ff,1])/(duration_ff/videoframerate)
        foot_left[i,:] = markerdatavicon['LHEE'][stop_stride] - markerdatavicon['LHEE'][start_stride]  # np.sqrt((markerdatavicon['LHEE'][stop_stride,0] - markerdatavicon['LHEE'][start_stride,0])**2+(markerdatavicon['LHEE'][stop_stride,1] - markerdatavicon['LHEE'][start_stride,1])**2)
        # foot_left[i,:] = np.abs(markerdatavicon['LHEE'][HSL[i]] - markerdatavicon['LHEE'][TOL[TOL<HSL[i]][-1]])
        stridelengths_left[i,0] = start_swing #TOL[TOL<HSL[i]][-1]
        stridelengths_left[i,1] = stop_stride #HSL[i]
        stridelengths_left[i,2] = treadmill_left[i] - foot_left[i,1]
        # stridelengths_left[i,3] = treadmill_left[i]
        # stridelengths_left[i,4] = start_stride
        
    
    mslL = np.nanmedian(stridelengths_left[:,2])
    for i in range(0, len(stridelengths_left)):
        if stridelengths_left[i,2] > 1.6*mslL or stridelengths_left[i,2] < 200 or stridelengths_left[i,1]-stridelengths_left[i,0] > 1.5*np.median(SwTL):
            stridelengths_left[i,:] = np.nan
    stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1), :]
    
    stridelengths_right = np.zeros((len(HSR),3))*np.nan
    treadmill_right = np.zeros(len(HSR))*np.nan
    foot_right = np.zeros((len(HSR),3))*np.nan
    for i in range(1, len(HSR)):
        start_stride = HSR[HSR<HSR[i]][-1]
        stop_stride = HSR[i]
        duration_stride = stop_stride-start_stride
        
        start_swing = TOR[TOR<HSR[i]][-1]
        duration_swing = stop_stride-start_swing
        start_ff = int(start_swing+0.1*duration_swing)
        stop_ff = int(start_swing+0.6*duration_swing)
        duration_ff = stop_ff-start_ff
        treadmill_right[i] = (duration_stride/videoframerate) * np.nanmean(vlank[start_ff : stop_ff,1]) # assumed during left swing the right foot is at the treadmill
        # treadmill_right[i] = (markerdatavicon['LTOE'][stop_ff,1]-markerdatavicon['LTOE'][start_ff,1])/(duration_ff/videoframerate)
        foot_right[i,:] = markerdatavicon['RHEE'][stop_stride] - markerdatavicon['RHEE'][start_stride] #RHEE
        stridelengths_right[i,0] = start_swing #TOR[TOR<HSR[i]][-1]
        stridelengths_right[i,1] = stop_stride #HSR[i]
        stridelengths_right[i,2] = treadmill_right[i] - foot_right[i,1] #np.sqrt(foot_right[i,0]**2 + foot_right[i,1]**2))
        # stridelengths_right[i,3] = treadmill_right[i]
        # stridelengths_right[i,4] = start_stride
        
    mslR = np.nanmedian(stridelengths_right[:,2])
    for i in range(0, len(stridelengths_right)):
        if stridelengths_right[i,2] > 1.6*mslR or stridelengths_right[i,2] < 200 or stridelengths_right[i,1]-stridelengths_right[i,0] > 1.5*np.median(SwTR):
            stridelengths_right[i,:] = np.nan
    stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1), :]
    
    # Gait cycle duration = time from heel strike till heel strike of the same foot
    # In case of walking outside measuremnent volume, two consecutive IC's are not actually consecutive> correct by assuming a gait cycle duration has to be smaller than 2 seconds.
    GCDL = np.zeros((len(stridelengths_left),3))*np.nan
    GCDL[:,0] = stridelengths_left[:,0]
    GCDL[:,1] = stridelengths_left[:,1]
    # GCDL[:,2] = GCDL[:,1]-GCDL[:,0]
    GCDL[1:,2] = np.diff(stridelengths_left[:,1])
    GCDL[GCDL[:,2]>3*videoframerate,:] = np.nan
    GCDL[GCDL[:,2]<0.3*videoframerate,:] = np.nan
    GCDL[GCDL[:,2]>1.5*np.nanmedian(GCDL[:,2]),:] = np.nan
    
    GCDR = np.zeros((len(stridelengths_right),3))*np.nan
    GCDR[:,0] = stridelengths_right[:,0]
    GCDR[:,1] = stridelengths_right[:,1]
    # GCDR[:,2] = GCDR[:,1]-GCDR[:,0]
    GCDR[1:,2] = np.diff(stridelengths_right[:,1])
    GCDR[GCDR[:,2]>3*videoframerate,:] = np.nan
    GCDR[GCDR[:,2]<0.3*videoframerate,:] = np.nan    
    GCDR[GCDR[:,2]>1.5*np.nanmedian(GCDR[:,2]),:] = np.nan
    
    # Velocity per stride
    velocity_stridesleft = np.zeros((len(stridelengths_left),3))*np.nan
    velocity_stridesleft[:,0] = stridelengths_left[:,0]
    velocity_stridesleft[:,1] = stridelengths_left[:,1]
    velocity_stridesleft[:,2] = (stridelengths_left[:,2]/1000) / ((GCDL[:,2])/videoframerate)
    velocity_stridesright = np.zeros((len(stridelengths_right),3))*np.nan
    velocity_stridesright[:,0] = stridelengths_right[:,0]
    velocity_stridesright[:,1] = stridelengths_right[:,1]
    velocity_stridesright[:,2] = (stridelengths_right[:,2]/1000) / ((GCDR[:,2])/videoframerate)
    # velocity_stridesleft = (stridelengths_left[:,2]/1000) / (GCDL[:,2]/videoframerate)
    # velocity_stridesright = (stridelengths_right[:,2]/1000) / (GCDR[:,2]/videoframerate)
    
    # # Step lengths and stepwidths
    # # Position difference between heel strikes of one foot and heel strike of other foot + time difference*velocity of the treadmill
    # steplengths_left = np.zeros((len(HSL),2))*np.nan
    # stepwidths_left = np.zeros((len(HSL),2))*np.nan
    # # steptime_left = np.zeros(len(HSL))*np.nan
    # # treadmill_left = np.zeros(len(HSL))*np.nan
    # foot_left = np.zeros((len(HSL),3))*np.nan
    # for i in range(0, len(HSL)):
    #     # treadmill_left[i] = (HSL[i] - TOL[i])/videoframerate * np.nanmean(vrank[TOL[i] : HSL[i],1]) # assumed during left swing the right foot is at the treadmill
    #     foot_left[i,:] = markerdatavicon['LHEE'][HSL[i]] - markerdatavicon['RHEE'][HSL[i]] # forward swing is in opposite direction of treadmill
    #     steplengths_left[i,0] = HSL[i]
    #     stepwidths_left[i,0] = HSL[i]
    #     steplengths_left[i,1] = np.abs(foot_left[i,1])
    #     stepwidths_left[i,1] = np.abs(foot_left[i,0])
    #     # if i == 0 and HSL[0]<HSR[0]:
    #     #     steptime_left[i] = HSL[i]-TOL[i]
    #     # elif i == 0 and HSL[0]>HSR[0]:
    #     #     steptime_left[i] = HSL[i]-HSR[HSR<HSL[i]][-1]
    #     # if i > 0:
    #     #     if (HSL[i] - HSR[HSR<HSL[i]][-1]) < 1.5*videoframerate:
    #     #         steptime_left[i] = HSL[i]-HSR[HSR<HSL[i]][-1]
    #     #     else:
    #     #         steptime_left[i] = np.nan
    
    # steplengths_right = np.zeros((len(HSR),2))*np.nan
    # stepwidths_right = np.zeros((len(HSR),2))*np.nan
    # # steptime_right = np.zeros(len(TOR))*np.nan
    # # treadmill_right = np.zeros(len(TOR))*np.nan
    # foot_right = np.zeros((len(HSR),3))*np.nan
    # for i in range(0, len(HSR)):
    # #     # treadmill_right[i] = (HSR[i] - TOR[i])/videoframerate * np.nanmean(vlank[TOR[i] : HSR[i],1]) # assumed during left swing the right foot is at the treadmill
    #     foot_right[i,:] = markerdatavicon['RHEE'][HSR[i]] - markerdatavicon['LHEE'][HSR[i]] # forward swing is in opposite direction of treadmill
    #     steplengths_right[i,0] = HSR[i]
    #     stepwidths_right[i,0] = HSR[i]
    #     steplengths_right[i,1] = np.abs(foot_right[i,1])
    #     stepwidths_right[i,1] = np.abs(foot_right[i,0])
    # #     if i == 0 and HSR[0]<HSL[0]:
    # #         steptime_right[i] = HSR[i]-TOR[i]
    # #     elif i == 0 and HSR[0]>HSL[0]:
    # #         steptime_right[i] = HSR[i]-HSL[HSL<HSR[i]][-1]
    # #     if i > 0:
    # #         if (HSR[i] - HSL[HSL<HSR[i]][-1]) < 1.5*videoframerate:
    # #             steptime_right[i] = HSR[i]-HSL[HSL<HSR[i]][-1]
    # #         else:
    # #             steptime_right[i] = np.nan
    

  
    
    # Create output dict
    spatiotemporals = {}
    spatiotemporals['Gait speed left (m/s)'] = (velocity_left) #np.nanmean
    spatiotemporals['Gait speed right (m/s)'] = (velocity_right) #np.nanmean
    spatiotemporals['Velocity left (m/s)'] = np.diff(markerdatavicon['LANK'][:,1])*videoframerate/1000
    spatiotemporals['Velocity right (m/s)'] = np.diff(markerdatavicon['RANK'][:,1])*videoframerate/1000
    spatiotemporals['Gait speed left strides (m/s)'] = velocity_stridesleft
    spatiotemporals['Gait speed right strides (m/s)'] = velocity_stridesright
    spatiotemporals['Gait Cycle duration left (s)'] = GCDL
    spatiotemporals['Gait Cycle duration left (s)'][:,2] = spatiotemporals['Gait Cycle duration left (s)'][:,2]/videoframerate
    spatiotemporals['Gait Cycle duration right (s)'] = GCDR
    spatiotemporals['Gait Cycle duration right (s)'][:,2] = spatiotemporals['Gait Cycle duration right (s)'][:,2]/videoframerate
    spatiotemporals['Stance time left (s)'] = StTL/videoframerate
    spatiotemporals['Stance time right (s)'] = StTR/videoframerate
    spatiotemporals['Swing time left (s)'] = SwTL/videoframerate
    spatiotemporals['Swing time right (s)'] = SwTR/videoframerate
    # spatiotemporals['Steplength left (mm)'] = steplengths_left
    # spatiotemporals['Steplength right (mm)'] = steplengths_right
    # spatiotemporals['Stepwidth left (mm)'] = stepwidths_left
    # spatiotemporals['Stepwidth right(mm)'] = stepwidths_right
    # spatiotemporals['Step time left (s)'] = steptime_left/videoframerate
    # spatiotemporals['Step time right (s)'] = steptime_right/videoframerate
    spatiotemporals['Stridelength left (mm)'] = stridelengths_left
    spatiotemporals['Stridelength right (mm)'] = stridelengths_right
    
    return spatiotemporals



def spatiotemporalsGBA(markerdatavicon, gait_events, videoframerate):
    HSL = gait_events['Index numbers heel strike left']
    TOL = gait_events['Index numbers toe off left']
    HSR = gait_events['Index numbers heel strike right']
    TOR = gait_events['Index numbers toe off right']
    
   
    # Stride lengths
    # Position difference between two heel strikes of one foot
    # stridelengths_left = np.zeros((len(TOL),3))*np.nan
    # foot_left = np.zeros((len(TOL),3))*np.nan
    # for i in range(0, len(TOL)):
    #     if len(HSL[HSL>TOL[i]]) >0:
    #         foot_left[i,:] = np.abs(markerdatavicon['LTOE'][HSL[HSL>TOL[i]][0]] - markerdatavicon['LTOE'][TOL[i]])
    #         stridelengths_left[i,0] = TOL[i]
    #         stridelengths_left[i,1] = HSL[HSL>TOL[i]][0]
    #         stridelengths_left[i,2] = np.sqrt(foot_left[i,0]**2 + foot_left[i,1]**2) # x-axis in length of GBA lab
    # stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1)]
    # for i in range(0,len(stridelengths_left[:,0])):
    #     if (stridelengths_left[i,1] > (stridelengths_left[i,0] + 1.5*videoframerate)) or (stridelengths_left[i,1] < (stridelengths_left[i,0] + 0.3*videoframerate)) or stridelengths_left[i,2] > 2000 or stridelengths_left[i,2] <10  : # assumed no true stride if stridetime exceeds 1.5 second or smaller than 0.3 seconds
    #         stridelengths_left[i,:] = np.nan
    # stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1)]
    # for i in range(0, len(stridelengths_left[:,0])):
    #     if (np.all(markerdatavicon['LTOE'][int(stridelengths_left[i,0]),:] == [0,0,0])) or (np.all(markerdatavicon['LTOE'][int(stridelengths_left[i,1]),:] == [0,0,0])):
    #         stridelengths_left[i,:] = np.nan
    # stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1)]
    
    # stridelengths_right = np.zeros((len(TOR),3))*np.nan
    # foot_right = np.zeros((len(TOR),3))*np.nan
    # for i in range(0, len(TOR)):
    #     if len(HSR[HSR>TOR[i]]) > 0:
    #         foot_right[i,:] = np.abs(markerdatavicon['RTOE'][HSR[HSR>TOR[i]][0]] - markerdatavicon['RTOE'][TOR[i]])
    #         stridelengths_right[i,0] = TOR[i]
    #         stridelengths_right[i,1] = HSR[HSR>TOR[i]][0]
    #         stridelengths_right[i,2] = np.sqrt(foot_right[i,0]**2 +foot_right[i,1]**2) # x-axis in length of GBA lab
    # stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1)]
    # for i in range(0,len(stridelengths_right[:,0])):
    #     if (stridelengths_right[i,1] > (stridelengths_right[i,0] + 1.5*videoframerate)) or (stridelengths_right[i,1] < (stridelengths_right[i,0] + 0.3*videoframerate)) or stridelengths_right[i,2] > 2000 or stridelengths_right[i,2] <10 : # assumed no true stride if swingtime exceeds 1.5 second or smaller than 0.3 seconds
    #         stridelengths_right[i,:] = np.nan
    # stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1)]
    # for i in range(0,len(stridelengths_right[:,0])):
    #     if (np.all(markerdatavicon['RTOE'][int(stridelengths_right[i,0]),:] == [0,0,0])) or (np.all(markerdatavicon['RTOE'][int(stridelengths_right[i,1]),:] == [0,0,0])):
    #         stridelengths_right[i,:] = np.nan
    # stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1)]
    
    # Stride lengths
    # Position difference between two heel strikes of one foot + time difference*velocity of the treadmill
    stridelengths_left = np.zeros((len(HSL),4))*np.nan
    foot_left = np.zeros((len(HSL),3))*np.nan
    for i in range(1, len(HSL)):
        start_stride = HSL[HSL<HSL[i]][-1]
        stop_stride = HSL[i]
        duration_stride = stop_stride-start_stride
        start_swing = TOL[TOL<HSL[i]][-1]
        # duration_swing = stop_stride-start_swing
        # start_ff = int(start_swing+0.1*duration_swing)
        # stop_ff = int(start_swing+0.6*duration_swing)
        # duration_ff = stop_ff-start_ff
        foot_left[i,:] = markerdatavicon['LHEE'][stop_stride] - markerdatavicon['LHEE'][start_stride]
        stridelengths_left[i,0] = start_swing
        stridelengths_left[i,1] = stop_stride
        stridelengths_left[i,2] = np.abs(foot_left[i,0])
        stridelengths_left[i,3] = start_stride
    
    uni, idx, counts = np.unique(stridelengths_left[:,3], return_index=True, return_counts=True)
    stridelengths_left = stridelengths_left[idx[counts<2],:]
    uni, idx, counts = np.unique(stridelengths_left[:,0],return_index=True, return_counts=True)
    stridelengths_left = stridelengths_left[idx[counts<2],:]
    
    for i in range(0, len(stridelengths_left)):
        if stridelengths_left[i,2] > 1800 or stridelengths_left[i,2] < 200:
            stridelengths_left[i,:] = np.nan
    mslL = np.nanmedian(stridelengths_left[:,2])
    for i in range(0, len(stridelengths_left)):
        if stridelengths_left[i,2] > 1.6*mslL:
            stridelengths_left[i,:] = np.nan
    stridelengths_left = stridelengths_left[~np.isnan(stridelengths_left).any(axis=1), :]
    
    stridelengths_right = np.zeros((len(HSR),4))*np.nan
    foot_right = np.zeros((len(HSR),3))*np.nan
    for i in range(1, len(HSR)):
        start_stride = HSR[HSR<HSR[i]][-1]
        stop_stride = HSR[i]
        duration_stride = stop_stride-start_stride
        start_swing = TOR[TOR<HSR[i]][-1]
        # duration_swing = stop_stride-start_swing
        # start_ff = int(start_swing+0.1*duration_swing)
        # stop_ff = int(start_swing+0.6*duration_swing)
        # duration_ff = stop_ff-start_ff
        foot_right[i,:] = markerdatavicon['RHEE'][stop_stride] - markerdatavicon['RHEE'][start_stride] #RHEE
        stridelengths_right[i,0] = start_swing
        stridelengths_right[i,1] = stop_stride
        stridelengths_right[i,2] = np.abs(foot_right[i,0])
        stridelengths_right[i,3] = start_stride
    
    uni, idx, counts = np.unique(stridelengths_right[:,3], return_index=True, return_counts=True)
    stridelengths_right = stridelengths_right[idx[counts<2],:]
    uni, idx, counts = np.unique(stridelengths_right[:,0], return_index=True, return_counts=True)
    stridelengths_right = stridelengths_right[idx[counts<2],:]
    
    for i in range(0, len(stridelengths_right)):
        if stridelengths_right[i,2] > 1800 or stridelengths_right[i,2] < 200:
            stridelengths_right[i,:] = np.nan
    mslR = np.nanmedian(stridelengths_right[:,2])
    for i in range(0, len(stridelengths_right)):
        if stridelengths_right[i,2] > 1.6*mslR:
            stridelengths_right[i,:] = np.nan
    stridelengths_right = stridelengths_right[~np.isnan(stridelengths_right).any(axis=1), :]

      
    # Stance time = time from heel strike of one foot till toe off of the same foot
    StTL = np.zeros((len(stridelengths_left),3))*np.nan
    StTL[:,0] = stridelengths_left[:,0]
    StTL[:,1] = stridelengths_left[:,1]
    for i in range(1, len(stridelengths_left[:,0])):
        StTL[i,2] = stridelengths_left[i,0] - stridelengths_left[i-1,1]
    
    StTR = np.zeros((len(stridelengths_right),3))*np.nan
    StTR[:,0] = stridelengths_right[:,0]
    StTR[:,1] = stridelengths_right[:,1]
    for i in range(1, len(stridelengths_right[:,0])):
        StTR[i,2] = stridelengths_right[i,0] - stridelengths_right[i-1,1]
    
    
    # Swing time = time from toe off of one foot till heel strike of the same foot
    SwTL = np.zeros((len(stridelengths_left),3))*np.nan
    SwTL[:,0] = stridelengths_left[:,0]
    SwTL[:,1] = stridelengths_left[:,1]
    SwTL[:,2] = stridelengths_left[:,1] - stridelengths_left[:,0]
    
    SwTR = np.zeros((len(stridelengths_right),3))*np.nan
    SwTR[:,0] = stridelengths_right[:,0]
    SwTR[:,1] = stridelengths_right[:,1]
    SwTR[:,2] = stridelengths_right[:,1] - stridelengths_right[:,0]
    
    
    # Gait cycle duration = time from heel strike till heel strike of the same foot
    GCDL = np.zeros((len(stridelengths_left),3))*np.nan
    GCDL[:,0] = stridelengths_left[:,3]
    GCDL[:,1] = stridelengths_left[:,1]
    GCDL[:,2] = stridelengths_left[:,1]-stridelengths_left[:,3]
    # GCDL[1:,2] = np.diff(stridelengths_left[:,1])
    GCDL[GCDL[:,2]>2.3*videoframerate,:] = np.nan
    GCDL[GCDL[:,2]<0.3*videoframerate,:] = np.nan
    GCDL[GCDL[:,2]>1.5*np.nanmedian(GCDL[:,2]),:] = np.nan
    GCDL[GCDL[:,2]==0,:] = np.nan

    GCDR = np.zeros((len(stridelengths_right),3))*np.nan
    GCDR[:,0] = stridelengths_right[:,3]
    GCDR[:,1] = stridelengths_right[:,1]
    GCDR[:,2] = stridelengths_right[:,1]-stridelengths_right[:,3]
    # GCDR[1:,2] = np.diff(stridelengths_right[:,1])
    GCDR[GCDR[:,2]>2.3*videoframerate,:] = np.nan
    GCDR[GCDR[:,2]<0.3*videoframerate,:] = np.nan
    GCDR[GCDR[:,2]>1.5*np.nanmedian(GCDR[:,2]),:] = np.nan
    GCDR[GCDR[:,2]==0,:] = np.nan


    # Velocity per stride
    velocity_stridesleft = np.zeros((len(stridelengths_left),3))*np.nan
    velocity_stridesleft[:,0] = stridelengths_left[:,0]
    velocity_stridesleft[:,1] = stridelengths_left[:,1]
    velocity_stridesleft[:,2] = (stridelengths_left[:,2]/1000)/(GCDL[:,2]/videoframerate)
    for i in range(len(GCDL)):
        if np.isnan(GCDL[i,2]) == True:
            velocity_stridesleft[i,2] = np.nan
            
    velocity_stridesright = np.zeros((len(stridelengths_right),3))*np.nan
    velocity_stridesright[:,0] = stridelengths_right[:,0]
    velocity_stridesright[:,1] = stridelengths_right[:,1]
    velocity_stridesright[:,2] = (stridelengths_right[:,2]/1000)/(GCDR[:,2]/videoframerate)
    for i in range(len(GCDR)):
        if np.isnan(GCDR[i,2]) == True:
            velocity_stridesright[i,2] = np.nan
    
    # Correct for false strides based on GCD
    stridelengths_left = stridelengths_left[~np.isnan(GCDL).any(axis=1), :]
    stridelengths_right = stridelengths_right[~np.isnan(GCDR).any(axis=1), :]
    velocity_stridesleft = velocity_stridesleft[~np.isnan(GCDL).any(axis=1), :]
    velocity_stridesright = velocity_stridesright[~np.isnan(GCDR).any(axis=1), :]
    GCDL = GCDL[~np.isnan(GCDL).any(axis=1), :]
    GCDR = GCDR[~np.isnan(GCDR).any(axis=1), :]
    
    # Set second-order low-pass butterworth filter;
    fc1 = 5  # Cut-off frequency of the first low pass filter
    wn1 = fc1 / (videoframerate / 2) # Normalize the frequency
    fc2 = 8 # Cut-off frequency of the second low pass filter
    wn2 = fc2 / (videoframerate / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    B1, A1 = signal.butter(N, wn1, filter_type) # First low-pass filterdesign
    B2, A2 = signal.butter(N, wn2, filter_type) # Second low-pass filterdesign
    
    vlank = {}
    vrank = {}
    # Apply low pass filter on data and calculate velocity
    vlank['vlankx'] = signal.filtfilt(B1, A1, markerdatavicon['LANK'][:,0])
    vlank['vlanky'] = signal.filtfilt(B1, A1, markerdatavicon['LANK'][:,1])
    vlank['vlankz'] = signal.filtfilt(B1, A1, markerdatavicon['LANK'][:,2])
    
    vrank['vrankx'] = signal.filtfilt(B1, A1, markerdatavicon['RANK'][:,0])
    vrank['vranky'] = signal.filtfilt(B1, A1, markerdatavicon['RANK'][:,1])
    vrank['vrankz'] = signal.filtfilt(B1, A1, markerdatavicon['RANK'][:,2])
    
    # Back to one matrix
    vlank['vlank'] = np.swapaxes(np.vstack((vlank['vlankx'],vlank['vlanky'], vlank['vlankz'])), 1, 0)
    vrank['vrank'] = np.swapaxes(np.vstack((vrank['vrankx'],vrank['vranky'], vrank['vrankz'])), 1, 0)
    vlank = vlank['vlank']
    vrank = vrank['vrank']
    
    # velocity profile = differentiated markerdatavicon['LANK']
    velocity_left = np.abs(np.diff(vlank, axis=0))
    velocity_right = np.abs(np.diff(vrank, axis=0))
    # Convert mm/frame to m/s
    velocity_left = velocity_left/1000 * videoframerate
    velocity_right = velocity_right/1000 * videoframerate
    
    
    # # Step lengths and stepwidths
    # # Position difference between heel strikes of one foot and heel strike of other foot
    # steplengths_left = np.zeros(len(HSL))*np.nan
    # stepwidths_left = np.zeros(len(HSL))*np.nan
    # steptime_left = np.zeros(len(HSL))*np.nan
    # foot_left = np.zeros((len(HSL),3))*np.nan
    # for i in range(0, len(HSL)):
    #     foot_left[i,:] = np.abs(markerdatavicon['LHEE'][HSL[i]] - markerdatavicon['RHEE'][HSL[i]]) # forward swing is in opposite direction of treadmill
    #     steplengths_left[i] = foot_left[i,1]
    #     stepwidths_left[i] = np.abs(foot_left[i,0])
    #     if i == 0 and HSL[0]<HSR[0]:
    #         steptime_left[i] = HSL[i]-TOL[i]
    #     elif i == 0 and HSL[0]>HSR[0]:
    #         steptime_left[i] = HSL[i]-HSR[HSR<HSL[i]][-1]
    #     if i > 0:
    #         if (HSL[i] - HSR[HSR<HSL[i]][-1]) < 2.3*videoframerate:
    #             steptime_left[i] = HSL[i]-HSR[HSR<HSL[i]][-1]
    #         else:
    #             steptime_left[i] = np.nan
    
    # steplengths_right = np.zeros(len(HSR))*np.nan
    # stepwidths_right = np.zeros(len(HSR))*np.nan
    # steptime_right = np.zeros(len(HSR))*np.nan
    # foot_right = np.zeros((len(HSR),3))*np.nan
    # for i in range(0, len(HSR)):
    #     foot_right[i,:] = (-1) * (markerdatavicon['RHEE'][HSR[i]] - markerdatavicon['LHEE'][HSR[i]]) # forward swing is in opposite direction of treadmill
    #     steplengths_right[i] = foot_right[i,1]
    #     stepwidths_right[i] = np.abs(foot_right[i,0])
    #     if i == 0 and HSR[0]<HSL[0]:
    #         steptime_right[i] = HSR[i]-TOR[i]
    #     elif i == 0 and HSR[0]>HSL[0]:
    #         steptime_right[i] = HSR[i]-HSL[HSL<HSR[i]][-1]
    #     if i > 0:
    #         if (HSR[i] - HSL[HSL<HSR[i]][-1]) < 2.3*videoframerate:
    #             steptime_right[i] = HSR[i]-HSL[HSL<HSR[i]][-1]
    #         else:
    #             steptime_right[i] = np.nan

    
    # Create output dict
    spatiotemporals = {}
    spatiotemporals['Velocity left (m/s)'] = velocity_left
    spatiotemporals['Velocity right (m/s)'] = velocity_right
    spatiotemporals['Gait speed left strides (m/s)'] = velocity_stridesleft
    spatiotemporals['Gait speed right strides (m/s)'] = velocity_stridesright
    spatiotemporals['Gait Cycle duration left (s)'] = GCDL
    spatiotemporals['Gait Cycle duration left (s)'][:,2] = spatiotemporals['Gait Cycle duration left (s)'][:,2]/videoframerate
    spatiotemporals['Gait Cycle duration right (s)'] = GCDR
    spatiotemporals['Gait Cycle duration right (s)'][:,2] = spatiotemporals['Gait Cycle duration right (s)'][:,2]/videoframerate
    spatiotemporals['Stance time left (s)'] = StTL/videoframerate
    spatiotemporals['Stance time right (s)'] = StTR/videoframerate
    spatiotemporals['Swing time left (s)'] = SwTL/videoframerate
    spatiotemporals['Swing time right (s)'] = SwTR/videoframerate
    # spatiotemporals['Steplength left (mm)'] = steplengths_left
    # spatiotemporals['Steplength right (mm)'] = steplengths_right
    # spatiotemporals['Stepwidth left (mm)'] = stepwidths_left
    # spatiotemporals['Stepwidth right(mm)'] = stepwidths_right
    # spatiotemporals['Step time left (s)'] = steptime_left/videoframerate
    # spatiotemporals['Step time right (s)'] = steptime_right/videoframerate
    spatiotemporals['Stridelength left (mm)'] = stridelengths_left[:,0:3]
    spatiotemporals['Stridelength right (mm)'] = stridelengths_right[:,0:3]
    
    return spatiotemporals