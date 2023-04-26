# -*- coding: utf-8 -*-
"""
Detect strikes from VICON data.
Can be applied to both treadmill and overground data.

Based on:
    Zeni, J. A., Jr, Richards, J. G., & Higginson, J. S. (2008).
    Two simple methods for determining gait events during treadmill and overground walking using kinematic data.
    Gait & posture, 27(4), 710–714. https://doi.org/10.1016/j.gaitpost.2007.07.007
    
Input
    'markerdata': dict of VICON C3D labeled markerdata 
    'sample_frequency': sample frequency of VICON C3D labeled markerdata
Output
    'gaitevents': dict of index numbers of heel strike and toe off events

Versions
    2022-07-25 - C.J. Ensink - cleaned up coding, 
    2021-11-25 - C.J. Ensink - gait event closest to zero-crossing instead of firt negative/positive value, cleaned up coding
    2021-03-09 - C.J. Ensink - velocity based
    2021-01-12 - C.J. Ensink - coordinate based

    
"""
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt

def zenieventdetection(markerdata, sample_frequency, **kwargs):
    
    # Define coordinate or velocity based algorithm
    algorithmtype = 'velocity' # default
    # Check for inputname in **kwargs items
    for key, value in kwargs.items():
        if key == 'algorithmtype':
            algorithmtype = value
 
    # Define if treadmill or overground trial
    trialtype = 'treadmill' # default
    for key, value in kwargs.items():
        if key == 'trialtype':
            trialtype = value
    
    # Define heel, toe, sacrum markers
    # Sacrum
    if 'LPSI' in markerdata:
        sacrum = (markerdata['LPSI'] + markerdata['RPSI']) / 2 # Middle between Left and Right Posterior Superior Iliac Spine
    # Correct for missing data
    for i in range(len(markerdata['LPSI'])):
        if np.all(markerdata['LPSI'][i,:] == [0,0,0]) or np.all(markerdata['RPSI'][i,:] == [0,0,0]):
            sacrum[i,:]=[0,0,0]
            
    # Left foot
    heel_left = markerdata['LHEE']
    toe_left = markerdata['LTOE']
    # Right foot
    heel_right = markerdata['RHEE']
    toe_right = markerdata['RTOE']
    
    # Low pass butterworth filter, order = 2, cut-off frequecy = 15
    fc = 15  # Cut-off frequency of the filter
    w = fc / (sample_frequency / 2) # Normalize the frequency

    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, w, filter_type)

    # Apply filter on marker data
    fheel_Lx = signal.filtfilt(b, a, heel_left[:,0])
    fheel_Ly = signal.filtfilt(b, a, heel_left[:,1])
    fheel_Lz = signal.filtfilt(b, a, heel_left[:,2])
    heelL = np.vstack((fheel_Lx, fheel_Ly))
    heelL = np.vstack((heelL, fheel_Lz))
    heel_left = np.transpose(heelL)
    # Correct for missing data
    for i in range(len(markerdata['LHEE'])):
        if np.all(markerdata['LHEE'][i,:] == [0,0,0]):
            heel_left[i,:] = [0,0,0]
            
    fheel_Rx = signal.filtfilt(b, a, heel_right[:,0])
    fheel_Ry = signal.filtfilt(b, a, heel_right[:,1])
    fheel_Rz = signal.filtfilt(b, a, heel_right[:,2])
    heelR = np.vstack((fheel_Rx, fheel_Ry))
    heelR = np.vstack((heelR, fheel_Rz))
    heel_right = np.transpose(heelR)
    # Correct for missing data
    for i in range(len(markerdata['RHEE'])):
        if np.all(markerdata['RHEE'][i,:] == [0,0,0]):
            heel_right[i,:] = [0,0,0]
            
    ftoe_Lx = signal.filtfilt(b, a, toe_left[:,0])
    ftoe_Ly = signal.filtfilt(b, a, toe_left[:,1])
    ftoe_Lz = signal.filtfilt(b, a, toe_left[:,2])
    toeL = np.vstack((ftoe_Lx, ftoe_Ly))
    toeL = np.vstack((toeL, ftoe_Lz))
    toe_left = np.transpose(toeL)
    # Correct for missing data
    for i in range(len(markerdata['LTOE'])):
        if np.all(markerdata['LTOE'][i,:] == [0,0,0]):
            toe_left[i,:] = [0,0,0]
            
    ftoe_Rx = signal.filtfilt(b, a, toe_right[:,0])
    ftoe_Ry = signal.filtfilt(b, a, toe_right[:,1])
    ftoe_Rz = signal.filtfilt(b, a, toe_right[:,2])
    toeR = np.vstack((ftoe_Rx, ftoe_Ry))
    toeR = np.vstack((toeR, ftoe_Rz))
    toe_right = np.transpose(toeR)
    # Correct for missing data
    for i in range(len(markerdata['RTOE'])):
        if np.all(markerdata['RTOE'][i,:] == [0,0,0]):
            toe_right[i,:] = [0,0,0]
            
    if trialtype == 'treadmill':
        APdirection = 1 # y-axis
        axisdef = -1 # along negative axis
        
    elif trialtype == 'overground':
        APdirection = 0 # x-axis
        axisdef = 1
    
    # Velocity of sacrum
    velocity_sacrum = np.append(np.array([0]), np.diff(axisdef*sacrum[:,APdirection]))
    # velocity_sacrum[np.abs(velocity_sacrum)>50] = 0
                
    thprom = 60 # threshold for peak prominence coordinate based algorithm
    ms,_=find_peaks(heel_left[:,2], prominence=thprom) # Estimate average gait cycle duration
    thdist = 0.6*np.median(np.diff(ms)) # threshold for distance between peaks
    
    
    # VELOCITY BASED ALGORITHM
    if algorithmtype == 'velocity':
        
        # To apply to overground data, subtract the anterior-posterior coordinate of the sacral marker from the x coordinate of each marker
        if trialtype == 'overground':
            heel_left[:,APdirection] = heel_left[:,APdirection] - sacrum[:,APdirection]
            toe_left[:,APdirection] = toe_left[:,APdirection] - sacrum[:,APdirection]
            heel_right[:,APdirection] = heel_right[:,APdirection] - sacrum[:,APdirection]
            toe_right[:,APdirection] = toe_right[:,APdirection] - sacrum[:,APdirection]
        
        # Calculate markervelocity, to make equal length, initial velocity is set to 0
        velocity_LHEE = np.append(np.array([0]), np.diff(axisdef*heel_left[:,APdirection])) # mm/frame, anterior posterior direction
        velocity_LTOE = np.append(np.array([0]), np.diff(axisdef*toe_left[:,APdirection])) # mm/frame, anterior posterior direction
        velocity_RHEE = np.append(np.array([0]), np.diff(axisdef*heel_right[:,APdirection])) # mm/frame, anterior posterior direction    
        velocity_RTOE = np.append(np.array([0]), np.diff(axisdef*toe_right[:,APdirection])) # mm/frame, anterior posterior direction
    
        # Overground walking trials are walking up-and-down the positive axis, correct for this.
        if trialtype == 'overground':
            # Correct for changing walking direction
            velocity_LHEE[velocity_sacrum < 0] = -1*velocity_LHEE[velocity_sacrum < 0]
            velocity_LTOE[velocity_sacrum < 0] = -1*velocity_LTOE[velocity_sacrum < 0]
            velocity_RHEE[velocity_sacrum < 0] = -1*velocity_RHEE[velocity_sacrum < 0]
            velocity_RTOE[velocity_sacrum < 0] = -1*velocity_RTOE[velocity_sacrum < 0]
            
        # Correct for high velocity due to missing data
        velocity_LHEE[np.abs(velocity_LHEE)>100] = 0
        velocity_LTOE[np.abs(velocity_LTOE)>100] = 0
        velocity_RHEE[np.abs(velocity_RHEE)>100] = 0
        velocity_RTOE[np.abs(velocity_RTOE)>100] = 0
                        
        # Find sign changes in veloctiy data
        signLHEE = np.sign(velocity_LHEE)
        signchangeLHEE = np.argwhere(((np.roll(signLHEE, 1) - signLHEE) != 0).astype(int))
        idxHSleft = np.array([0], dtype = int)
        for i in range(0, len(signchangeLHEE)):
            if velocity_LHEE[signchangeLHEE[i]] < 0 and heel_left[signchangeLHEE[i],2] < 120:
                if signchangeLHEE[i] > idxHSleft[-1] + thdist:
                    if np.abs(velocity_LHEE[signchangeLHEE[i]]) < np.abs(velocity_LHEE[signchangeLHEE[i]-1]):
                        idxHSleft = np.append(idxHSleft, signchangeLHEE[i]) # AP component of velocity vector changes from positive to negative
                    else:
                        idxHSleft = np.append(idxHSleft, signchangeLHEE[i]-1)
        idxHSleft = idxHSleft[1:]
        
        
        signLTOE = np.sign(velocity_LTOE)
        signchangeLTOE = np.argwhere(((np.roll(signLTOE, 1) - signLTOE) != 0).astype(int))
        idxTOleft = np.array([0], dtype = int)
        for i in range(0, len(signchangeLTOE)):
            if velocity_LTOE[signchangeLTOE[i]] > 0 and toe_left[signchangeLTOE[i],2] < 100:
                if signchangeLTOE[i] > idxTOleft[-1] + thdist:
                    if np.abs(velocity_LTOE[signchangeLTOE[i]]) < np.abs(velocity_LTOE[signchangeLTOE[i]-1]):
                        idxTOleft = np.append(idxTOleft, signchangeLTOE[i])# AP component of velocity vector changes from negative to positive
                    else:
                        idxTOleft = np.append(idxTOleft, signchangeLTOE[i]-1)# AP component of velocity vector changes from negative to positive
        idxTOleft = idxTOleft[1:] 
        
        
        signRHEE = np.sign(velocity_RHEE)
        signchangeRHEE = np.argwhere(((np.roll(signRHEE, 1) - signRHEE) != 0).astype(int))
        idxHSright = np.array([0], dtype = int)
        for i in range(0, len(signchangeRHEE)):
            if velocity_RHEE[signchangeRHEE[i]] < 0 and heel_right[signchangeRHEE[i],2] < 120:
                if signchangeRHEE[i] > idxHSright[-1] + thdist:
                    if np.abs(velocity_RHEE[signchangeRHEE[i]]) < np.abs(velocity_RHEE[signchangeRHEE[i]-1]):
                        idxHSright = np.append(idxHSright, signchangeRHEE[i]) # AP component of velocity vector changes from positive to negative
                    else:
                        idxHSright = np.append(idxHSright, signchangeRHEE[i]-1) # AP component of velocity vector changes from positive to negative
        idxHSright = idxHSright[1:]
                

        signRTOE = np.sign(velocity_RTOE)
        signchangeRTOE = np.argwhere(((np.roll(signRTOE, 1) - signRTOE) != 0).astype(int))
        idxTOright = np.array([0], dtype = int)
        for i in range(0, len(signchangeRTOE)):
            if velocity_RTOE[signchangeRTOE[i]] > 0 and toe_right[signchangeRTOE[i],2] < 100:
                if signchangeRTOE[i] > idxTOright[-1] + thdist:
                    if np.abs(velocity_RTOE[signchangeRTOE[i]]) < np.abs(velocity_RTOE[signchangeRTOE[i]-1]):
                        idxTOright = np.append(idxTOright, signchangeRTOE[i])# AP component of velocity vector changes from negative to positive
                    else:
                        idxTOright = np.append(idxTOright, signchangeRTOE[i]-1)# AP component of velocity vector changes from negative to positive
        idxTOright = idxTOright[1:]
        
            
            
    elif algorithmtype == 'coordinate':
        # Subract sacrum from heel and toe in Anterior-Posterior direction
        diffHeel_left = heel_left
        diffHeel_left[:,APdirection] = heel_left[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from heel
        diffToe_left = toe_left
        diffToe_left[:,APdirection] = toe_left[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from toe
        
        diffHeel_right = heel_right
        diffHeel_right[:,APdirection] = heel_right[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from heel
        diffToe_right = toe_right
        diffToe_right[:,APdirection] = toe_right[:,APdirection] - sacrum[:,APdirection] # Subtract AP coordinate of sacrum from toe
    
    
        idxHSleft, _ = find_peaks(-diffHeel_left[:,1], prominence = thprom) # Find negative peaks (Heel strike)
        idxTOleft, _ = find_peaks(diffToe_left[:,1], prominence = thprom) # Find positive peaks (Toe off)
        
        idxHSright, _ = find_peaks(-diffHeel_right[:,1], prominence = thprom) # Find negative peaks (Heel strike)
        idxTOright, _ = find_peaks(diffToe_right[:,1], prominence = thprom) # Find positive peaks (Toe off)
        
        # Deem heel strikes before the first toe off as artefact
        remove = np.array([], dtype = int)
        for i in range(0,len(idxHSleft)):
            if idxHSleft[i] < idxTOleft[0]:
                remove = np.append(remove, idxHSleft[i])
        idxHSleft = idxHSleft[~np.in1d(idxHSleft,remove)]
        remove = np.array([], dtype = int)
        for i in range(0,len(idxHSright)):
            if idxHSright[i] < idxTOright[0]:
                remove = np.append(remove, idxHSright[i])
        idxHSright = idxHSright[~np.in1d(idxHSright, remove)]
        # Deem toe off after last heel strike as artefact
        remove = np.array([], dtype = int)
        for i in range(0,len(idxTOleft)):
            if idxTOleft[i] > idxHSleft[-1]:
                remove = np.append(remove, idxTOleft[i])
        idxTOleft = idxTOleft[~np.in1d(idxTOleft, remove)]
        remove = np.array([], dtype = int)
        for i in range(0,len(idxTOright)):
            if idxTOright[i] > idxHSright[-1]:
                remove = np.append(remove, idxTOright[i])
        idxTOright = idxTOright[~np.in1d(idxTOright, remove)]

    
    elif algorithmtype =='SMK':
        # Stagiaire Tom van Lysanne heeft hetzelfde probleem gehad. Dit hebben we als volgt opgelost:
        # Heel strike was determined as the mean of the instant that the vertical position of the heel marker was lowest and the heel marker maximally decelerated.
        # Toe-off was determined by taking the mean of the instant that the vertical position of the toe marker increased and started to accelerate.
        # Matlab programma is terug te vinden in:
        # V:\research_reva_studies\827_sensor_drukplaat_GBA\II_Onderzoeksdata\Syntaxen\Tom
        # File heet: detectstepGBATom
        # Dit hebben we toen vergeleken met de forceplate voor de stappen die op de forceplate kwamen. Ging best goed wat ik mij kan herinneren.
        
        # Heel strike detection
        velocity_LHEE = np.diff(heel_left[:,2]) # mm/frame, vertical direction
        velocity_RHEE = np.diff(heel_right[:,2]) # mm/frame, vertical direction
        
        peakpromL = 0.5*np.max(velocity_LHEE)
        peakpromR = 0.5*np.max(velocity_RHEE)
    
        # Find sign changes in veloctiy data
        signLHEE = np.sign(velocity_LHEE)
        signchangeLHEE = ((np.roll(signLHEE, 1) - signLHEE) != 0).astype(int)
        idxVelleft = np.array([0], dtype = int)
        for i in range(0, len(signchangeLHEE)):
            if signchangeLHEE[i] == 1:
                if velocity_LHEE[i] > 0:
                    if i > idxVelleft[-1] + thdist:
                        if np.abs(velocity_LHEE[i]) > np.abs(velocity_LHEE[i-1]):
                            idxVelleft = np.append(idxVelleft, i) # AP component of velocity vector changes from positive to negative
                        else:
                            idxVelleft = np.append(idxVelleft, i-1)
        idxVelleft = idxVelleft[1:]
        
        # Find sign changes in veloctiy data
        signRHEE = np.sign(velocity_RHEE)
        signchangeRHEE = ((np.roll(signRHEE, 1) - signRHEE) != 0).astype(int)
        idxVelright = np.array([0], dtype = int)
        for i in range(0, len(signchangeRHEE)):
            if signchangeRHEE[i] == 1:
                if velocity_RHEE[i] > 0:
                    if i > idxVelright[-1] + thdist:
                        if np.abs(velocity_RHEE[i]) > np.abs(velocity_RHEE[i-1]):
                            idxVelright = np.append(idxVelright, i) # AP component of velocity vector changes from positive to negative
                        else:
                            idxVelright = np.append(idxVelright, i-1)
        idxVelright = idxVelright[1:]
        
        
        idxPosleft, _ = find_peaks(-heel_left[:,2], distance=thdist) # Find negative peaks (Heel strike)
        idxPosright, _ = find_peaks(-heel_right[:,2], distance=thdist)
        
        idxHSleft = idxVelleft
        idxHSright = idxVelright
        
        # Toe-off detection
        velocity_LTOE = np.diff(toe_left[:,2]) # mm/frame, vertical direction
        velocity_RTOE = np.diff(toe_right[:,2]) # mm/frame, vertical direction
        
        # Find sign changes in veloctiy data
        signLTOE = np.sign(velocity_LTOE)
        signchangeLTOE = ((np.roll(signLTOE, 1) - signLTOE) != 0).astype(int)
        idxVelleft = np.array([0], dtype = int)
        for i in range(0, len(signchangeLTOE)):
            if signchangeLTOE[i] == 1:
                if velocity_LTOE[i] > 0:
                    if i > idxVelleft[-1] + thdist:
                        if np.abs(velocity_LTOE[i]) > np.abs(velocity_LTOE[i-1]):
                            idxVelleft = np.append(idxVelleft, i) # AP component of velocity vector changes from positive to negative
                        else:
                            idxVelleft = np.append(idxVelleft, i-1)
        idxVelleft = idxVelleft[1:]
        
        # Find sign changes in veloctiy data
        signRTOE = np.sign(velocity_RTOE)
        signchangeRTOE = ((np.roll(signRTOE, 1) - signRTOE) != 0).astype(int)
        idxVelright = np.array([0], dtype = int)
        for i in range(0, len(signchangeRTOE)):
            if signchangeRTOE[i] == 1:
                if velocity_RTOE[i] > 0:
                    if i > idxVelright[-1] + thdist:
                        if np.abs(velocity_RTOE[i]) > np.abs(velocity_RTOE[i-1]):
                            idxVelright = np.append(idxVelright, i) # AP component of velocity vector changes from positive to negative
                        else:
                            idxVelright = np.append(idxVelright, i-1)
        idxVelright = idxVelright[1:]
                
        idxTOleft = idxVelleft
        idxTOright = idxVelright
        
    elif algorithmtype == 'pijnappels':
        # The timing of HS correlated closely to the time of a local minimum in the vertical velocity component of the toe marker.
        # The timing of TO was closely correlated to the time of a local maximum in the vertical velocity component of the heel marker.
        
        # Heel strike detection
        velocity_LTOE = np.diff(toe_left[:,2]) # mm/frame, vertical direction
        velocity_RTOE = np.diff(toe_right[:,2]) # mm/frame, vertical direction
        
        # peakpromL = 0.5*np.max(velocity_LTOE)
        # peakpromR = 0.5*np.max(velocity_RTOE)
    
        idxHSleft, _ = find_peaks(-velocity_LTOE, distance=thdist)#, prominence=peakpromL) # Find negative peaks (Heel strike)
        idxHSright, _ = find_peaks(-velocity_RTOE, distance=thdist)#, prominence=peakpromR) # Find negative peaks (Heel strike)
        # idxHSleft = idxHSleft[heel_left[idxHSleft,2] < 1.2*np.min(heel_left[idxHSleft,2])]
        # idxHSright = idxHSright[heel_right[idxHSright,2] < 1.2*np.min(heel_right[idxHSright,2])]
        
        # Toe off detection
        velocity_LHEE = np.diff(heel_left[:,2]) # mm/frame, vertical direction
        velocity_RHEE = np.diff(heel_right[:,2]) # mm/frame, vertical direction    
        
        idxTOleft, _ = find_peaks(velocity_LHEE, distance=thdist) # Find positive peaks (Toe off)
        idxTOright, _ = find_peaks(velocity_RHEE, distance=thdist) # Find positive peaks (Toe off)
        # idxTOleft = idxTOleft[toe_left[idxTOleft-1,2] < toe_left[idxTOleft,2]]
        # idxTOright = idxTOright[toe_right[idxTOright-1,2] < toe_right[idxTOright,2]]
        
        
        
    # remove all events in first 50 and last 50 samples
    firstn = 50
    lastn = 50
    idxTOleft = idxTOleft[idxTOleft>firstn]
    idxTOleft = idxTOleft[idxTOleft<len(toe_left)-lastn]
    idxHSleft = idxHSleft[idxHSleft>firstn]
    idxHSleft = idxHSleft[idxHSleft<len(heel_left)-lastn]
    idxHSleft = idxHSleft[idxHSleft>idxTOleft[0]]
    idxTOleft = idxTOleft[idxTOleft<idxHSleft[-1]]
    idxTOright = idxTOright[idxTOright>firstn]
    idxTOright = idxTOright[idxTOright<len(toe_right)-lastn]
    idxHSright = idxHSright[idxHSright>firstn]
    idxHSright = idxHSright[idxHSright<len(heel_right)-lastn]
    idxHSright = idxHSright[idxHSright>idxTOright[0]]
    idxTOright = idxTOright[idxTOright<idxHSright[-1]]
    
    # remove events in case markerdata is missing at instant of found event
    removeHSleft=np.array([])
    for t in range(0,len(idxHSleft)):
        if np.all(markerdata['LPSI'][idxHSleft[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxHSleft[t],:] == [0,0,0]) or np.all(markerdata['LHEE'][idxHSleft[t],:] == [0,0,0]) or np.all(markerdata['LTOE'][idxHSleft[t],:] == [0,0,0]):
            removeHSleft = np.append(removeHSleft, idxHSleft[t])
    for r in range(0,len(removeHSleft)):
        idxHSleft = np.delete(idxHSleft, np.where(idxHSleft == removeHSleft[r]))
    
    removeHSright=np.array([])
    for t in range(0,len(idxHSright)):
        if np.all(markerdata['LPSI'][idxHSright[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxHSright[t],:] == [0,0,0]) or np.all(markerdata['RHEE'][idxHSright[t],:] == [0,0,0]) or np.all(markerdata['RTOE'][idxHSright[t],:] == [0,0,0]):
            removeHSright = np.append(removeHSright, idxHSright[t])
    for r in range(0,len(removeHSright)):
        idxHSright = np.delete(idxHSright, np.where(idxHSright == removeHSright[r]))
        
    removeTOleft=np.array([])
    for t in range(0,len(idxTOleft)):
        if np.all(markerdata['LPSI'][idxTOleft[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxTOleft[t],:] == [0,0,0]) or np.all(markerdata['LTOE'][idxTOleft[t],:] == [0,0,0]) or np.all(markerdata['LHEE'][idxTOleft[t],:] == [0,0,0]):
            removeTOleft = np.append(removeTOleft, idxTOleft[t])
    for r in range(0,len(removeTOleft)):
        idxTOleft = np.delete(idxTOleft, np.where(idxTOleft == removeTOleft[r]))
    
    removeTOright=np.array([])
    for t in range(0,len(idxTOright)):
        if np.all(markerdata['LPSI'][idxTOright[t],:] == [0,0,0]) or np.all(markerdata['RPSI'][idxTOright[t],:] == [0,0,0]) or np.all(markerdata['RTOE'][idxTOright[t],:] == [0,0,0]) or np.all(markerdata['RHEE'][idxTOright[t],:] == [0,0,0]):
            removeTOright = np.append(removeTOright, idxTOright[t])
    for r in range(0,len(removeTOright)):
        idxTOright = np.delete(idxTOright, np.where(idxTOright == removeTOright[r]))
    
    
    
    
    # # Debug figure
    # figaxis = 2
    # fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    # # ax1.plot(heel_left[:,figaxis], label = 'Heel marker left')
    # # ax1.plot(idxHSleft, heel_left[idxHSleft,figaxis], 'gv', label = 'HS left')
    # # ax1.plot(toe_left[:,figaxis], label = 'Toe marker left')
    # # ax1.plot(idxTOleft, toe_left[idxTOleft,figaxis], 'r^', label = 'TO left')
    
    # ax1.plot(velocity_LHEE, label = 'Heel velocity left')
    # ax1.plot(idxHSleft, velocity_LHEE[idxHSleft], 'mv')
    # ax1.plot(idxpeaksleft, velocity_LHEE[idxpeaksleft], 'kx')
    # ax1.plot(velocity_LTOE, label = 'Toe velocity left')
    # ax1.plot(idxTOleft, velocity_LTOE[idxTOleft], 'b^')
    # ax1.plot(idxpeaksleft, velocity_LTOE[idxpeaksleft], 'kx')
    
    # # ax2.plot(heel_right[:,figaxis], label = 'Heel marker right')
    # # ax2.plot(idxHSright, heel_right[idxHSright,figaxis], 'g^', label = 'HS right')
    # # ax2.plot(toe_right[:,figaxis], label = 'Toe marker right')
    # # ax2.plot(idxTOright, toe_right[idxTOright,figaxis], 'rv', label = 'TO right')
    
    # ax2.plot(velocity_RHEE, label = 'Heel velocity right')
    # ax2.plot(idxHSright, velocity_RHEE[idxHSright], 'mv')
    # ax2.plot(idxpeaksright, velocity_RHEE[idxpeaksright], 'kx')
    # ax2.plot(velocity_RTOE, label = 'Toe velocity right')
    # ax2.plot(idxTOright, velocity_RTOE[idxTOright], 'b^')
    # ax2.plot(idxpeaksright, velocity_RTOE[idxpeaksright], 'kx')
    
    # plt.legend()
    
    
    # Put gait event index numbers in dict
    gaitevents={}
    gaitevents['Index numbers heel strike left'] = np.sort(idxHSleft)
    gaitevents['Index numbers heel strike right'] = np.sort(idxHSright)
    gaitevents['Index numbers toe off left'] = np.sort(idxTOleft)
    gaitevents['Index numbers toe off right'] = np.sort(idxTOright)
    
    return gaitevents
    