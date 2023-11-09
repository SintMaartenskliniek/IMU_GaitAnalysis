"""
Spatiotemporal functions

Last update: July 2022
Author: Carmen Ensink

"""
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from scipy import interpolate

def walkingsamples(data, errors):
    # INPUT:    indexToeOffLeft; index numbers of Toe-Off points left
    #           indexToeOffRight; index numbers of Toe-Off points right
    #           indexHeelStrikeLeft; index numbers of Heel-Strike points left
    #           indexHeelStrikeRight; index numbers of Heel-Strike points right
    # OUTPUT:   ['Gait Phases']['Walking samples']; index numbers of samples identified as
    #           part of walkingperiod
    # 
    # 
    # According to Fang et al. (2018) full gait cycle from Heel-Strike to
    # Heel-Strike of the same leg. Max. found duration: 1.15(0.13) seconds
    # Assumed is that a person stopped walking when no new Toe-Off point is
    # found within 5 seconds after the last ToeOff point.
    
    if errors['No walking period'] == False:
        indexToeOffLeft = data['Left foot']['Gait Events']['Terminal Contact']
        indexToeOffRight = data['Right foot']['Gait Events']['Terminal Contact']
        indexHeelStrikeLeft = data['Left foot']['Gait Events']['Initial Contact']
        indexHeelStrikeRight = data['Right foot']['Gait Events']['Initial Contact']
        sample_frequency = data['Sample Frequency (Hz)']
        
        idxAllTO = np.sort(np.unique(np.append(indexToeOffLeft, indexToeOffRight)), axis=None)
        idxAllHS = np.sort(np.unique(np.append(indexHeelStrikeLeft, indexHeelStrikeRight)), axis=None)
        
        timerange = 5*sample_frequency
        interval = np.diff(idxAllTO) # first index includes amount samples until second TO
    
        locidx = (np.zeros((idxAllHS[-1],1), dtype = 'int64')).flatten()
    
        for i in range(0,len(interval)):
            if interval[i] < timerange:
                startTO = idxAllTO[i]
                stopTO = idxAllTO[i+1]
                locidx[startTO:stopTO] = 1
                
                idxlastTOinWalking = np.where(locidx == 1)[-1][-1]
    
                finalHSinWalking = np.where(idxAllHS > idxlastTOinWalking)[0][0]
                idxfinalHSinWalking = idxAllHS[finalHSinWalking]
                locidx[idxlastTOinWalking:idxfinalHSinWalking] = 1
        
        data['Left foot']['Gait Phases']['Walking samples'] = (np.asarray(np.where(locidx == 1))).flatten()
        data['Right foot']['Gait Phases']['Walking samples'] = (np.asarray(np.where(locidx == 1))).flatten()
    else:
        data['Left foot']['Gait Phases']['Walking samples'] = np.array([])
        data['Right foot']['Gait Phases']['Walking samples'] = np.array([])
    return data





def nonactivesamples(data):

    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction of the analyzed trial
    #           indexWalkingPeriods; index numbers of samples identified as
    #           part of walkingperiod
    # OUTPUT:   ['Gait Phases']['Non-Active samples']; index numbers of samples identified as
    #           part of Non-Active period
    #
    # Assumed is that a person is non-active when sample is not identified as walking.
    
    length = np.min( [len(data['Left foot']['raw']['Gyroscope']), len(data['Right foot']['raw']['Gyroscope'])])
    indexWalkingPeriods = data['Left foot']['Gait Phases']['Walking samples']
    
    locidx = np.ones((length), dtype='int64')
    for i in range(0,len(indexWalkingPeriods)):
        locidx[indexWalkingPeriods[i]] = 0
    
    data['Left foot']['Gait Phases']['Non-Active samples'] = (np.argwhere(locidx ==1)).flatten()
    data['Right foot']['Gait Phases']['Non-Active samples'] = (np.argwhere(locidx ==1)).flatten()

    return data




def stepcount(data):
    # INPUT:    indexToeOffleft; index numbers of Toe-Off points left
    #           indexToeOffright; index numbers of Toe-Off points right
    # OUTPUT:   ['Spatiotemporals']['Number of steps (#)']; number of steps made within analyzed data
    
    indexToeOffleft = data['Left foot']['Gait Events']['Terminal Contact']
    indexToeOffright = data['Right foot']['Gait Events']['Terminal Contact']
    
    # Amount Toe-Off points right +left
    stepsleft = len(indexToeOffleft)
    stepsright = len(indexToeOffright)
    data['Spatiotemporals']['Number of steps (#)'] = stepsleft + stepsright
    
    return data





def cadence(data, errors):

    # INPUT:    indexToeOffleft; index numbers of Toe-Off points left
    #           indexToeOffright; index numbers of Toe-Off points right
    #           indexWalkingPeriods; index numbers of samples identified as
    #           part of walkingperiod
    # OUTPUT:   ['Spatiotemporals']['Cadence (steps/minute)']; cadance (amount of steps made per minute)
    #           
    
    if errors['No walking period'] == False:
        indexToeOffleft = data['Left foot']['Gait Events']['Terminal Contact']
        indexToeOffright = data['Right foot']['Gait Events']['Terminal Contact']
        indexWalkingPeriods = data['Left foot']['Gait Phases']['Walking samples']
        sample_frequency = data['Sample Frequency (Hz)']
        
        idxAllTO = np.append(indexToeOffleft, indexToeOffright)
        
        amountWalkingSamples = len(indexWalkingPeriods)
        amountWalkingMinutes = amountWalkingSamples/sample_frequency/60
        
        stepsperminute = len(idxAllTO)/amountWalkingMinutes
        data['Spatiotemporals']['Cadence (steps/minute)'] = round(stepsperminute,2)
    else:
        data['Spatiotemporals']['Cadence (steps/minute)'] = np.nan
    return data





def swingtime(data, errors):
    # INPUT:    indexToeOff; index numbers Toe-Off points
    #           indexHeelStrike; index numbers Heel-Strike points
    #           sample_frequency; sample frequency of the sensordata
    # OUTPUT:   ['Gait Phases']['Swing samples']; index numbers of samples identified as Swing-Time period
    #           ['Gait Phases']['Swing time (samples)']; Swing-Time duration for each identified swing in samples
    #           ['Spatiotemporals']['Swing time right (s)']; median Swing-Time duration in seconds
    #
    # Swing time; time between Toe-Off and Heel-Strike of the same foot
    # As an output; all index numbers of samples identified as Swing-Time period
    # And sort all calculated Swing-Time durations and take the median
    
    def functionswingtime(indexToeOff, indexHeelStrike, sample_frequency):
        swing_samples = np.array([], dtype = int)
        swing_time_samples = np.array([], dtype = int)
        swing_time_median_seconds = np.array([])
        
        for i in range(0,len(indexHeelStrike)):
            lastTO = np.argwhere(indexToeOff < indexHeelStrike[i])
            if len(lastTO) > 0:
                lastTO = lastTO[-1][0]
                swing_samples = np.append(swing_samples, np.arange(start=indexToeOff[lastTO], stop=indexHeelStrike[i], dtype = int))
                swing_time_samples = np.append(swing_time_samples, (indexHeelStrike[i] - indexToeOff[lastTO]))
        
        SwingTime = np.nanmedian(swing_time_samples)/sample_frequency
        swing_time_median_seconds = round(SwingTime,2)
    
        return swing_samples, swing_time_samples, swing_time_median_seconds
    
    sample_frequency = data['Sample Frequency (Hz)']
    
    sides = ['left', 'right']
    if errors['No walking period'] == False:
        for i in range(0, len(sides)):
            if sides[i] == 'left':
                indexToeOff = data['Left foot']['Gait Events']['Terminal Contact']
                indexHeelStrike = data['Left foot']['Gait Events']['Initial Contact']
                
                data['Left foot']['Gait Phases']['Swing samples'], data['Left foot']['Gait Phases']['Swing time (samples)'], data['Spatiotemporals']['Swing time left (s)']= functionswingtime(indexToeOff, indexHeelStrike, sample_frequency)
                
            elif sides[i] == 'right':
                indexToeOff = data['Right foot']['Gait Events']['Terminal Contact']
                indexHeelStrike = data['Right foot']['Gait Events']['Initial Contact']
                
                data['Right foot']['Gait Phases']['Swing samples'], data['Right foot']['Gait Phases']['Swing time (samples)'], data['Spatiotemporals']['Swing time right (s)']= functionswingtime(indexToeOff, indexHeelStrike, sample_frequency)
    else:
        data['Left foot']['Gait Phases']['Swing samples'] = np.array([],dtype=int)
        data['Left foot']['Gait Phases']['Swing time (samples)'] = np.array([],dtype=int)
        data['Spatiotemporals']['Swing time left (s)'] = np.nan
        data['Right foot']['Gait Phases']['Swing samples'] = np.array([],dtype=int)
        data['Right foot']['Gait Phases']['Swing time (samples)'] = np.array([],dtype=int)
        data['Spatiotemporals']['Swing time right (s)'] = np.nan
    return data





def stancetime(data, errors):
    # INPUT:    indexToeOff; index numbers Toe-Off points
    #           indexHeelStrike; index numbers Heel-Strike points
    #           sample_frequency; sample frequency of the sensordata
    # OUTPUT:   ['Gait Phases']['Stance samples']; index numbers of samples identified as Stance-Time period
    #           ['Gait Phases']['Stance time (samples)']; Stance-Time duration for each identified stance in samples
    #           ['Spatiotemporals']['Stance time right (s)']; median Stance-Time duration in seconds
    #
    #
    # Stance time; time between Heel-Strike and Toe-Off of the same foot
    # As an output; all index numbers of samples identified as Swing-Time period
    # And sort all calculated Swing-Time durations and take the median
    
    def functionstancetime(indexToeOff, indexHeelStrike, sample_frequency):
        stance_samples = np.array([], dtype = int)
        stance_time_samples = np.array([], dtype = int)
        stance_time_median_seconds = np.array([])
        
        for i in range(0,len(indexToeOff)):
            lastHS = np.argwhere(indexHeelStrike < indexToeOff[i])
            if len(lastHS) > 0:
                lastHS = lastHS[-1][0]
                stance_samples = np.append(stance_samples, np.arange(start=indexHeelStrike[lastHS], stop=indexToeOff[i], dtype = int))
                stance_time_samples = np.append(stance_time_samples, (indexToeOff[i] - indexHeelStrike[lastHS]))
        StanceTime = st.median(stance_time_samples)/sample_frequency
        stance_time_median_seconds = round(StanceTime,2)
    
        return stance_samples, stance_time_samples, stance_time_median_seconds
    
    sample_frequency = data['Sample Frequency (Hz)']
    
    sides = ['left', 'right']
    if errors['No walking period'] == False:
        for i in range(0, len(sides)):
            if sides[i] == 'left':
                indexToeOff = data['Left foot']['Gait Events']['Terminal Contact']
                indexHeelStrike = data['Left foot']['Gait Events']['Initial Contact']
                
                data['Left foot']['Gait Phases']['Stance samples'], data['Left foot']['Gait Phases']['Stance time (samples)'], data['Spatiotemporals']['Stance time left (s)']= functionstancetime(indexToeOff, indexHeelStrike, sample_frequency)
                
            elif sides[i] == 'right':
                indexToeOff = data['Right foot']['Gait Events']['Terminal Contact']
                indexHeelStrike = data['Right foot']['Gait Events']['Initial Contact']
                
                data['Right foot']['Gait Phases']['Stance samples'], data['Right foot']['Gait Phases']['Stance time (samples)'], data['Spatiotemporals']['Stance time right (s)']= functionstancetime(indexToeOff, indexHeelStrike, sample_frequency)
    else:
        data['Left foot']['Gait Phases']['Stance samples'] = np.array([],dtype=int)
        data['Left foot']['Gait Phases']['Stance time (samples)'] = np.array([],dtype=int)
        data['Spatiotemporals']['Stance time left (s)'] = np.nan
        data['Right foot']['Gait Phases']['Stance samples'] = np.array([],dtype=int)
        data['Right foot']['Gait Phases']['Stance time (samples)'] = np.array([],dtype=int)
        data['Spatiotemporals']['Stance time right (s)'] = np.nan
    return data





def stridetime(data, errors):
    # INPUT:    indexHeelStrike; index numbers of samples identified as
    #           Heel-Strike
    #           sample_frequency; sample frequency of the sensordata
    # OUTPUT:   ['Spatiotemporals']['Stride time (s)']; median Stride-Time duration in seconds 
    #           ['Gait Phases']['Stride time per stride (samples)']; stride duration for each stride in samples
    #
    # Time between Heel-Strike and next Heel-Strike of the same foot
    # As an output; sort all calculated Stride-Time durations and take the median
    
    def functionstridetime(indexHeelStrike, sample_frequency):
        StrideDurations = np.diff(indexHeelStrike)
        StrideTime = np.nanmedian(StrideDurations)/sample_frequency
        StrideTime = round(StrideTime,2)
        
        return StrideTime, StrideDurations

    sample_frequency = data['Sample Frequency (Hz)']
    
    sides = ['left', 'right']
    if errors['No walking period'] == False:
        for i in range(0, len(sides)):
            if sides[i] == 'left':
                indexHeelStrike = data['Left foot']['Gait Events']['Initial Contact']
                
                data['Spatiotemporals']['Stride time left (s)'], data['Left foot']['Gait Phases']['Stride time per stride (samples)']= functionstridetime(indexHeelStrike, sample_frequency)
                
            elif sides[i] == 'right':
                indexHeelStrike = data['Right foot']['Gait Events']['Initial Contact']
                
                data['Spatiotemporals']['Stride time right (s)'], data['Right foot']['Gait Phases']['Stride time per stride (samples)']= functionstridetime(indexHeelStrike, sample_frequency)
    else:
        data['Spatiotemporals']['Stride time left (s)'] = np.nan
        data['Left foot']['Gait Phases']['Stride time per stride (samples)'] = np.array([],dtype=int)
        data['Spatiotemporals']['Stride time right (s)'] = np.nan
        data['Right foot']['Gait Phases']['Stride time per stride (samples)'] = np.array([],dtype=int)
    return data





def steptime(data, errors):
    # INPUT:    indexHeelStrikeLeft; index numbers Heel-Strike points left data
    #           indexHeelStrikeRight; index numbers Heel-Strike points right data
    #           sample_frequency; sample frequency of the sensordata
    # OUTPUT:   ['Gait Phases']['Step samples']; index numbers of samples identified as Step-Time period
    #           ['Gait Phases']['Step time (samples)']; Step-Time duration for each identified step in samples
    #           ['Spatiotemporals']['Step time (s)']; median Step-Time duration in seconds
    #
    #
    # Step time; time between Heel-Strike of one foot and Heel-Strike of the other foot
    # As an output; sort all calculated Step-Time durations and take the median
    #
    # Step time left is the time from Heel-Strike right until Heel-Strike left
    # Step time right is the time from Heel-Strike left until Heel-Strike right
    
    indexHeelStrikeLeft = data['Left foot']['Gait Events']['Initial Contact']
    indexHeelStrikeRight = data['Right foot']['Gait Events']['Initial Contact']
    sample_frequency = data['Sample Frequency (Hz)']
    
    data['Right foot']['Gait Phases']['Step samples'] = np.array([], dtype = int)
    data['Left foot']['Gait Phases']['Step samples'] = np.array([], dtype = int)
    data['Right foot']['Gait Phases']['Step time (samples)'] = np.array([], dtype = int)
    data['Left foot']['Gait Phases']['Step time (samples)'] = np.array([], dtype = int)
    
    if errors['No walking period'] == False:
        data['Spatiotemporals']['Step time left (s)'] = np.array([])
        data['Spatiotemporals']['Step time right (s)'] = np.array([])
        
        for i in range(0,len(indexHeelStrikeRight)):
            lastHSleft = np.argwhere(indexHeelStrikeLeft < indexHeelStrikeRight[i])
            if len(lastHSleft) > 0:
                lastHSleft = lastHSleft[-1][0]
                data['Right foot']['Gait Phases']['Step samples'] = np.append(data['Right foot']['Gait Phases']['Step samples'], np.arange(start=indexHeelStrikeLeft[lastHSleft], stop=indexHeelStrikeRight[i], dtype = int))
                data['Right foot']['Gait Phases']['Step time (samples)'] = np.append(data['Right foot']['Gait Phases']['Step time (samples)'], (indexHeelStrikeRight[i] - indexHeelStrikeLeft[lastHSleft]))
        StepTime = np.nanmedian(data['Right foot']['Gait Phases']['Step time (samples)'])/sample_frequency
        data['Spatiotemporals']['Step time right (s)'] = round(StepTime,2)
        
        for i in range(0,len(indexHeelStrikeLeft)):
            lastHSright = np.argwhere(indexHeelStrikeRight < indexHeelStrikeLeft[i])
            if len(lastHSright) > 0:
                lastHSright = lastHSright[-1][0]
                data['Left foot']['Gait Phases']['Step samples'] = np.append(data['Left foot']['Gait Phases']['Step samples'], np.arange(start=indexHeelStrikeRight[lastHSright], stop=indexHeelStrikeLeft[i], dtype = int))
                data['Left foot']['Gait Phases']['Step time (samples)'] = np.append(data['Left foot']['Gait Phases']['Step time (samples)'], (indexHeelStrikeLeft[i] - indexHeelStrikeRight[lastHSright]))
        StepTime = st.median(data['Left foot']['Gait Phases']['Step time (samples)'])/sample_frequency
        data['Spatiotemporals']['Step time left (s)'] = round(StepTime,2)
    
    else:
        data['Spatiotemporals']['Step time left (s)'] = np.nan
        data['Spatiotemporals']['Step time right (s)'] = np.nan
    return data





def doublesupport(data, errors):
    # INPUT:    indexToeOffLeft; index numbers of samples identified as Toe Off of left data
    #           indexToeOffRight; index numbers of samples identified as Toe Off of right data
    #           indexHeelStrikeLeft; index numbers of samples identified as Heel-Stike left data
    #           indexHeelStrikeRight; index numbers of samples identified as Heel-Strike right data 
    #           sample_frequency; sample frequency of the sensordata
    # OUTPUT:   ['Left foot']['Gait Phases']['Double support samples']; index numbers of samples identified as Double-Support period after left Heel-Strike
    #           ['Right foot']['Gait Phases']['Double support samples']; index numbers of samples identified as Double-Support period after right Heel-Strike
    #           ['Spatiotemporals']['Double support time left (s)']; median Double-Support duration in seconds after left Heel-Strike
    #           ['Spatiotemporals']['Double support time right (s)']; median Double-Support duration in seconds after right Heel-Strike
    #
    # Double support time: time between Heel-Stike of one foot and Toe-Off of the other foot (time that both feet are in stance-phase)
    
    indexToeOffLeft = data['Left foot']['Gait Events']['Terminal Contact']
    indexToeOffRight = data['Right foot']['Gait Events']['Terminal Contact']
    indexHeelStrikeLeft = data['Left foot']['Gait Events']['Initial Contact']
    indexHeelStrikeRight = data['Right foot']['Gait Events']['Initial Contact']
    sample_frequency = data['Sample Frequency (Hz)']
    
    data['Right foot']['Gait Phases']['Double support samples'] = np.array([], dtype = int)
    data['Left foot']['Gait Phases']['Double support samples'] = np.array([], dtype = int)
    data['Right foot']['Gait Phases']['Double support time (samples)'] = np.array([], dtype = int)
    data['Left foot']['Gait Phases']['Double support time (samples)'] = np.array([], dtype = int)
    
    if errors['No walking period'] == False:
        data['Spatiotemporals']['Double support time right (s)'] = np.array([])
        data['Spatiotemporals']['Double support time left (s)'] = np.array([])
        
        for i in range(0,len(indexHeelStrikeRight)):
            firstTOleft = np.argwhere(indexToeOffLeft > indexHeelStrikeRight[i])
            if len(firstTOleft) > 0:
                firstTOleft = firstTOleft[0][0]
                data['Right foot']['Gait Phases']['Double support samples'] = np.append(data['Right foot']['Gait Phases']['Double support samples'], np.arange(start=indexHeelStrikeRight[i], stop=indexToeOffLeft[firstTOleft], dtype = int))
                data['Right foot']['Gait Phases']['Double support time (samples)'] = np.append(data['Right foot']['Gait Phases']['Double support time (samples)'], (indexToeOffLeft[firstTOleft] - indexHeelStrikeRight[i]))
        DoubleSupportTime = st.median(data['Right foot']['Gait Phases']['Double support time (samples)'])/sample_frequency
        data['Spatiotemporals']['Double support time right (s)'] = round(DoubleSupportTime,2)
        
        for i in range(0,len(indexHeelStrikeLeft)):
            firstTOright = np.argwhere(indexToeOffRight > indexHeelStrikeLeft[i])
            if len(firstTOright) > 0:
                firstTOright = firstTOright[0][0]
                data['Left foot']['Gait Phases']['Double support samples'] = np.append(data['Left foot']['Gait Phases']['Double support samples'], np.arange(start=indexHeelStrikeLeft[i], stop=indexToeOffRight[firstTOright], dtype = int))
                data['Left foot']['Gait Phases']['Double support time (samples)'] = np.append(data['Left foot']['Gait Phases']['Double support time (samples)'], (indexToeOffRight[firstTOright] - indexHeelStrikeLeft[i]))
        DoubleSupportTime = np.nanmedian(data['Left foot']['Gait Phases']['Double support time (samples)'])/sample_frequency
        data['Spatiotemporals']['Double support time left (s)'] = round(DoubleSupportTime,2)
    
    else:
        data['Spatiotemporals']['Double support time right (s)'] = np.nan
        data['Spatiotemporals']['Double support time left (s)'] = np.nan
    return data





def velocity(data, errors, showfigure):
    # INPUT:  EFacc: acceleration data in Earth Frame (g)
    #         indexMidStance: index numbers of point identified as Mid-Stance
    #         indexNonActive: index numbers of point identified as Non-Active
    #         time: time passed since start of measurement (s)
    #         sample_frequency: sample frequency (Hz)
    #         showfigure: either 'view' or 'hide' in order to show figures or not
    # OUTPUT: ['derived']['Velocity (m/s)']: full velocity profile over time (m/s)
    
    
    def functiongaitspeed(EFacc, indexMidStance, indexNonActive, indexIC, indexTC, showfigure):
        # Detect stationary periods
        # Stationary period = Mid-Stance en Non-Active time
        magACC = np.sqrt((EFacc[:,0])**2 + (EFacc[:,1])**2 + (EFacc[:,2])**2)
        
        stationary = np.zeros((len(EFacc)), dtype = 'int64')
        for i in range(0,len(indexMidStance)):
            stationary[indexMidStance[i]] = 1
        for i in range(0,len(indexNonActive)):
            stationary[indexNonActive[i]] = 1
        
        # Fill gaps smaller than 'ntolerance' samples
        ntolerance = 10
        b = np.array((np.diff(stationary)), dtype = int)
        c = np.where(b != 0)[0]
        for i in range(1,len(c)):
            if (c[i]-c[i-1]<ntolerance) & (stationary[c[i-1]] > 0):
                stationary[c[i-1]+1:c[i]]=1
        c = np.where(b == 0)[0]
        for i in range(1,len(c)):
            if (c[i]-c[i-1]<ntolerance) & (stationary[c[i-1]] < 0):
                stationary[c[i-1]+1:c[i]]=1
        
        # Define start and end of stationary periods
        stationaryStart = (np.asarray(np.argwhere(np.append(0, np.diff(stationary)) == 1))).flatten()
        stationaryEnd = (np.asarray(np.argwhere(np.append(0, np.diff(stationary)) == -1))).flatten()
        if stationary[0] == 1:
            stationaryStart = np.insert(stationaryStart,0 , 0)
        if stationaryStart[-1] > stationaryEnd[-1]:
            stationaryStart = stationaryStart[0:-1]
        if (len(stationaryStart) > len(stationaryEnd)) & (stationaryStart[-1] > stationaryEnd[-1]):
            stationaryEnd = np.append(stationaryEnd, len(EFacc))
        
        # Plot sensor data and stationary periods - debug figure
        if showfigure == 'view':
            plt.figure()
            plt.plot(time, magACC, 'r-', label = 'Magnitude ACC')
            # plt.plot(time, stationarymag*5, 'b', label = 'Stationary - magnitude based')
            plt.plot(time, stationary*10, 'k', label = 'Stationary - mid-stance based')
            plt.plot(time[stationaryStart], stationary[stationaryStart], 'kv')
            plt.plot(time[stationaryEnd], stationary[stationaryEnd], 'k^')
            plt.title('Accelerometer - Earth frame')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (m/s/s)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
        
        # Compute translational velocities
        # Integrate acceleration to yield velocity
        velocity = np.zeros((np.shape(EFacc)))
        for t in range(1,len(velocity)):
            velocity[t,:] = velocity[(t-1),:] + (EFacc[t,:] * 1/sample_frequency)
            if stationary[t] == 1:
                velocity[t,:] = [0, 0, 0] # force zero velocity when foot stationary
        
        # Compute integral drift during non-stationary periods
        velDrift = np.zeros((np.shape(velocity)))
        
        for i in range(0,np.size(stationaryEnd)):
            driftRate = velocity[stationaryEnd[i]-1, :] / (stationaryEnd[i] - stationaryStart[i])
            enum = (np.array(range(1,(stationaryEnd[i] - stationaryStart[i])))).transpose()
            drift = (np.array([enum * driftRate[0], enum * driftRate[1], enum*driftRate[2]])).transpose()
            velDrift[stationaryStart[i]:stationaryEnd[i]-1, :] = drift
            
        # Remove integral drift
        velocity = velocity - velDrift
        
        # Compute magnitude velocity
        magnitudeVelocity = np.sqrt(velocity[:,0]*velocity[:,0] + velocity[:,1]*velocity[:,1] + velocity[:,2]*velocity[:,2])
        
        # Plot translational velocity
        if showfigure == 'view':
            plt.figure()
            plt.plot(time, velocity[:,0], 'r-', label = 'Velocity - X')
            plt.plot(time, velocity[:,1], 'g', label = 'Velocity - Y')
            plt.plot(time, velocity[:,2], 'b', label = 'Velocity - Z')
            plt.plot(time, magnitudeVelocity, ':k', label = 'Velocity - magnitude')
            plt.title('Velocity - Earth frame')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (m/s)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
        
        
        # Velocity integration based on:
        # Mariani, Benoit & Hoskovec, Constanze & Rochat, Stephane & Büla, Christophe & Penders, Julien & Aminian, Kamiar. (2010). 3D gait assessment in young and elderly subjects using foot-worn inertial sensors. Journal of biomechanics. 43. 2999-3006. 10.1016/j.jbiomech.2010.07.003. 
        # Full foot-flat periods
        tff = np.argwhere(stationary==1).flatten()
        
        velocity2 = np.zeros((np.shape(EFacc)))
        for t in range(1,len(velocity2)):
            velocity2[t,:] = velocity2[(t-1),:] + (EFacc[t,:] * 1/sample_frequency)
                
        x_observed = np.unique(tff)
        y_observed = velocity2[x_observed]
        x = np.arange(0, len(velocity2))

        y0 = interpolate.PchipInterpolator(x_observed, y_observed[:,0])
        y1 = interpolate.PchipInterpolator(x_observed, y_observed[:,1])
        y2 = interpolate.PchipInterpolator(x_observed, y_observed[:,2])
        # y0 = interpolate.pchip_interpolate(x_observed, y_observed[:,0], x, der=0)
        # y1 = interpolate.pchip_interpolate(x_observed, y_observed[:,1], x, der=0)
        # y2 = interpolate.pchip_interpolate(x_observed, y_observed[:,2], x, der=0)
        
        velocity2_corr = np.zeros(shape = velocity2.shape)
        velocity2_corr[x,0] = velocity2[x,0]-y0(x)
        velocity2_corr[x,1] = velocity2[x,1]-y1(x)
        velocity2_corr[x,2] = velocity2[x,2]-y2(x)
        
        for t in range(1,len(velocity2)):
            if stationary[t] == 1:
                velocity2_corr[t,:] = [0,0,0] # force zero velocity when foot is stationary
            
        if showfigure == 'view':
            plt.figure()
            plt.plot()
            # plt.plot(x_observed, y_observed[:,0], "k.", label="foot-flat")
            # plt.plot(indexIC, velocity2_corr[indexIC,:], 'cx', label= 'Initial contact')
            plt.plot(x, y0(x), label="0 pchip interpolation function")
            plt.plot(x, y1(x), label="1 pchip interpolation function")
            plt.plot(x, y2(x), label="2 pchip interpolation function")
            # plt.plot(velocity[:,0], label='velocity linear integration - X')
            # plt.plot(velocity[:,1], label='velocity linear integration - Y')
            # plt.plot(velocity[:,2], label='velocity linear integration - Z')

            plt.plot(velocity2[:,0], label='velocity before dedrift compensation - X')
            plt.plot(velocity2_corr[:,0], label='velocity after dedrift compensation - X')
            plt.plot(velocity2[:,1], label='velocity before dedrift compensation - Y')
            plt.plot(velocity2_corr[:,1], label='velocity after dedrift compensation - Y')
            plt.plot(velocity2[:,2], label='velocity before dedrift compensation - Z')
            plt.plot(velocity2_corr[:,2], label='velocity after dedrift compensation - Z')
            plt.legend()
            plt.show()
            
        return velocity, velocity2_corr


    
    sample_frequency = data['Sample Frequency (Hz)']
    time = data['Timestamp']
    
    sides = ['left', 'right', 'lumbar']
    for i in range(0, len(sides)):
        if sides[i] == 'left' and errors['No walking period'] == False:
            EFacc = data['Left foot']['raw']['Accelerometer Earth Frame']
            indexMidStance = data['Left foot']['Gait Phases']['Mid-Stance']
            indexNonActive = data['Left foot']['Gait Phases']['Non-Active samples']
            indexIC = data['Left foot']['Gait Events']['Initial Contact']
            indexTC = data['Left foot']['Gait Events']['Terminal Contact']
            
            data['Left foot']['derived']['Velocity - linear (m/s)'], data['Left foot']['derived']['Velocity (m/s)'] = functiongaitspeed(EFacc, indexMidStance, indexNonActive, indexIC, indexTC, showfigure)
            
        elif sides[i] == 'right' and errors['No walking period'] == False:
            EFacc = data['Right foot']['raw']['Accelerometer Earth Frame']
            indexMidStance = data['Right foot']['Gait Phases']['Mid-Stance']
            indexNonActive = data['Right foot']['Gait Phases']['Non-Active samples']
            indexIC = data['Right foot']['Gait Events']['Initial Contact']
            indexTC = data['Right foot']['Gait Events']['Terminal Contact']
            
            data['Right foot']['derived']['Velocity - linear (m/s)'], data['Right foot']['derived']['Velocity (m/s)'] = functiongaitspeed(EFacc, indexMidStance, indexNonActive, indexIC, indexTC, showfigure)
        
        elif sides[i] == 'lumbar' and (data['trialType'] == 'L-test' or data['trialType']=='STS-transfer'):
            EFacc = data['Lumbar']['raw']['Accelerometer Earth Frame']
            indexMidStance = np.array([], dtype=int)
            indexNonActive = np.array([0, len(EFacc)-1], dtype=int)
            indexIC = np.array([], dtype=int)
            indexTC = np.array([], dtype=int)
            
            data['Lumbar']['derived']['Velocity - linear (m/s)'], data['Lumbar']['derived']['Velocity (m/s)'] = functiongaitspeed(EFacc, indexMidStance, indexNonActive, indexIC, indexTC, showfigure)
            
                    
    return data




def turnidentification(data):
    from scipy import signal
    
    if 'Lumbar' not in data['Missing Sensors']:
        fGYRz = data['Lumbar']['derived']['Gyroscope Earth frame'][:,2]
    elif 'Sternum' not in data['Missing Sensors']:
        fGYRz = data['Sternum']['derived']['Gyroscope Earth frame'][:,2]
    sample_frequency = data['Sample Frequency (Hz)']
    
    # Turn identification algorithm based on:
    # El-Gohary, M., Pearson, S., McNames, J., Mancini, M., Horak, F., Mellone, S., & Chiari, L. (2014). 
    # Continuous monitoring of turning in patients with movement disability.
    # Sensors, 14(1), 356-369.
    
    # Threshold (adapted value) in case at least one moment of at least 70 degrees/s turning has taken place, otherwise, assume only straight walking (not in El-Gohary et al.)
    if np.max(np.abs(fGYRz)) > 70/180*np.pi:
        threshold = 0.3 * np.max(np.abs(fGYRz))
        peaks = (signal.find_peaks(np.abs(fGYRz)))[0]
        idxturnpeaks = peaks[np.argwhere(np.abs(fGYRz[peaks]) > threshold).flatten()]
        
        idxturn=np.array([], dtype = int)
        
        # Finde preceding and following 5 deg/s crossings
        crossing = 5*np.pi/180
        for i in range(0,len(idxturnpeaks)):
            initialcrossing = np.argwhere(np.abs(fGYRz[0:idxturnpeaks[i]]) < crossing)
            if len(initialcrossing)>0:
                initialcrossing = initialcrossing[-1]
            else:
                initialcrossing = idxturnpeaks[i]-1
            finalcrossing = idxturnpeaks[i]+np.argwhere(np.abs(fGYRz[idxturnpeaks[i]:-1]) < crossing)
            if len(finalcrossing)>0:
                finalcrossing = finalcrossing[0]
            else:
                finalcrossing = len(fGYRz)-1
                    
            if len(idxturn)<1:
                # If turn duration > 10 s or turn duration < 0.05 s; eliminate turn
                # if finalcrossing - initialcrossing < 10 * sample_frequency or finalcrossing - initialcrossing > 0.05 * sample_frequency:
                if (finalcrossing - initialcrossing < (10 * sample_frequency)) and ((finalcrossing - initialcrossing) > (np.nanmean([data['Spatiotemporals']['Step time left (s)'], data['Spatiotemporals']['Step time right (s)']]) * sample_frequency)):    
                    idxturn = np.append(idxturn, np.arange(initialcrossing, finalcrossing+1))
            
            elif len(idxturn)>0:
                # If intra-turn duration <0.1 s and previous turn is in same direction then combine with previous turn
                if initialcrossing < idxturn[-1] + 0.1 * sample_frequency:
                    if (fGYRz[idxturnpeaks[i]] < 0 and fGYRz[idxturnpeaks[i-1]] < 0) or (fGYRz[idxturnpeaks[i]] > 0 and fGYRz[idxturnpeaks[i-1]] > 0):
                        idxturn = np.append(idxturn, np.arange(idxturn[-1], finalcrossing+1))
                else:
                    # If turn duration > 10 s or turn duration < 0.05 s; eliminate turn
                    # if finalcrossing - initialcrossing < 10 * sample_frequency or finalcrossing - initialcrossing > 0.05 * sample_frequency:
                    if ((finalcrossing - initialcrossing) < (10 * sample_frequency)) and ((finalcrossing - initialcrossing) > (np.nanmean([data['Spatiotemporals']['Step time left (s)'], data['Spatiotemporals']['Step time right (s)']]) * sample_frequency)):
                        idxturn = np.append(idxturn, np.arange(initialcrossing, finalcrossing+1))
                
                # If turn duration > 10 s or turn duration < 0.05 s; eliminate turn
                if finalcrossing - initialcrossing > 10 * sample_frequency:
                    pass
    else:
        idxturn=np.array([])
    
    if 'Lumbar' not in data['Missing Sensors']:
        data['Lumbar']['derived']['Change in Walking Direction samples'] = np.sort(np.unique(idxturn))
    elif 'Sternum' not in data['Missing Sensors']:
        data['Sternum']['derived']['Change in Walking Direction samples'] = np.sort(np.unique(idxturn))
        
    return data





def turnparameters(data):
    # INPUT:  GYRlumbar: gyroscope data of the lumbar sensor
    #         idxturn: index numbers of identified turning movement
    #         sample_frequency: sample frequency (Hz)
    # OUTPUT: ['Spatiotemporals']['Peak turn velocity - average all peaks (deg/s)']
    #         ['Spatiotemporals']['Peak turn velocity - standard deviation all peaks (deg/s)']
    #         ['Spatiotemporals']['Peak turn velocity - average 180 deg peaks (deg/s)']
    #         ['Spatiotemporals']['Peak turn velocity - standard deviation 180 deg peaks (deg/s)']
    #         ['Lumbar']['derived']['Peak turn velocity - all turns (deg/s)']

    if 'Lumbar' not in data['Missing Sensors']:
        GYRlumbar = data['Lumbar']['raw']['Gyroscope']
        EUL = data['Lumbar']['raw']['Orientation Euler']
        idxturn = data['Lumbar']['derived']['Change in Walking Direction samples']
    elif 'Sternum' not in data['Missing Sensors']:
        GYRlumbar = data['Sternum']['raw']['Gyroscope']
        EUL = data['Sternum']['raw']['Orientation Euler']
        idxturn = data['Sternum']['derived']['Change in Walking Direction samples']
    
    if len(idxturn) > 0:
        diffturnidx = np.diff(idxturn)
        newturnstart = np.array([idxturn[0]], dtype = int)
        newendturn = np.array([idxturn[-1]], dtype = int)
        for i in range(0,len(diffturnidx)):
            if diffturnidx[i] > 1:
                newturnstart = np.append(newturnstart, idxturn[i+1])
                newendturn = np.append(newendturn, idxturn[i])
        newendturn = np.sort(newendturn)
        
        yawunwrap = np.unwrap(EUL[:,2],150)
        allPeakTurnVelocity = np.array([])
        peaks180deg = np.array([])
        for j in range(0,len(newturnstart)):
            peak = np.max([ np.abs( np.max( GYRlumbar[newturnstart[j]:newendturn[j], 0] ) ), np.abs( np.min( GYRlumbar[newturnstart[j]:newendturn[j], 0] ) ) ]) * 180/np.pi
            allPeakTurnVelocity = np.append(allPeakTurnVelocity, peak)
            if np.abs(yawunwrap[newturnstart[j]]-yawunwrap[newendturn[j]]) > 150:
                peaks180deg = np.append(peaks180deg, peak)
                
        data['Spatiotemporals']['Peak turn velocity - average all peaks (deg/s)'] = round(np.mean(allPeakTurnVelocity), 2)
        data['Spatiotemporals']['Peak turn velocity - standard deviation all peaks (deg/s)'] = round(np.std(allPeakTurnVelocity, dtype=np.float64), 2)
        data['Spatiotemporals']['Peak turn velocity - average 180 deg peaks (deg/s)'] = round(np.mean(peaks180deg), 2)
        data['Spatiotemporals']['Peak turn velocity - standard deviation 180 deg peaks (deg/s)'] = round(np.std(peaks180deg, dtype=np.float64), 2)
        if 'Lumbar' not in data['Missing Sensors']:
            data['Lumbar']['derived']['Peak turn velocity - all turns (deg/s)'] = allPeakTurnVelocity
            data['Lumbar']['derived']['Peak turn velocity - 180 deg turns (deg/s)'] = peaks180deg
        elif 'Sternum' not in data['Missing Sensors']:
            data['Sternum']['derived']['Peak turn velocity - all turns (deg/s)'] = allPeakTurnVelocity
            data['Sternum']['derived']['Peak turn velocity - 180 deg turns (deg/s)'] = peaks180deg
    else:
        data['Spatiotemporals']['Peak turn velocity - average (deg/s)'] = np.nan
        data['Spatiotemporals']['Peak turn velocity - standard deviation (deg/s)'] = np.nan
        if 'Lumbar' not in data['Missing Sensors']:
            data['Lumbar']['derived']['Peak turn velocity - all turns (deg/s)'] = np.nan
        elif 'Sternum' not in data['Missing Sensors']:
            data['Sternum']['derived']['Peak turn velocity - all turns (deg/s)'] = np.nan
            
    return data





def positionestimation(data, errors, showfigure):
    # INPUT:  velocity: full velocity profile over time (m/s)
    #         time: time passed since start of measurement (s)
    #         sample_frequency: sample frequency (Hz)
    # OUTPUT: ['derived']['Position (m)']: position in global frame with respect to initial position (m)
    #
    
    def functionpositionestimation(velocity, time, sample_frequency, showfigure):
        # Integrate velocity to yield position
        position = np.zeros((np.shape(velocity)))
        for t in range(1,len(position)):
            position[t,:] = position[t-1,:] + (velocity[t,:] * 1/sample_frequency) # integrate velocity to yield position
            # if np.all(velocity[t,:]) == False:
            #     position[t,2] = 0 # ZUPT in z-axis
        
        # Plot translational position - debug figure
        if showfigure == 'view':
            plt.figure()
            plt.plot(time, position[:,0], 'r', label = 'X')
            plt.plot(time, position[:,1], 'g', label = 'Y')
            plt.plot(time, position[:,2], 'b', label = 'Z')
            # plt.plot(time[indexHeelStrike], position[indexHeelStrike,0],'mv', label = 'Heel strike')
            # plt.plot(time[indexHeelOff], position[indexHeelOff,0],'g^', label = 'Heel Off')
            # plt.plot(time[indexTurns], position[indexTurns], 'r.', label = 'Turning')
            # if contains(inputname(1),{'left'})
            # title('Position left sensor');
            # elseif contains(inputname(1),{'right'})
            # title('Position right sensor');
            # else
            plt.title('Position sensor')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        return position
    
    
    
    sample_frequency = data['Sample Frequency (Hz)']
    time = data['Timestamp']
    
    sides = ['left', 'right', 'lumbar']
    for i in range(0, len(sides)):
        if sides[i] == 'left' and errors['No walking period'] == False:
            velocity = data['Left foot']['derived']['Velocity (m/s)']
            data['Left foot']['derived']['Position (m)'] = functionpositionestimation(velocity, time, sample_frequency, showfigure)
            # velocity = data['Left foot']['derived']['Velocity - linear (m/s)']
            # data['Left foot']['derived']['Position - linear (m)'] = functionpositionestimation(velocity, time, sample_frequency, showfigure)
            
        elif sides[i] == 'right' and errors['No walking period'] == False:
            velocity = data['Right foot']['derived']['Velocity (m/s)']
            data['Right foot']['derived']['Position (m)'] = functionpositionestimation(velocity, time, sample_frequency, showfigure)
            velocity = data['Right foot']['derived']['Velocity - linear (m/s)']
            data['Right foot']['derived']['Position - linear (m)'] = functionpositionestimation(velocity, time, sample_frequency, showfigure)
            
        elif sides[i] == 'lumbar' and (data['trialType'] == 'L-test' or data['trialType'] == 'STS-transfer'):
            velocity = data['Lumbar']['derived']['Velocity (m/s)']
            data['Lumbar']['derived']['Position (m)'] = functionpositionestimation(velocity, time, sample_frequency, showfigure)
            # velocity = data['Lumbar']['derived']['Velocity - linear (m/s)']
            # data['Lumbar']['derived']['Position - linear (m)'] = functionpositionestimation(velocity, time, sample_frequency, showfigure)
            
    return data




       
def stridelength(data, errors, showfigure):
    # INPUT:  position: position in global frame with respect to initial position (m)
    #         indexToeOff: index numbers of Toe-Off points
    #         indexHeelStrike: index numbers of Heel-Strike points
    #         indexTurns: index numbers of samples that are within a change of
    #         sample_frequency: sample frequency (Hz)
    # OUTPUT: ['derived']['Stride length - all strides (m)']: index numbers of start [:,0] and stop [:,1] indices and corresponding calculated stride lengths [:,2] of all identified strides (m)
    ##         ['derived']['Stride length - straight walking (m)']: index numbers of start [:,0] and stop [:,1] indices and corresponding calculated stride lengths [:,2] of all identified strides (m) that are not within change of direction (indexTurns) (m)
    ##         ['derived']['Stride length - in turn (m)']: index numbers of start [:,0] and stop [:,1] indices and corresponding calculated stride lengths [:,2] of all identified strides (m) that are within change of direction (indexTurns) (m)
    #         ['derived']['Stride length - average all strides (m)']: average distance between Heel-Off and Heel-Strike of one foot in all strides (m)
    ##         ['derived']['Stride length - average straight walking (m)']['derived']: average distance between Heel-Off and Heel-Strike of one foot in straight walking (m)
    
        
    def functionstridelength(position, indexToeOff, indexHeelStrike, indexTurns, sample_frequency, showfigure):
        # The distance between two points in a three dimensional - 3D - coordinate system can be calculated as. d = sqrt((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)
        allstrides = np.zeros((len(indexHeelStrike),3))
        
        for i in range(0,len(indexHeelStrike)):
            lastToeOff = (np.asarray(np.where(indexToeOff < indexHeelStrike[i]))[-1])
            lastHeelStrike = (np.asarray(np.where(indexHeelStrike < indexHeelStrike[i]))[-1])
            # if len(lastToeOff) > 0:
            #     lastToeOff = (np.asarray(np.where(indexToeOff < indexHeelStrike[i]))[-1][-1])
            #     allstrides[i,0] = indexToeOff[lastToeOff]
            #     allstrides[i,1] = indexHeelStrike[i]
            if len(lastHeelStrike) > 0:
                lastHeelStrike = (np.asarray(np.where(indexHeelStrike < indexHeelStrike[i]))[-1][-1])
                allstrides[i,0] = indexHeelStrike[lastHeelStrike]
                allstrides[i,1] = indexHeelStrike[i]
        try:
            remove = np.array([], dtype = int)
            for i in range(0, len(allstrides)):
                if np.all(allstrides[i,0:2]) == False:
                    remove = np.append(remove, i)
            if len(remove)>0:
                allstrides = np.delete(allstrides, remove, axis=0)
        finally:
            allstrides = allstrides
        
        # All strides
        stridelength = np.zeros((len(allstrides),3))
        x1 = position[allstrides[:,0].astype(int),0]
        y1 = position[allstrides[:,0].astype(int),1]
        z1 = position[allstrides[:,0].astype(int),2]
    
        x2 = position[allstrides[:,1].astype(int),0]
        y2 = position[allstrides[:,1].astype(int),1]
        z2 = position[allstrides[:,1].astype(int),2]
        
        # distances(i) = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2);
        stridelength[:,0] = np.abs(x2-x1)
        stridelength[:,1] = np.abs(y2-y1)
        stridelength[:,2] = np.abs(z2-z1)
        
        #Discard strides smaller than 10 cm as artefact or larger than 1.8*the mean stridelength
        allstrides[:,2] = np.sqrt((stridelength[:,0])**2 + (stridelength[:,1])**2)
        mas = np.nanmean(allstrides[:,2])
        for i in range(len(allstrides)):
            if allstrides[i,2]<0.20 or allstrides[i,2]>1.6*mas:
                allstrides[i,:]=np.nan
        
        # Set toe off event at 0th column
        for i in range(0,len(allstrides[:,1])):
            lastToeOff = (np.asarray(np.where(indexToeOff < allstrides[i,1]))[-1])
            if len(lastToeOff) > 0:
                lastToeOff = (np.asarray(np.where(indexToeOff < allstrides[i,1]))[-1][-1])
                allstrides[i,0] = indexToeOff[lastToeOff]
            
        allstrides = allstrides[~np.isnan(allstrides[:,2])]
        averageallstridelengths = np.nanmedian(allstrides[:,2])
            
    
        # Split strides on strides made in straight walking and strides made in turns
        straightwalking = np.array([0,0,0])
        inturn = np.array([0,0,0])
        for i in range(0, len(allstrides)):
            if np.all(np.isin(allstrides[i,0:2], indexTurns, invert = True)): # both gait events (heel off and heel strike) of this stride are not identified in the turn indices in case this statement is true
                straightwalking = np.vstack((straightwalking, allstrides[i,:]))
            
            else:
                inturn = np.vstack((inturn, allstrides[i,:]))
        
        straightwalking = np.delete(straightwalking, 0, axis =0) # remove first row of nan's
        inturn = np.delete(inturn, 0, axis =0)
        if np.shape(straightwalking) == (2,):
            straightwalking = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        if np.shape(inturn) == (2,):
            inturn = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
    
        # Figure position HS, HO, Turn indices
        if showfigure == 'view':
            plt.figure()
            plt.plot(position[:,0], 'b', label = 'Position (X)')
            if len(indexTurns) > 0:
                plt.plot(indexTurns,position[indexTurns,0],'.r', label = 'Turn')
            plt.plot(straightwalking[:,0],position[straightwalking[:,0].astype(int),0],'g^', label = 'Heel Off - straight walking')
            plt.plot(straightwalking[:,1],position[straightwalking[:,1].astype(int),0],'rv', label = 'Heel Strike - straight walking')
            if np.isnan(inturn).all() == False:
                plt.plot(inturn[:,0],position[inturn[:,0].astype(int),0]+0.02,'k^', label = 'Heel Off - in turn')
                plt.plot(inturn[:,1],position[inturn[:,1].astype(int),0]+0.02,'kv', label = 'Heel Strike - in turn')
            plt.title('Position sensor')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        return allstrides, straightwalking, inturn, averageallstridelengths #, averageallstridelengths, averagestridelengthSW
    
    
    sample_frequency = data['Sample Frequency (Hz)']
    
    sides = ['left', 'right']
    if errors['No walking period'] == False:
        for i in range(0, len(sides)):
            if sides[i] == 'left':
                position = data['Left foot']['derived']['Position (m)']
                indexToeOff = data['Left foot']['Gait Events']['Terminal Contact']
                indexHeelStrike = data['Left foot']['Gait Events']['Initial Contact']
                if 'Lumbar' not in data['Missing Sensors']:
                    indexTurns = data['Lumbar']['derived']['Change in Walking Direction samples']
                elif 'Sternum' not in data['Missing Sensors']:
                    indexTurns = data['Sternum']['derived']['Change in Walking Direction samples']
                else:
                    indexTurns = []
                data['Left foot']['derived']['Stride length - all strides (m)'], data['Left foot']['derived']['Stride length - straight walking (m)'], data['Left foot']['derived']['Stride length - in turn (m)'], data['Left foot']['derived']['Stride length - average all strides (m)'] = functionstridelength(position, indexToeOff, indexHeelStrike, indexTurns, sample_frequency, showfigure)
                
            elif sides[i] == 'right':
                position = data['Right foot']['derived']['Position (m)']
                indexToeOff = data['Right foot']['Gait Events']['Terminal Contact']
                indexHeelStrike = data['Right foot']['Gait Events']['Initial Contact']
                if 'Lumbar' not in data['Missing Sensors']:
                    indexTurns = data['Lumbar']['derived']['Change in Walking Direction samples']
                elif 'Sternum' not in data['Missing Sensors']:
                    indexTurns = data['Sternum']['derived']['Change in Walking Direction samples']
                else:
                    indexTurns = []
                data['Right foot']['derived']['Stride length - all strides (m)'], data['Right foot']['derived']['Stride length - straight walking (m)'], data['Right foot']['derived']['Stride length - in turn (m)'], data['Right foot']['derived']['Stride length - average all strides (m)'] = functionstridelength(position, indexToeOff, indexHeelStrike, indexTurns, sample_frequency, showfigure)
    else:
        data['Left foot']['derived']['Stride length - all strides (m)'] = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        data['Left foot']['derived']['Stride length - straight walking (m)'] = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        data['Left foot']['derived']['Stride length - in turn (m)'] = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        data['Left foot']['derived']['Stride length - average all strides (m)'] = np.nan
        data['Right foot']['derived']['Stride length - all strides (m)'] = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        data['Right foot']['derived']['Stride length - straight walking (m)'] = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        data['Right foot']['derived']['Stride length - in turn (m)'] = np.transpose(np.array([[np.nan],[np.nan],[np.nan]]))
        data['Right foot']['derived']['Stride length - average all strides (m)'] = np.nan
    
    # Steady state gait
    stridesinturn = np.vstack((data['Left foot']['derived']['Stride length - in turn (m)'], data['Right foot']['derived']['Stride length - in turn (m)']))
    stridesinturn = stridesinturn[stridesinturn[:, 0].argsort()]
    
    # Method: Muir et al.
    # Filtered on not-turning strides and remove every 2 strides (so 1 left and 1 right stride) around the turning strides
    # Additionally, remove the first 2 and last 2 strides of the trial
    data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'] = np.transpose(np.array([[], [], []]))
    data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'] = np.transpose(np.array([[], [], []]))
    
    remove = np.array([])
    for j in range(0, len(stridesinturn)):
        lastHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Left foot']['derived']['Stride length - straight walking (m)'][:,1] < stridesinturn[j,1] ]
        if lastHSleft.size > 0:
            lastHSleft=lastHSleft[-1]
        else:
            lastHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][0,1]
        
        remove = np.append(remove, lastHSleft)
        
        lastHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Right foot']['derived']['Stride length - straight walking (m)'][:,1] < stridesinturn[j,1] ]
        if lastHSright.size > 0:
            lastHSright=lastHSright[-1]
        else:
            lastHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][0-1,1]
        
        remove = np.append(remove, lastHSright)
        
               
        firstHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Left foot']['derived']['Stride length - straight walking (m)'][:,1] > stridesinturn[j,1] ]
        if firstHSleft.size > 0:
            firstHSleft=firstHSleft[0]
        else:
            firstHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][-1,1]
        
        remove = np.append(remove, firstHSleft)
        
        firstHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Right foot']['derived']['Stride length - straight walking (m)'][:,1] > stridesinturn[j,1] ]
        if firstHSright.size > 0:
            firstHSright=firstHSright[0]
        else:
            firstHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][-1,1]
        
        remove = np.append(remove, firstHSright)
        
        remove = np.sort(np.unique(remove))
    for j in range(2, len(data['Left foot']['derived']['Stride length - straight walking (m)'])-2):
        if data['Left foot']['derived']['Stride length - straight walking (m)'][j,1] not in remove:
            data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'] = np.vstack((data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'], data['Left foot']['derived']['Stride length - straight walking (m)'][j,:]))
    for j in range(2, len(data['Right foot']['derived']['Stride length - straight walking (m)'])-2):
        if data['Right foot']['derived']['Stride length - straight walking (m)'][j,1] not in remove:
            data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'] = np.vstack((data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'], data['Right foot']['derived']['Stride length - straight walking (m)'][j,:]))
         
    
    # Method: No 1 stride around turn
    # Gait speed straight walking 1 stride around turn removed
    # Additionally, remove the first and last stride of the trial
    data['Left foot']['derived']['Stride length - no 1 steps around turn (m)'] = np.transpose(np.array([[], [], []]))
    data['Right foot']['derived']['Stride length - no 1 steps around turn (m)'] = np.transpose(np.array([[], [], []]))
    
    remove = np.array([])
    for j in range(0, len(stridesinturn)):
        
        lastHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Left foot']['derived']['Stride length - straight walking (m)'][:,1] < stridesinturn[j,1] ]
        if lastHSleft.size > 0:
            lastHSleft=lastHSleft[-1]
        else:
            lastHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][0,1]
        
        lastHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Right foot']['derived']['Stride length - straight walking (m)'][:,1] < stridesinturn[j,1] ]
        if lastHSright.size > 0:
            lastHSright=lastHSright[-1]
        else:
            lastHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][0,1]
        
        lastHS = np.argmax([lastHSleft, lastHSright])
        remove = np.append(remove, [lastHSleft, lastHSright][lastHS])
        
        
        firstHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Left foot']['derived']['Stride length - straight walking (m)'][:,1] > stridesinturn[j,1] ]
        if firstHSleft.size > 0:
            firstHSleft=firstHSleft[0]
        else:
            firstHSleft = data['Left foot']['derived']['Stride length - straight walking (m)'][-1,1] #[0,0]
        
        firstHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][:,1][ data['Right foot']['derived']['Stride length - straight walking (m)'][:,1] > stridesinturn[j,1] ]
        if firstHSright.size > 0:
            firstHSright=firstHSright[0]
        else:
            firstHSright = data['Right foot']['derived']['Stride length - straight walking (m)'][-1,1] #[0,0]
        
        firstHS = np.argmin([firstHSleft, firstHSright])
        remove = np.append(remove, [firstHSleft, firstHSright][firstHS])
        
    for j in range(1, len(data['Left foot']['derived']['Stride length - straight walking (m)'])-1):
        if data['Left foot']['derived']['Stride length - straight walking (m)'][j,1] not in remove:
            data['Left foot']['derived']['Stride length - no 1 steps around turn (m)'] = np.vstack((data['Left foot']['derived']['Stride length - no 1 steps around turn (m)'], data['Left foot']['derived']['Stride length - straight walking (m)'][j,:]))
    for j in range(1, len(data['Right foot']['derived']['Stride length - straight walking (m)'])-1):
        if data['Right foot']['derived']['Stride length - straight walking (m)'][j,1] not in remove:
            data['Right foot']['derived']['Stride length - no 1 steps around turn (m)'] = np.vstack((data['Right foot']['derived']['Stride length - no 1 steps around turn (m)'], data['Right foot']['derived']['Stride length - straight walking (m)'][j,:]))
    
    
    return data

def gaitspeed(data, errors):
    # Gait speed all strides = stride length / stride time
    # First stridetime from heel off - heel strike, then heel strike - heel strike
    try:
        HOl = data['Left foot']['Gait Events']['Heel Off'][np.max(np.where(data['Left foot']['Gait Events']['Heel Off']<data['Left foot']['derived']['Stride length - all strides (m)'][0,1]))]
    except ValueError:
        HOl = 0
    try:
        HOr = data['Right foot']['Gait Events']['Heel Off'][np.max(np.where(data['Right foot']['Gait Events']['Heel Off']<data['Right foot']['derived']['Stride length - all strides (m)'][0,1]))]
    except ValueError:
        HOr = 0
    
    timel = data['Left foot']['derived']['Stride length - all strides (m)'][0,1]-HOl
    timer = data['Right foot']['derived']['Stride length - all strides (m)'][0,1]-HOr
    data['Left foot']['derived']['Stride time per stride - all strides (s)'] = np.diff(data['Left foot']['derived']['Stride length - all strides (m)'][:,1])/data['Sample Frequency (Hz)'] # Heel strike to heel strike 
    data['Right foot']['derived']['Stride time per stride - all strides (s)'] = np.diff(data['Right foot']['derived']['Stride length - all strides (m)'][:,1])/data['Sample Frequency (Hz)'] # Heel strike to heel strike
    
    data['Left foot']['derived']['Stride time per stride - all strides (s)'] = np.insert(data['Left foot']['derived']['Stride time per stride - all strides (s)'], 0, timel, axis=0)
    data['Right foot']['derived']['Stride time per stride - all strides (s)'] = np.insert(data['Right foot']['derived']['Stride time per stride - all strides (s)'], 0, timer, axis=0)
    
    data['Left foot']['derived']['Stride time per stride - all strides (s)'][data['Left foot']['derived']['Stride time per stride - all strides (s)']>1.5*np.nanmedian(data['Left foot']['derived']['Stride time per stride - all strides (s)'])] = np.nan
    data['Right foot']['derived']['Stride time per stride - all strides (s)'][data['Right foot']['derived']['Stride time per stride - all strides (s)']>1.5*np.nanmedian(data['Right foot']['derived']['Stride time per stride - all strides (s)'])] = np.nan
    
    data['Left foot']['derived']['Gait speed per stride (m/s)'] = np.copy(data['Left foot']['derived']['Stride length - all strides (m)'])
    data['Right foot']['derived']['Gait speed per stride (m/s)'] = np.copy(data['Right foot']['derived']['Stride length - all strides (m)'])
    
    data['Left foot']['derived']['Gait speed per stride (m/s)'][:,2] = data['Left foot']['derived']['Stride length - all strides (m)'][:,2] / (data['Left foot']['derived']['Stride time per stride - all strides (s)'] ) # stride by stride
    data['Right foot']['derived']['Gait speed per stride (m/s)'][:,2] = data['Right foot']['derived']['Stride length - all strides (m)'][:,2] / (data['Right foot']['derived']['Stride time per stride - all strides (s)'] ) 
    
    # Gait speed straight walking
    data['Left foot']['derived']['Stride time per stride - straight walking (s)'] = np.array([])
    for i in range(0,len(data['Left foot']['derived']['Stride length - straight walking (m)'])):
        lastHS = data['Left foot']['derived']['Stride length - all strides (m)'][:,1][ data['Left foot']['derived']['Stride length - all strides (m)'][:,1] < data['Left foot']['derived']['Stride length - straight walking (m)'][i,1] ] 
        if lastHS.size > 0:
            lastHS=lastHS[-1]
        else:
            lastHS = data['Left foot']['derived']['Stride length - straight walking (m)'][i,0]
        sw = data['Left foot']['derived']['Stride length - straight walking (m)'][i,1] - lastHS
        data['Left foot']['derived']['Stride time per stride - straight walking (s)'] = np.append(data['Left foot']['derived']['Stride time per stride - straight walking (s)'], sw/data['Sample Frequency (Hz)']) 
    data['Right foot']['derived']['Stride time per stride - straight walking (s)'] = np.array([])
    for i in range(0,len(data['Right foot']['derived']['Stride length - straight walking (m)'])):
        lastHS = data['Right foot']['derived']['Stride length - all strides (m)'][:,1][ data['Right foot']['derived']['Stride length - all strides (m)'][:,1] < data['Right foot']['derived']['Stride length - straight walking (m)'][i,1] ]
        if lastHS.size > 0:
            lastHS=lastHS[-1]
        else:
            lastHS = data['Right foot']['derived']['Stride length - straight walking (m)'][i,0]
        sw = data['Right foot']['derived']['Stride length - straight walking (m)'][i,1] - lastHS
        data['Right foot']['derived']['Stride time per stride - straight walking (s)'] = np.append(data['Right foot']['derived']['Stride time per stride - straight walking (s)'], sw/data['Sample Frequency (Hz)']) 

    
    
    # Gait speed steady state gait
    # Method: Muir et al.
    indices= np.array([], dtype = int)
    for iy in data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0]:
        indices = np.append(indices, int(np.argmax(data['Left foot']['derived']['Gait speed per stride (m/s)'][:,0]==iy)))
    data['Left foot']['derived']['Gait speed per stride - no 2 steps around turn (m/s)'] = data['Left foot']['derived']['Gait speed per stride (m/s)'][indices,:]
    indices= np.array([], dtype = int)
    for iy in data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0]:
        indices = np.append(indices, int(np.argmax(data['Right foot']['derived']['Gait speed per stride (m/s)'][:,0]==iy)))
    data['Right foot']['derived']['Gait speed per stride - no 2 steps around turn (m/s)'] = data['Right foot']['derived']['Gait speed per stride (m/s)'][indices,:]
    
    # Method: No 1 stride around turn
    indices= np.array([], dtype = int)
    for iy in data['Left foot']['derived']['Stride length - no 1 steps around turn (m)'][:,0]:
        indices = np.append(indices, int(np.argmax(data['Left foot']['derived']['Gait speed per stride (m/s)'][:,0]==iy)))
    data['Left foot']['derived']['Gait speed per stride - no 1 steps around turn (m/s)'] = data['Left foot']['derived']['Gait speed per stride (m/s)'][indices,:]
    indices= np.array([], dtype = int)
    for iy in data['Right foot']['derived']['Stride length - no 1 steps around turn (m)'][:,0]:
        indices = np.append(indices, int(np.argmax(data['Right foot']['derived']['Gait speed per stride (m/s)'][:,0]==iy)))
    data['Right foot']['derived']['Gait speed per stride - no 1 steps around turn (m/s)'] = data['Right foot']['derived']['Gait speed per stride (m/s)'][indices,:]
        
    # Return average gait speed in steady state gait based on "Method: No 1 stride around turn"
    try:
        data['Spatiotemporals']['Gait speed (km/h)'] = round( (np.nanmedian(np.append(data['Left foot']['derived']['Gait speed per stride - no 2 steps around turn (m/s)'][:,2], data['Right foot']['derived']['Gait speed per stride - no 2 steps around turn (m/s)'][:,2]))  *3.6), 2)
    except RuntimeWarning:
        data['Spatiotemporals']['Gait speed (km/h)'] = round( (np.nanmedian(np.append(data['Left foot']['derived']['Gait speed per stride (m/s)'][:,2], data['Right foot']['derived']['Gait speed per stride (m/s)'][:,2])) *3.6), 2)
        errors['No steady state gait'] = True
        # data['Spatiotemporals']['Gait speed (km/h)'] = round( (np.nanmean(np.append(data['Left foot']['derived']['Gait speed per stride - no 1 steps around turn (m/s)'][:,2], data['Right foot']['derived']['Gait speed per stride - no 1 steps around turn (m/s)'][:,2])) *3.6), 2)
    return data


def asymmetry(leftvalue, rightvalue):
    # Input:  left value
    #         right value
    # Output: asymm: asymmetry value
    
    # Step length asymmetry: value l /(value l + value r)
    # As such, for each measure, a value of 0.50 reflects perfect symmetry, >0.5 reflects left larger, <0.5 reflects right larger
    
    asymm = leftvalue/(leftvalue+rightvalue)
    
    return asymm




def ststransfers(data, errors):
    from scipy import signal
    # Lean angle algorithm based on:
    # Pham MH, Warmerdam E, Elshehabi M, Schlenstedt C, Bergeest L-M, Heller M, Haertner L, Ferreira JJ, Berg D, Schmidt G, Hansen C and Maetzler W (2018) 
    # Validation of a Lower Back “Wearable”-Based Sit-to-Stand and Stand-to-Sit Algorithm for Patients With Parkinson's Disease and Older Adults in a Home-Like Environment.
    # Front. Neurol. 9:652. doi: 10.3389/fneur.2018.00652
    sample_frequency = data['Sample Frequency (Hz)']
    
    # if 'Sternum' not in data['Missing Sensors']:
    #     fGYRy = data['Sternum']['derived']['Gyroscope Earth frame'][:,1]
    #     # tilt = data['Sternum']['raw']['Orientation Euler'][:,1]
    #     tilt = np.zeros(shape=(len(data['Sternum']['derived']['Gyroscope Earth frame'][:,1]),1))
    #     for i in range(1,len(tilt)):
    #         tilt[i,0] = tilt[i-1,0] + (data['Sternum']['derived']['Gyroscope Earth frame'][i,1] * 1/sample_frequency)
    # elif 'Lumbar' not in data['Missing Sensors']:
    #     fGYRy = data['Lumbar']['derived']['Gyroscope Earth frame'][:,1]
    #     # tilt = data['Lumbar']['raw']['Orientation Euler'][:,1]
    #     tilt = np.zeros(shape=(len(data['Lumbar']['derived']['Gyroscope Earth frame'][:,1]),1))
    #     for i in range(1,len(tilt)):
    #         tilt[i,0] = tilt[i-1,0] + (data['Lumbar']['derived']['Gyroscope Earth frame'][i,1] * 1/sample_frequency)
    
    if 'Sternum' not in data['Missing Sensors']:
        fGYRy = data['Sternum']['raw']['Gyroscope'][:,1]
        # tilt = np.unwrap(np.deg2rad(data['Sternum']['raw']['Orientation Euler'][:,1]), period=2*np.pi)
        
        tilt2 = np.zeros(shape=(len(fGYRy),1))
        for i in range(1,len(tilt2)):
            tilt2[i,0] = tilt2[i-1,0] + (fGYRy[i] * 1/sample_frequency)
        
    elif 'Lumbar' not in data['Missing Sensors']:
        fGYRy = data['Lumbar']['raw']['Gyroscope'][:,1]
        # tilt = np.unwrap(np.deg2rad(data['Lumbar']['raw']['Orientation Euler'][:,1]), period=2*np.pi)
        
        tilt2 = np.zeros(shape=(len(fGYRy),1))
        for i in range(1,len(tilt2)):
            tilt2[i,0] = tilt2[i-1,0] + (fGYRy[i] * 1/sample_frequency)
    
    # Low pass filter fGYRy
    fc = 5  # Cut-off frequency of the filter
    w = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, w, filter_type)
    fGYRy = signal.filtfilt(b, a, fGYRy) # Apply filter on data
    
    # Drift removal
    totaldrift = tilt2[-1] - tilt2[0]
    tilt = np.zeros((len(tilt2),1))
    for i in range(0,len(tilt)):
        tilt[i] = tilt2[i] - (totaldrift/len(tilt2))*float(i)
    tilt = tilt.flatten()
        
    idxsts=np.array([], dtype = int) # Array of index numbers identified as STS transfer
    
    if np.mean(np.abs(tilt[signal.find_peaks(tilt, prominence= 0.2)[0]])) >= np.mean(np.abs(tilt[signal.find_peaks(-tilt, prominence= 0.2)[0]])):
        idxstspeaks = signal.find_peaks(tilt, prominence= 0.2)[0]
    elif np.mean(np.abs(tilt[signal.find_peaks(tilt, prominence= 0.2)[0]])) < np.mean(np.abs(tilt[signal.find_peaks(-tilt, prominence= 0.2)[0]])):
        idxstspeaks = signal.find_peaks(-tilt, prominence= 0.2)[0]
        fGYRy=-fGYRy
        
    # Find preceding and following 0 deg/s crossings with negative/positive slope 
    # This slope is dependent on the orientation of the sensor, TODO: see if this can be recognized in the data and automatically corrected for
    # crossing = 0
    for i in range(0,len(idxstspeaks)):
        zero_crossings = np.argwhere(np.diff(np.sign(fGYRy[0:idxstspeaks[i]])))+1 # Zero-crossing before STS peak
        # zero_crossings_neg_slope = zero_crossings[fGYRy[zero_crossings+1] < fGYRy[zero_crossings]] # Secure negative slope
        zero_crossings_neg_slope = zero_crossings[fGYRy[zero_crossings+1] > fGYRy[zero_crossings]] # Secure positive slope
        
        if np.shape(zero_crossings_neg_slope) == (0,):
            initialcrossing = np.array([0], dtype=int)
        else:
            initialcrossing = zero_crossings_neg_slope[-1]
        
        zero_crossings = idxstspeaks[i]+np.argwhere(np.diff(np.sign(fGYRy[idxstspeaks[i]:-1])))+1
        # zero_crossings_neg_slope = zero_crossings[fGYRy[zero_crossings+1] < fGYRy[zero_crossings]] # Secure negative slope
        zero_crossings_neg_slope = zero_crossings[fGYRy[zero_crossings+1] > fGYRy[zero_crossings]] # Secure positive slope
        
        if np.shape(zero_crossings_neg_slope) == (0,):
            finalcrossing = np.array([len(fGYRy)-1], dtype=int)
        else:
            finalcrossing = zero_crossings_neg_slope[0]
        
        
        if len(idxsts)<1:
            # If sts duration > 10 s or sts duration < 0.1 s; eliminate sts
            if finalcrossing - initialcrossing < 10 * sample_frequency or finalcrossing - initialcrossing > 0.1 * sample_frequency:
                idxsts = np.append(idxsts, np.arange(initialcrossing, finalcrossing+1))
        
        elif len(idxsts)>0:
            # If intra-sts duration <0.1 s combine with previous sts
            if initialcrossing < idxsts[-1] + 0.1 * sample_frequency:
                # if (fGYRy[idxstspeaks[i]] < 0 and fGYRy[idxstspeaks[i-1]] < 0) or (fGYRy[idxstspeaks[i]] > 0 and fGYRy[idxstspeaks[i-1]] > 0):
                idxsts = np.append(idxsts, np.arange(idxsts[-1], finalcrossing+1))
            else:
                # If sts duration > 10 s or sts duration < 0.1 s; eliminate sts
                if finalcrossing - initialcrossing < 10 * sample_frequency or finalcrossing - initialcrossing > 0.1 * sample_frequency:
                    idxsts = np.append(idxsts, np.arange(initialcrossing, finalcrossing+1))
            
            # If sts duration > 10 s or sts duration < 0.05 s; eliminate sts
            if finalcrossing - initialcrossing > 10 * sample_frequency:
                continue
    
    idxsts = np.sort(np.unique(idxsts))
    
    # Divide in different transfers
    ends = np.argwhere(np.diff(idxsts)>1)
    transfers = np.zeros((len(ends)+1,4)) # array with start[n,0] and stop[n,1] of each transfer
    if len(ends) > 0:
        for i in range(0,len(ends)):
            transfers[i,1] = idxsts[ends[i]] # end of transfer
            transfers[i+1,0] = idxsts[ends[i]+1] # start of transfer
        transfers[0,0] = int(idxsts[0])
        transfers[-1,1] = int(idxsts[-1])
        
        # Determine type of transfer; sit-to-stand or stand-to-sit
        lumbarvelocity = data['Lumbar']['derived']['Velocity (m/s)']
        lumbaracc = data['Lumbar']['raw']['Accelerometer Earth Frame']
        for i in range(0, np.shape(transfers)[0]):
            initiation = np.arange(0, int(transfers[i,0])) #, int(transfers[i,0]+0.2*sample_frequency))
            termination = np.arange(int(transfers[i,1]), len(lumbaracc)) #-0.2*sample_frequency), int(transfers[i,1]))
            
            # Assume vertical velocity variation of lumbar sensor smaller at the start of an sit-to-stand transfer compared to the end and smaller than 0.5 m/s
            if np.nanstd(lumbarvelocity[initiation,2]) < np.nanstd(lumbarvelocity[termination,2]):
                if np.nanstd(lumbaracc[initiation,2]) < 1:
                    transfers[i,2] = 1 #'sit to stand'
                else:
                    transfers[i,2] = np.nan
                
            # Assume vertical velocity variation of lumbar sensor larger at the start of an stand-to-sit transfer compared to the end and smaller than 0.5 m/s
            elif np.nanstd(lumbarvelocity[initiation,2]) > np.nanstd(lumbarvelocity[termination,2]):
                if np.nanstd(lumbaracc[termination,2]) < 1:
                    transfers[i,2] = 2 #'stand to sit'
                else:
                    transfers[i,2] = np.nan
            
            else:
                transfers[i,2] = np.nan
    else:
        transfers[:,2] = np.nan
        
    if np.all(transfers[~np.isnan(transfers[:,2])] == np.array([0, 0, 0, 0])):
        errors['Trial type'] = True
        print('Data was processed as if L-test, but no STS-transfers were detected!')
    
    if 'Sternum' not in data['Missing Sensors']:
        data['Sternum']['derived']['Transfers'] = transfers[~np.isnan(transfers[:,2])]
    elif 'Lumbar' not in data['Missing Sensors']:
        data['Lumbar']['derived']['Transfers'] = transfers[~np.isnan(transfers[:,2])]
       
    return data, errors





def leanangle(data, showfigure):
    # INPUT:  sensor data (STS transfers, accelerometer data earth frame, euler orientation)
    # OUTPUT: lean angle (deg) during sit to stand and stand to sit transfers
    import pyquaternion as pyq
    import math
    import sksurgerycore.algorithms.averagequaternions as aveq
    
    if 'Sternum' not in data['Missing Sensors']:
        transfers = data['Sternum']['derived']['Transfers']
    elif 'Lumbar' not in data['Missing Sensors']:
        transfers = data['Lumbar']['derived']['Transfers']
    
    sample_frequency = data['Sample Frequency (Hz)']
        
    # Calculate lean angle by calculation difference between two quaternion orientations
    for i in range(0, np.shape(transfers)[0]):
        
        if transfers[i,2] == 1 and transfers[i,0] > 0.2*sample_frequency: # Sit-to-stand
        
            phi   = np.zeros((int(transfers[i,1]-transfers[i,0]+0.2*sample_frequency),1))
            theta = np.zeros((int(transfers[i,1]-transfers[i,0]+0.2*sample_frequency),1))
            psi   = np.zeros((int(transfers[i,1]-transfers[i,0]+0.2*sample_frequency),1))
        
            q_mean = pyq.Quaternion( aveq.average_quaternions(data['Sternum']['raw']['Orientation Quaternion'][int(transfers[i,0]-0.2*sample_frequency):int(transfers[i,0]), :]) )
            
            for j in range(int(transfers[i,0]-0.2*sample_frequency), int(transfers[i,1])):
                q_new = pyq.Quaternion(data['Sternum']['raw']['Orientation Quaternion'][j,:])
                # Get the 3D difference between these two orientations
                qd = q_mean.conjugate * q_new
            
                # Calculate Euler angles from this difference quaternion
                phi[int(j-(transfers[i,0]-0.2*sample_frequency))]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) ) )
                theta[int(j-(transfers[i,0]-0.2*sample_frequency))] = np.rad2deg( math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) ) )
                psi[int(j-(transfers[i,0]-0.2*sample_frequency))]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) ) )
            
            minvalue = np.min(theta)
            maxvalue = np.max(theta)
            meanvalue = np.mean(theta[0:int(0.2*sample_frequency)])
            if np.abs( minvalue - meanvalue ) > np.abs( maxvalue - meanvalue ):
                transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            elif np.abs( maxvalue - meanvalue ) > np.abs( minvalue - meanvalue ):
                transfers[i,3] = round( np.abs( maxvalue - meanvalue ), 1)
        
        elif transfers[i,2] == 1 and transfers[i,0] <= 0.2*sample_frequency: # Sit-to-stand
        
            phi   = np.zeros((int(transfers[i,1]-transfers[i,0]),1))
            theta = np.zeros((int(transfers[i,1]-transfers[i,0]),1))
            psi   = np.zeros((int(transfers[i,1]-transfers[i,0]),1))
        
            q_mean = pyq.Quaternion( aveq.average_quaternions(data['Sternum']['raw']['Orientation Quaternion'][int(transfers[i,0]):int(transfers[i,0]+0.2*sample_frequency), :]) )
            # q_mean = pyq.Quaternion(np.mean(data['Sternum']['raw']['Orientation Quaternion'][int(transfers[i,0]):int(transfers[i,0]+0.2*sample_frequency), :], axis=0))
            for j in range(int(transfers[i,0]), int(transfers[i,1])):
                q_new = pyq.Quaternion(data['Sternum']['raw']['Orientation Quaternion'][j,:])
                # Get the 3D difference between these two orientations
                qd = q_mean.conjugate * q_new
            
                # Calculate Euler angles from this difference quaternion
                phi[int(j-transfers[i,0])]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) ) )
                theta[int(j-transfers[i,0])] = np.rad2deg( math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) ) )
                psi[int(j-transfers[i,0])]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) ) )
            
            # minvalue = np.min(theta)
            # meanvalue = np.mean(theta[0:int(0.2*sample_frequency)])
            # transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            minvalue = np.min(theta)
            maxvalue = np.max(theta)
            meanvalue = np.mean(theta[0:int(0.2*sample_frequency)])
            if np.abs( minvalue - meanvalue ) > np.abs( maxvalue - meanvalue ):
                transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            elif np.abs( maxvalue - meanvalue ) > np.abs( minvalue - meanvalue ):
                transfers[i,3] = round( np.abs( maxvalue - meanvalue ), 1)
                
        elif transfers[i,2] == 2 and transfers[i,1] < (len(data['Sternum']['raw']['Orientation Quaternion'])-0.2*sample_frequency): # Stand-to-sit
            
            phi   = np.zeros((int(transfers[i,1]+0.2*sample_frequency-transfers[i,0]),1))
            theta = np.zeros((int(transfers[i,1]+0.2*sample_frequency-transfers[i,0]),1))
            psi   = np.zeros((int(transfers[i,1]+0.2*sample_frequency-transfers[i,0]),1))
            
            q_mean = pyq.Quaternion( aveq.average_quaternions(data['Sternum']['raw']['Orientation Quaternion'][int(transfers[i,1]):int(transfers[i,1]+0.2*sample_frequency), :]) )
            
            for j in range(int(transfers[i,0]), int(transfers[i,1]+0.2*sample_frequency)):
                q_new = pyq.Quaternion(data['Sternum']['raw']['Orientation Quaternion'][j,:])
                # Get the 3D difference between these two orientations
                qd = q_mean.conjugate * q_new
            
                # Calculate Euler angles from this difference quaternion
                phi[int(j-transfers[i,0])]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) ) )
                theta[int(j-transfers[i,0])] = np.rad2deg( math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) ) )
                psi[int(j-transfers[i,0])]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) ) )
            
            # minvalue = np.min(theta)
            # meanvalue = np.mean(theta[int(-0.2*sample_frequency):])
            # transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            minvalue = np.min(theta)
            maxvalue = np.max(theta)
            meanvalue = np.mean(theta[int(-0.2*sample_frequency):])
            if np.abs( minvalue - meanvalue ) > np.abs( maxvalue - meanvalue ):
                transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            elif np.abs( maxvalue - meanvalue ) > np.abs( minvalue - meanvalue ):
                transfers[i,3] = round( np.abs( maxvalue - meanvalue ), 1)
                    
        elif transfers[i,2] == 2 and transfers[i,1] >= 0.2*sample_frequency: # Stand-to-sit
            
            phi   = np.zeros((int(transfers[i,1]-transfers[i,0]),1))
            theta = np.zeros((int(transfers[i,1]-transfers[i,0]),1))
            psi   = np.zeros((int(transfers[i,1]-transfers[i,0]),1))
            
            q_mean = pyq.Quaternion( aveq.average_quaternions(data['Sternum']['raw']['Orientation Quaternion'][int(transfers[i,1]-0.2*sample_frequency):int(transfers[i,1]), :]) )
            
            for j in range(int(transfers[i,0]), int(transfers[i,1])):
                q_new = pyq.Quaternion(data['Sternum']['raw']['Orientation Quaternion'][j,:])
                # Get the 3D difference between these two orientations
                qd = q_mean.conjugate * q_new
            
                # Calculate Euler angles from this difference quaternion
                phi[int(j-transfers[i,0])]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) ) )
                theta[int(j-transfers[i,0])] = np.rad2deg( math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) ) )
                psi[int(j-transfers[i,0])]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) ) )
            
            # minvalue = np.min(theta)
            # meanvalue = np.mean(theta[int(-0.2*sample_frequency):])
            # transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            minvalue = np.min(theta)
            maxvalue = np.max(theta)
            meanvalue = np.mean(theta[int(-0.2*sample_frequency):])
            if np.abs( minvalue - meanvalue ) > np.abs( maxvalue - meanvalue ):
                transfers[i,3] = round( np.abs( minvalue - meanvalue ), 1)
            elif np.abs( maxvalue - meanvalue ) > np.abs( minvalue - meanvalue ):
                transfers[i,3] = round( np.abs( maxvalue - meanvalue ), 1)
            
    
    # Determine mean of maximum lean angle per type of tranfer
    # Return lean angle per transfer to data dict
    if (transfers[transfers[:,2] == 1]).size > 0:
        leananglesittostand = np.mean( transfers[transfers[:,2] == 1, 3] )
        data['Spatiotemporals']['Lean angle sit-to-stand (deg)'] = np.round(leananglesittostand, 1)
    else:
        data['Spatiotemporals']['Lean angle sit-to-stand (deg)'] = np.nan
    if (transfers[transfers[:,2] == 2]).size > 0:
        leananglestandtosit = np.mean( transfers[transfers[:,2] == 2, 3] )
        data['Spatiotemporals']['Lean angle stand-to-sit (deg)'] = np.round(leananglestandtosit, 1)
    else:
        data['Spatiotemporals']['Lean angle stand-to-sit (deg)'] = np.nan
        
    return data



def trunk_ROM(data):
    from scipy import signal
    
    sample_frequency = data['Sample Frequency (Hz)']
    
    if 'Sternum' not in data['Missing Sensors']:
        GYRz = data['Sternum']['raw']['Gyroscope'][:,2]
        eulerX = data['Sternum']['raw']['Orientation Euler'][:,0]
        eulerZ = data['Sternum']['raw']['Orientation Euler'][:,2]
        
    elif 'Lumbar' not in data['Missing Sensors']:
        GYRz = data['Lumbar']['raw']['Gyroscope'][:,2]
        eulerX = data['Lumbar']['raw']['Orientation Euler'][:,0]
        eulerZ = data['Lumbar']['raw']['Orientation Euler'][:,2]
    
    if 'Lumbar' not in data['Missing Sensors']:
        idx_turn = data['Lumbar']['derived']['Change in Walking Direction samples']
    elif 'Sternum' not in data['Missing Sensors']:
        idx_turn = data['Sternum']['derived']['Change in Walking Direction samples']
        
    idx_straight = np.arange(0,len(GYRz))
    idx_straight = idx_straight[~np.isin(idx_straight, idx_turn)]
    
    if len(idx_turn) > 0:
        diffturnidx = np.diff(idx_turn)
        newturnstart = np.array([idx_turn[0]], dtype = int)
        newendturn = np.array([idx_turn[-1]], dtype = int)
        for i in range(0,len(diffturnidx)):
            if diffturnidx[i] > 1:
                newturnstart = np.append(newturnstart, idx_turn[i+1])
                newendturn = np.append(newendturn, idx_turn[i])
        newendturn = np.sort(newendturn)
    else:
        newturnstart = np.array([], dtype = int)
        newendturn = np.array([], dtype = int)
    
    idx_steadystate = idx_straight
    TC = np.sort(np.append(data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0], data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0]))
    IC = np.sort(np.append(data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,1], data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,1]))
    for i in range(0, len(newturnstart)):
        try:
            initiationphase = np.arange(newendturn[i], TC[TC>newendturn[i]][0])
        except IndexError:
            initiationphase = np.array([])
        idx_steadystate = idx_steadystate[~np.isin(idx_steadystate, initiationphase)]
    for i in range(0, len(newturnstart)):
        try:
            terminationphase = np.arange(IC[IC<newturnstart[i]][-1], newturnstart[i])
        except IndexError:
            terminationphase = np.array([])
        idx_steadystate = idx_steadystate[~np.isin(idx_steadystate, terminationphase)]
    
    try:
        initialinitiation = np.arange(0, TC[0])
    except IndexError:
        initialinitiation = np.array([])
    try:
        lasttermination = np.arange(IC[-1], len(GYRz))
    except IndexError:
        lasttermination = np.array([])
    idx_steadystate = idx_steadystate[~np.isin(idx_steadystate, initialinitiation)]
    idx_steadystate = idx_steadystate[~np.isin(idx_steadystate, lasttermination)]
    
    # Correct gimbal lock
    idxdifs = np.argwhere(np.diff(eulerX)>300)
    idxdifs = np.sort(np.append(idxdifs, np.argwhere(np.diff(eulerX)<-300)))
    for i in np.arange(start=0, stop=len(idxdifs)-1, step=1):
        # if np.min(euler[idxdifs[i]:idxdifs[i+1]]) > 50:
        eulerX[idxdifs[i]:idxdifs[i+1]] = -1*(eulerX[idxdifs[i]:idxdifs[i+1]]) - np.diff([eulerX[idxdifs[i]], -1*eulerX[idxdifs[i]]])
    idxdifs = np.argwhere(np.diff(eulerZ)>300)
    idxdifs = np.sort(np.append(idxdifs, np.argwhere(np.diff(eulerZ)<-300)))
    for i in np.arange(start=0, stop=len(idxdifs)-1, step=1):
        # if np.min(euler[idxdifs[i]:idxdifs[i+1]]) > 50:
        eulerZ[idxdifs[i]:idxdifs[i+1]] = -1*(eulerZ[idxdifs[i]:idxdifs[i+1]]) - np.diff([eulerZ[idxdifs[i]], -1*eulerZ[idxdifs[i]]])
    
    # # Low pass filter euler angles
    fc = 5  # Cut-off frequency of the filter
    w = fc / (sample_frequency / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, w, filter_type)
    eulerX = signal.filtfilt(b, a, eulerX) # Apply filter on data
    eulerZ = signal.filtfilt(b, a, eulerZ) # Apply filter on data
        
    rom_per_strideX = np.zeros((len(IC),1))
    rom_per_strideZ = np.zeros((len(IC),1))
    for i in range(0,len(IC)-1):
        idxstride = np.arange(IC[i], IC[i+1])
        if np.all(np.isin(idxstride, idx_steadystate)) and ((IC[i+1] - IC[i]) < 150*data['Spatiotemporals']['Stride time left (s)']):
            rom_per_strideX[i] = np.max(eulerX[idxstride.astype(int)]) - np.min(eulerX[idxstride.astype(int)])
            rom_per_strideZ[i] = np.max(eulerZ[idxstride.astype(int)]) - np.min(eulerZ[idxstride.astype(int)])
    rom_per_strideX = rom_per_strideX[rom_per_strideX>0] # Rejection in case of gimbal lock
    rom_per_strideX = rom_per_strideX[rom_per_strideX<180] # Rejection in case of gimbal lock
    trunkROMX = np.nanmean(rom_per_strideX)            
    rom_per_strideZ = rom_per_strideZ[rom_per_strideZ>0] # Rejection in case of gimbal lock
    rom_per_strideZ = rom_per_strideZ[rom_per_strideZ<180] # Rejection in case of gimbal lock
    trunkROMZ = np.nanmean(rom_per_strideZ)
    # trunkROM = np.max(euler[idx_steadystate]) - np.min(euler[idx_steadystate])
    
    data['Lumbar']['derived']['Steady-state walking samples'] = idx_steadystate.astype(int)
    data['Sternum']['derived']['RoM per steady-state stride X'] = rom_per_strideX
    data['Spatiotemporals']['Trunk transverse range of motion (deg)'] = np.round(trunkROMX, 1)
    data['Sternum']['derived']['RoM per steady-state stride Z'] = rom_per_strideZ
    data['Spatiotemporals']['Trunk coronal range of motion (deg)'] = np.round(trunkROMZ, 1)
    
    
    return data



def relative_orientation(sensorRelative, sensorFixed, relative_to):
    import pyquaternion as pyq
    import math
    # sensorRelative should include the orientation of the sensor relative to the orientation of another sensor over time.
    # sensorFixed should include the orientation of the sensor to which another sensor's orientation should be calculated over time.
    
    quat_sensorRelative = sensorRelative['raw']['Orientation Quaternion']
    quat_sensorFixed = sensorFixed['raw']['Orientation Quaternion']
    
    phi = np.zeros((len(quat_sensorRelative), 1))
    theta = np.zeros((len(quat_sensorRelative), 1))
    psi = np.zeros((len(quat_sensorRelative), 1))
    
    for j in range(len(quat_sensorFixed)):
        q_fixed = pyq.Quaternion( quat_sensorFixed[j,:] )
        q_relative = pyq.Quaternion( quat_sensorRelative[j,:] )
        
        # Get the 3D difference between these two orientations
        qd = q_relative * q_fixed.conjugate
        qd = qd.normalised
    
        # Calculate Euler angles from this difference quaternion
        phi[j]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) ) )
        theta[j] = np.rad2deg( math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) ) )
        psi[j]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) ) )
    
    sensorRelative['derived']['Orientation relative to '+relative_to] = np.array([phi.flatten(), theta.flatten(), psi.flatten()]).T
        
    return sensorRelative



def footAngle(data, errors):
    # Adjusted by Carmen Ensink, 28/07/22, Sint Maartenskliniek
    # Added by Jean Ormiston, 18/07/22 Sint Maartenskliniek
    # HS dorsiflexion positive angle, TO plantarflexion negative
        
    sides = ['Left foot', 'Right foot']
    if errors['No walking period'] == False:
        for side in sides:
            idxTC = data[side]['Gait Events']['Terminal Contact']
            idxIC = data[side]['Gait Events']['Initial Contact']
            idxMSt = data[side]['Gait Phases']['Mid-Stance']
            
            data[side]['derived']['Foot angle at IC'] = np.array([])
            data[side]['derived']['Foot angle at TC'] = np.array([])
            
            for i in range(0,len(idxIC)):
                # find first instance of mid-stance after IC
                try:
                    idxMSt_first = np.argwhere(idxMSt>idxIC[i])[0]
                except IndexError:
                    idxMSt_first = np.argwhere(idxMSt<idxIC[i])[-1]
                # correct foot angle at IC for angle at following mid-stance, unless no mid-stance is following, than take last instance of mid-stance
                data[side]['derived']['Foot angle at IC'] = np.append(data[side]['derived']['Foot angle at IC'], round((data[side]['raw']['Orientation Euler'][idxIC[i], 1] - data[side]['raw']['Orientation Euler'][idxMSt[idxMSt_first], 1])[0], 1))
            
            for i in range(0,len(idxTC)):
                try:
                    idxMSt_last = np.argwhere(idxMSt<idxTC[i])[-1]
                except IndexError:
                    idxMSt_last = np.argwhere(idxMSt>idxTC[i])[0]
                # correct foot angle at TC for angle at previous mid-stance, unless no mid-stance is following, than take following instance of mid-stance
                data[side]['derived']['Foot angle at TC'] = np.append(data[side]['derived']['Foot angle at TC'], round((data[side]['raw']['Orientation Euler'][idxTC[i], 1] - data[side]['raw']['Orientation Euler'][idxMSt[idxMSt_last], 1])[0], 1))
    else:
        data['Left foot']['derived']['Foot angle at IC'] = np.transpose(np.array([[np.nan],[np.nan]]))
        data['Right foot']['derived']['Foot angle at IC'] = np.transpose(np.array([[np.nan],[np.nan]]))
        data['Left foot']['derived']['Foot angle at TC'] = np.transpose(np.array([[np.nan],[np.nan]]))
        data['Right foot']['derived']['Foot angle at TC'] = np.transpose(np.array([[np.nan],[np.nan]]))
    
    data['Spatiotemporals']['Foot angle at IC left'] = round(np.mean(data['Left foot']['derived']['Foot angle at IC']), 1)
    data['Spatiotemporals']['Foot angle at IC right'] = round(np.mean(data['Right foot']['derived']['Foot angle at IC']), 1)
    data['Spatiotemporals']['Foot angle at TC left'] = round(np.mean(data['Left foot']['derived']['Foot angle at TC']), 1)
    data['Spatiotemporals']['Foot angle at TC right'] = round(np.mean(data['Right foot']['derived']['Foot angle at TC']), 1)
    
    return data