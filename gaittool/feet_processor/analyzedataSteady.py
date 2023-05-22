# -*- coding: utf-8 -*-
"""
Last update: May 2023 by Jean Ormiston
Author: Carmen Ensink

"""


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from ..helpers.gaitphasefunctions import InitialSwing, PreSwing, MidSwing, TerminalSwing, LoadingResponse, TerminalStance, MidStance
from ..helpers.gaiteventfunctions import TSwOnset

def consecutive_numbers(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    b = np.abs(np.diff(a))
    iszero = np.concatenate(([0], np.equal(b, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def analyzedataSteady(data, errors, showfigure,removeSteps):

    try:

        gyroscopedataLeft = data['Left foot']['raw']['Gyroscope']
        gyroscopedataRight = data['Right foot']['raw']['Gyroscope']
        accelerometerdataLeft = data['Left foot']['raw']['Accelerometer Earth Frame']
        accelerometerdataRight = data['Right foot']['raw']['Accelerometer Earth Frame']
        sample_frequency = data['Sample Frequency (Hz)']

        # Set second-order low-pass butterworth filter;
        # Cut-off frequency: 17Hz accelrometer signals, cutoff frequency: 15 Hz gyroscope signals
        # Values based on:
        # Sabatini, A. M., Martelloni, C., Scapellato, S., & Cavallo, F. (2005). Assessment of walking features from foot inertial sensing. IEEE Transactions on biomedical engineering, 52(3), 486-494.
        fcgyr = 15  # Cut-off frequency of the filter for gyroscope data
        fcacc = 17  # Cut-off frequency of the filter for accelerometer data
        wgyr = fcgyr / (sample_frequency / 2) # Normalize the frequency
        wacc = fcacc / (sample_frequency / 2) # Normalize the frequency
        N = 2 # Order of the butterworth filter
        filter_type = 'lowpass' # Type of the filter
        bgyr, agyr = signal.butter(N, wgyr, filter_type)
        bacc, aacc = signal.butter(N, wacc, filter_type)

        # Apply filter on gyroscope data (medio-lateral direction)
        fGYRyL = signal.filtfilt(bgyr, agyr, gyroscopedataLeft[:,1])
        fGYRyR = signal.filtfilt(bgyr, agyr, gyroscopedataRight[:,1])

        # Apply filter on accelerometer signal
        fACCzL = signal.filtfilt(bacc, aacc, accelerometerdataLeft[:,2])
        fACCzR = signal.filtfilt(bacc, aacc, accelerometerdataRight[:,2])



        # Mid-Swing - based on:
        # Sabatini, A. M., Martelloni, C., Scapellato, S., & Cavallo, F. (2005). Assessment of walking features from foot inertial sensing. IEEE Transactions on biomedical engineering, 52(3), 486-494.
        # Angular velocity reaches maximum value in counter clockwise direction
        idxOnsMSwleft = (signal.find_peaks(fGYRyL, distance = 0.7*sample_frequency, prominence=1, height=0.3))[0]
        idxOnsMSwright = (signal.find_peaks(fGYRyR, distance = 0.7*sample_frequency, prominence=1, height=0.3))[0]


        # Heel-Strike - based on:
        # Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-time detection of seven phases of gait in children with cerebral palsy using two gyroscopes. Sensors, 19(11), 2517.
        # Zero-crossing after mid swing in the gait cycle.
        # Left
        idxHSleft = np.asarray([], dtype=int)
        for i in range(0, len(idxOnsMSwleft)):
            try:
                firstnegative = idxOnsMSwleft[i] + np.argwhere(fGYRyL[idxOnsMSwleft[i]:] < 0)[0]
            except IndexError:
                firstnegative = idxOnsMSwleft[i] + np.argmin(fGYRyL[idxOnsMSwleft[i]:])
            lastpositive = firstnegative-1
            if np.abs(fGYRyL[lastpositive]) < np.abs(fGYRyL[firstnegative]):
                lastzerocrossing = lastpositive
            else:
                lastzerocrossing = firstnegative
            idxHSleft = np.append(idxHSleft, lastzerocrossing)
        # Right
        idxHSright = np.asarray([], dtype=int)
        for i in range(0, len(idxOnsMSwright)):
            try:
                firstnegative = idxOnsMSwright[i] + np.argwhere(fGYRyR[idxOnsMSwright[i]:] < 0)[0]
            except IndexError:
                firstnegative = idxOnsMSwright[i] + np.argmin(fGYRyR[idxOnsMSwright[i]:])
            lastpositive = firstnegative-1
            if np.abs(fGYRyR[lastpositive]) < np.abs(fGYRyR[firstnegative]):
                lastzerocrossing = lastpositive
            else:
                lastzerocrossing = firstnegative
            idxHSright = np.append(idxHSright, lastzerocrossing)



        # Toe off - based on:
        # Mo, S., & Chow, D. H. (2018). Accuracy of three methods in gait event detection during overground running. Gait & posture, 59, 93-98.
        # Original paper: Mercer, J. A., Bates, B. T., Dufek, J. S., & Hreljac, A. (2003). Characteristics of shock attenuation during fatigued running. Journal of Sports Science, 21(11), 911-919.
        # From midswing to midswing; find positive peak.
        # Left
        idxTOleft=np.asarray([], dtype=int)
        for i in range(1,len(idxOnsMSwleft)):
            startwindow = np.asarray(idxOnsMSwleft[i-1] + ((idxOnsMSwleft[i]-idxOnsMSwleft[i-1])/2.5), dtype=int)
            window = range(startwindow, idxOnsMSwleft[i]-int(0.05*sample_frequency))
            idxT2temp = signal.find_peaks((fACCzL)[window])[0] #accelerometerdataLeft[:,2], height=0.3*np.max(fACCzL)
            if len(idxT2temp) > 0:
                mingyr = np.argmin(fGYRyL[startwindow + idxT2temp])
                idxT2temp2 = idxT2temp[mingyr]
                idxTOleft = np.append(idxTOleft, (startwindow + idxT2temp2))
        # Right
        idxTOright=np.asarray([], dtype=int)
        for i in range(1,len(idxOnsMSwright)):
            startwindow = np.asarray(idxOnsMSwright[i-1] + ((idxOnsMSwright[i]-idxOnsMSwright[i-1])/2.5), dtype=int)
            window = range(startwindow, idxOnsMSwright[i]-int(0.05*sample_frequency))
            idxT2temp = signal.find_peaks((fACCzR)[window])[0] #accelerometerdataRight[:,2], height=0.3*np.max(fACCzR)
            if len(idxT2temp) > 0:
                mingyr = np.argmin(fGYRyR[startwindow + idxT2temp])
                idxT2temp2 = idxT2temp[mingyr]
                idxTOright = np.append(idxTOright, (startwindow + idxT2temp2))

        # Remove gait events that occur during turns, choose to remove 1 or 2 steps before turn:
        if 'Lumbar' not in data['Missing Sensors']:
            try:
                idxturn = data['Lumbar']['derived']['Change in Walking Direction samples']
            except:
                Warning('No turn in trial')
        elif 'Sternum' not in data['Missing Sensors']:
            try:
                idxturn = data['Sternum']['derived']['Change in Walking Direction samples']
            except:
                Warning('No turn in trial')

        if idxturn:
            events = [idxTOleft, idxTOright, idxHSright, idxHSleft,idxOnsMSwright,idxOnsMSwleft]
            for ev in events:  # loop over events
                for e in ev:
                    if ev[e] in idxturn:
                        ev.remove(ev[e])

            # Options: remove 1 or 2 strides before and after turn
            remove = np.array([])
            if removeSteps==1 or removeSteps==2:
                #find number of turns in idxturns
                turns = consecutive_numbers(idxturn)
                nturns = len(turns)
                #Loop over turns
                for turn in range(0,nturns):
                    lastHSleft = idxHSleft[idxHSleft < turns[turn,0]]
                    if lastHSleft.size > 0:
                        lastHSleft = lastHSleft[-1]
                    idxHSleft.remove(lastHSleft)
                    lastMSwleft = idxOnsMSwleft[idxOnsMSwleft < turns[turn, 0]]
                    if lastMSwleft.size > 0:
                        lastMSwleft = lastMSwleft[-1]
                    idxOnsMSwleft.remove(lastMSwleft)
                    remove = np.append(remove, lastHSleft)

                    lastHSright = idxHSright[
                        idxHSright < turns[turn,0]]
                    if lastHSright.size > 0:
                        lastHSright = lastHSright[-1]
                    idxHSright.remove(lastHSright)
                    lastMSwright = idxOnsMSwright[idxOnsMSwright < turns[turn, 0]]
                    if lastMSwright.size > 0:
                        lastMSwright = lastMSwright[-1]
                    idxOnsMSwright.remove(lastMSwright)
                    remove = np.append(remove, lastHSright)

                    if removeSteps == 2:
                        firstHSleft = idxHSleft[
                            idxHSleft > turns[turn,1]]
                        if firstHSleft.size > 0:
                            firstHSleft = firstHSleft[0]
                        idxHSleft.remove(firstHSleft)
                        firstMSwleft = idxOnsMSwleft[idxOnsMSwleft > turns[turn, 1]]
                        if firstMSwleft.size > 0:
                            firstMSwleft = firstMSwleft[-1]
                        idxOnsMSwleft.remove(firstMSwleft)
                        remove = np.append(remove, firstHSleft)

                        firstHSright = idxHSright[
                            idxHSright > turns[turn,1]]
                        if firstHSright.size > 0:
                            firstHSright = firstHSright[0]
                        idxHSright.remove(firstHSright)
                        firstMSwright = idxOnsMSwright[idxOnsMSwright > turns[turn, 1]]
                        if firstMSwright.size > 0:
                            firstMSwright = firstMSwright[-1]
                        idxOnsMSwright.remove(firstMSwright)
                        remove = np.append(remove, firstHSright)
                remove = np.sort(np.unique(remove))

        # Define the other gait events / gait phases in gait cylce.
        idxOnsTSwleft = TSwOnset(fGYRyL, idxTOleft)
        idxOnsMStleft = idxTOright  # Based on: Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-Time Detection of Seven Phases of Gait in Children with Cerebral Palsy Using Two Gyroscopes. Sensors, 19(11), 2517.
        idxHOleft = idxOnsMSwright  # Based on: Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-Time Detection of Seven Phases of Gait in Children with Cerebral Palsy Using Two Gyroscopes. Sensors, 19(11), 2517.
        idxOnsPSwleft = idxHSright  # Based on: Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-Time Detection of Seven Phases of Gait in Children with Cerebral Palsy Using Two Gyroscopes. Sensors, 19(11), 2517.
        idxMStleft = MidStance(gyroscopedataLeft, idxOnsMStleft, idxHOleft)
        idxISwleft = InitialSwing(fGYRyL, idxTOleft, idxOnsMSwleft)
        idxMSwleft = MidSwing(fGYRyL, idxOnsMSwleft, idxOnsTSwleft)
        idxTSwleft = TerminalSwing(fGYRyL, idxOnsTSwleft, idxHSleft)
        idxLRleft = LoadingResponse(fGYRyL, idxHSleft, idxMStleft)
        idxTStleft = TerminalStance(fGYRyL, idxHOleft, idxOnsPSwleft)
        idxPSwleft = PreSwing(fGYRyL, idxOnsPSwleft, idxTOleft)

        idxOnsTSwright = TSwOnset(fGYRyR, idxTOright)
        idxOnsMStright = idxTOleft  # Based on: Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-Time Detection of Seven Phases of Gait in Children with Cerebral Palsy Using Two Gyroscopes. Sensors, 19(11), 2517.
        idxHOright = idxOnsMSwleft  # Based on: Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-Time Detection of Seven Phases of Gait in Children with Cerebral Palsy Using Two Gyroscopes. Sensors, 19(11), 2517.
        idxOnsPSwright = idxHSleft  # Based on: Behboodi, A., Zahradka, N., Wright, H., Alesi, J., & Lee, S. (2019). Real-Time Detection of Seven Phases of Gait in Children with Cerebral Palsy Using Two Gyroscopes. Sensors, 19(11), 2517.
        idxMStright = MidStance(gyroscopedataRight, idxOnsMStright, idxHOright)
        idxISwright = InitialSwing(fGYRyR, idxTOright, idxOnsMSwright)
        idxMSwright = MidSwing(fGYRyR, idxOnsMSwright, idxOnsTSwright)
        idxTSwright = TerminalSwing(fGYRyR, idxOnsTSwright, idxHSright)
        idxLRright = LoadingResponse(fGYRyR, idxHSright, idxMStright)
        idxTStright = TerminalStance(fGYRyR, idxHOright, idxOnsPSwright)
        idxPSwright = PreSwing(fGYRyR, idxOnsPSwright, idxTOright)

        # Visual check - event detection
        if showfigure == "view":
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_title('Left foot')
            ax1.plot(fGYRyL, 'k', label='gyroscopedata y')
            ax1.plot(idxTOleft, fGYRyL[idxTOleft], 'k^', markersize=4, label='Toe-Off')
            ax1.plot(idxOnsMSwleft, fGYRyL[idxOnsMSwleft], 'kx', markersize=4, label='Mid-Swing onset')  # g
            ax1.plot(idxOnsTSwleft, fGYRyL[idxOnsTSwleft], 'kd', markersize=4, label='Terminal-Swing onset')  # c
            ax1.plot(idxHSleft, fGYRyL[idxHSleft], 'kv', markersize=4, label='Heel-Strike')
            ax1.plot(idxOnsMStleft, fGYRyL[idxOnsMStleft], 'ms', markersize=4, label='Mid-Stance onset')  # m
            ax1.plot(idxHOleft, fGYRyL[idxHOleft], 'bH', markersize=4, label='Terminal-Stance onset')  # r
            ax1.plot(idxOnsPSwleft, fGYRyL[idxOnsPSwleft], 'ko', markersize=4, label='Pre-Swing onset')

            # ax1.set_xlabel('Time (samples)')
            ax1.set_ylabel('Angular velocity (rad/s)')
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax2.set_title('Right foot')
            ax2.plot(fGYRyR, 'k')  # , label='gyroscopedata y')
            ax2.plot(idxTOright, fGYRyR[idxTOright], 'k^', markersize=4)  # , label='TO')
            ax2.plot(idxOnsMSwright, fGYRyR[idxOnsMSwright], 'kx', markersize=4)  # , label='MSw onset') #g
            ax2.plot(idxOnsTSwright, fGYRyR[idxOnsTSwright], 'kd', markersize=4)  # , label='TSw onset') #c
            ax2.plot(idxHSright, fGYRyR[idxHSright], 'kv', markersize=4)  # , label='HS')
            ax2.plot(idxOnsMStright, fGYRyR[idxOnsMStright], 'ms', markersize=4)  # , label='MSt onset') #m
            ax2.plot(idxHOright, fGYRyR[idxHOright], 'bH', markersize=4)  # , label='HO') #r
            ax2.plot(idxOnsPSwright, fGYRyR[idxOnsPSwright], 'ko', markersize=4)  # , label='PSw onset')

            ax2.set_xlabel('Time (samples)')
            ax2.set_ylabel('Angular velocity (rad/s)')
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        # Visual check - phase detection
        if showfigure == 'view':
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_title('Left foot')
            ax1.plot(fGYRyL, 'k', label='gyroscopedata y')
            ax1.plot(idxISwleft, fGYRyL[idxISwleft], 'gs', markersize=1, label='Initial-Swing')
            ax1.plot(idxMSwleft, fGYRyL[idxMSwleft], 'cs', markersize=1, label='Mid-Swing')
            ax1.plot(idxTSwleft, fGYRyL[idxTSwleft], 'ks', markersize=1, label='Terminal-Swing')
            ax1.plot(idxLRleft, fGYRyL[idxLRleft], 'mx', markersize=1, label='Loading Response')
            ax1.plot(idxMStleft, fGYRyL[idxMStleft], 'rx', markersize=1, label='Mid-Stance')
            ax1.plot(idxTStleft, fGYRyL[idxTStleft], 'bx', markersize=1, label='Terminal-Stance')
            ax1.plot(idxPSwleft, fGYRyL[idxPSwleft], 'kx', markersize=1, label='Pre-Swing')

            ax1.set_xlabel('Time (samples)')
            ax1.set_ylabel('Angular velocity (rad/s)')
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax2.set_title('Right foot')
            ax2.plot(fGYRyR, 'k', label='gyroscopedata y')
            ax2.plot(idxISwright, fGYRyR[idxISwright], 'gs', markersize=1)
            ax2.plot(idxMSwright, fGYRyR[idxMSwright], 'cs', markersize=1)
            ax2.plot(idxTSwright, fGYRyR[idxTSwright], 'ks', markersize=1)
            ax2.plot(idxLRright, fGYRyR[idxLRright], 'mx', markersize=1)
            ax2.plot(idxMStright, fGYRyR[idxMStright], 'rx', markersize=1)
            ax2.plot(idxTStright, fGYRyR[idxTStright], 'bx', markersize=1)
            ax2.plot(idxPSwright, fGYRyR[idxPSwright], 'kx', markersize=1)

            ax2.set_xlabel('Time (samples)')
            ax2.set_ylabel('Angular velocity (rad/s)')

        # Place gait events and gait phases in structured dictionary.
        data['Left foot']['Gait Events'] = dict()
        data['Right foot']['Gait Events'] = dict()
        data['Left foot']['Gait Phases'] = dict()
        data['Right foot']['Gait Phases'] = dict()
        data['Left foot']['Gait Events']['Terminal Contact'] = idxTOleft
        data['Right foot']['Gait Events']['Terminal Contact'] = idxTOright
        data['Left foot']['Gait Events']['Mid-Swing Onset'] = idxOnsMSwleft
        data['Right foot']['Gait Events']['Mid-Swing Onset'] = idxOnsMSwright
        data['Left foot']['Gait Events']['Terminal-Swing Onset'] = idxOnsTSwleft
        data['Right foot']['Gait Events']['Terminal-Swing Onset'] = idxOnsTSwright
        data['Left foot']['Gait Events']['Initial Contact'] = idxHSleft
        data['Right foot']['Gait Events']['Initial Contact'] = idxHSright
        data['Left foot']['Gait Events']['Mid-Stance Onset'] = idxOnsMStleft
        data['Right foot']['Gait Events']['Mid-Stance Onset'] = idxOnsMStright
        data['Left foot']['Gait Events']['Heel Off'] = idxHOleft
        data['Right foot']['Gait Events']['Heel Off'] = idxHOright
        data['Left foot']['Gait Events']['Pre-Swing Onset'] = idxOnsPSwleft
        data['Right foot']['Gait Events']['Pre-Swing Onset'] = idxOnsPSwright
        data['Left foot']['Gait Phases']['Initial-Swing'] = idxISwleft
        data['Right foot']['Gait Phases']['Initial-Swing'] = idxISwright
        data['Left foot']['Gait Phases']['Mid-Swing'] = idxMSwleft
        data['Right foot']['Gait Phases']['Mid-Swing'] = idxMSwright
        data['Left foot']['Gait Phases']['Terminal-Swing'] = idxTSwleft
        data['Right foot']['Gait Phases']['Terminal-Swing'] = idxTSwright
        data['Left foot']['Gait Phases']['Loading Response'] = idxLRleft
        data['Right foot']['Gait Phases']['Loading Response'] = idxLRright
        data['Left foot']['Gait Phases']['Mid-Stance'] = idxMStleft
        data['Right foot']['Gait Phases']['Mid-Stance'] = idxMStright
        data['Left foot']['Gait Phases']['Terminal-Stance'] = idxTStleft
        data['Right foot']['Gait Phases']['Terminal-Stance'] = idxTStright
        data['Left foot']['Gait Phases']['Pre-Swing'] = idxPSwleft
        data['Right foot']['Gait Phases']['Pre-Swing'] = idxPSwright

    except:
        # Place gait events and gait phases in structured dictionary.
        data['Left foot']['Gait Events'] = dict()
        data['Right foot']['Gait Events'] = dict()
        data['Left foot']['Gait Phases'] = dict()
        data['Right foot']['Gait Phases'] = dict()
        data['Left foot']['Gait Events']['Terminal Contact'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Terminal Contact'] = np.array([], dtype=int)
        data['Left foot']['Gait Events']['Mid-Swing Onset'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Mid-Swing Onset'] = np.array([], dtype=int)
        data['Left foot']['Gait Events']['Terminal-Swing Onset'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Terminal-Swing Onset'] = np.array([], dtype=int)
        data['Left foot']['Gait Events']['Initial Contact'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Initial Contact'] = np.array([], dtype=int)
        data['Left foot']['Gait Events']['Mid-Stance Onset'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Mid-Stance Onset'] = np.array([], dtype=int)
        data['Left foot']['Gait Events']['Heel Off'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Heel Off'] = np.array([], dtype=int)
        data['Left foot']['Gait Events']['Pre-Swing Onset'] = np.array([], dtype=int)
        data['Right foot']['Gait Events']['Pre-Swing Onset'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Initial-Swing'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Initial-Swing'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Mid-Swing'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Mid-Swing'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Terminal-Swing'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Terminal-Swing'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Loading Response'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Loading Response'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Mid-Stance'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Mid-Stance'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Terminal-Stance'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Terminal-Stance'] = np.array([], dtype=int)
        data['Left foot']['Gait Phases']['Pre-Swing'] = np.array([], dtype=int)
        data['Right foot']['Gait Phases']['Pre-Swing'] = np.array([], dtype=int)

        errors['No walking period'] = True


    return data