"""
Gait phase functions

Last update: Januari 2021
Author: Carmen Ensink

"""
import numpy as np

def PreSwing(gyroscopedata,indexPreSwingOnset,indexToeOff):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexPreSwingOnset; index numbers of Pre-Swing Onset points
    #           indexToeOff; index numbers of Toe-Off points
    # OUTPUT:   indexPreSwingPhase; index numbers of Pre-Swing phase
    #
    # Pre-Swing phase from Pre-Swing Onset up to next Toe-Off
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range(0,len(indexPreSwingOnset)):
        # firstTO = np.where(indexToeOff > indexPreSwingOnset[i])[0] # find(indexToeOff>indexPreSwingOnset[i],1,'first');
        firstTO = np.argwhere(indexToeOff > indexPreSwingOnset[i]) # find(indexToeOff>indexPreSwingOnset[i],1,'first');
        if len(firstTO) != 0:
            firstTO = np.where(indexToeOff > indexPreSwingOnset[i])[0][0]
            logicindices[indexPreSwingOnset[i]:(indexToeOff[firstTO]-1)] = 1
    # indexPreSwingPhase = np.where(logicindices == 1)[0]
    indexPreSwingPhase = np.argwhere(logicindices == 1)
    
    return indexPreSwingPhase




def InitialSwing(gyroscopedata,indexToeOff,indexMidSwingOnset):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexToeOff; index numbers of Toe-Off points
    #           indexMidSwingOnset; index numbers of Mid-Swing Onset points
    # OUTPUT:   indexInitialSwingPhase; index numbers of Initial-Swing phase
    #
    # Initial-Swing phase from Toe-Off up to next Mid-Swing Onset
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range(0,len(indexToeOff)):
        # firstOnsMSw = np.where(indexMidSwingOnset > indexToeOff[i])[0] # find(indexMidSwingOnset>indexToeOff(i));
        firstOnsMSw = np.argwhere(indexMidSwingOnset > indexToeOff[i]) # find(indexMidSwingOnset>indexToeOff(i));
        if len(firstOnsMSw) != 0:
            firstOnsMSw = np.where(indexMidSwingOnset > indexToeOff[i])[0][0]
            logicindices[indexToeOff[i]:(indexMidSwingOnset[firstOnsMSw]-1)] = 1
    # indexInitialSwingPhase = np.where(logicindices == 1)[0]
    indexInitialSwingPhase = np.argwhere(logicindices == 1)
    
    return indexInitialSwingPhase




def MidSwing(gyroscopedata,indexMidSwingOnset,indexTerminalSwingOnset):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexMidSwingOnset; index numbers of Mid-Swing Onset points
    #           indexTerminalSwingOnset; index numbers of Terminal-Swing Onset points
    # OUTPUT:   indexMidSwingPhase; index numbers of Mid-Swing phase
    # 
    # Mid-Swing phase from Mid-Swing Onset up to next Terminal-Swing Onset
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range (0,len(indexMidSwingOnset)):
        firstOnsTSw = np.where(indexTerminalSwingOnset>indexMidSwingOnset[i])[0]
        if len(firstOnsTSw) != 0:
            firstOnsTSw = np.where(indexTerminalSwingOnset > indexMidSwingOnset[i])[0][0]
            logicindices[indexMidSwingOnset[i]:(indexTerminalSwingOnset[firstOnsTSw]-1)] = 1
    indexMidSwingPhase = np.where(logicindices == 1)[0]

    return indexMidSwingPhase




def TerminalSwing(gyroscopedata,indexTerminalSwingOnset,indexHeelStrike):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexTerminalSwingOnset; index numbers of Terminal-Swing Onset points
    #           indexHeelStrike; index numbers of Heel-Strike points
    # OUTPUT:   indexTerminalSwingPhase; index numbers of Terminal-Swing phase
    # 
    # Terminal-Swing phase from Terminal-Swing Onset up to next Heel-Strike
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range(0,len(indexTerminalSwingOnset)):
        firstHS = np.where(indexHeelStrike > indexTerminalSwingOnset[i])[0]
        if len(firstHS) != 0:
            firstHS = np.where(indexHeelStrike > indexTerminalSwingOnset[i])[0][0]
            logicindices[indexTerminalSwingOnset[i]:(indexHeelStrike[firstHS]-1)] = 1
    indexTerminalSwingPhase = np.where(logicindices == 1)[0]
    
    return indexTerminalSwingPhase



def LoadingResponse(gyroscopedata,indexHeelStrike,indexMidStanceOnset):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexHeelStrike; index numbers of Heel-Strike points
    #           indexMidStanceOnset; index numbers of Mid-Stance Onset points
    # OUTPUT:   indexLoadingResponsePhase; index numbers of Loading Response phase
    # 
    # Loading Response phase from Heel-Strike up to next Mid-Stance Onset
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range(0,len(indexHeelStrike)):
        firstMSt = np.where(indexMidStanceOnset > indexHeelStrike[i])[0]
        if len(firstMSt) != 0:
            firstMSt = np.where(indexMidStanceOnset > indexHeelStrike[i])[0][0]
            logicindices[indexHeelStrike[i]:(indexMidStanceOnset[firstMSt]-1)] = 1
    indexLoadingResponsePhase = np.where(logicindices == 1)[0]
    
    return indexLoadingResponsePhase




def MidStance(gyroscopedata,indexMidStanceOnset,indexHeelOff):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexMidStanceOnset; index numbers of Mid-Stance Onset points
    #           indexHeelOff; index numbers of Heel-Off points
    # OUTPUT:   indexMidStancePhase; index numbers of Mid-Stance phase
    # 
    # Mid-Stance phase from Mid-Stance Onset up to next Heel-Off
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range(0,len(indexMidStanceOnset)):
        firstHO = np.where(indexHeelOff > indexMidStanceOnset[i])[0]
        if len(firstHO) != 0:
            firstHO = np.where(indexHeelOff > indexMidStanceOnset[i])[0][0]
            logicindices[indexMidStanceOnset[i]:(indexHeelOff[firstHO]-1)] = 1
    indexMidStancePhase = np.where(logicindices == 1)[0]
    
    return indexMidStancePhase




def TerminalStance(gyroscopedata,indexHeelOff,indexPreSwingOnset):
    # INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #           indexHeelOff; index numbers of Heel-Off points
    #           indexPreSwingOnset; index numbers of Pre-Swing Onset points
    # OUTPUT:   indexTerminalSwingPhase; index numbers of Terminal-Stance phase
    # 
    # Terminal-Stance phase from Heel-Off up to next Pre-Swing Onset
    
    # logicindices = (np.zeros((len(gyroscopedata),1), dtype = 'int64')).flatten()
    logicindices = np.zeros((len(gyroscopedata)), dtype = 'int64')
    for i in range(0,len(indexHeelOff)):
        firstPSw = np.where(indexPreSwingOnset > indexHeelOff[i])[0]
        if len(firstPSw) != 0:
            firstPSw = np.where(indexPreSwingOnset > indexHeelOff[i])[0][0]
            logicindices[indexHeelOff[i]:(indexPreSwingOnset[firstPSw]-1)] = 1
    indexTerminalStancePhase = np.where(logicindices == 1)[0]

    return indexTerminalStancePhase


