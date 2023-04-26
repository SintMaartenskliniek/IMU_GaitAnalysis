"""
Gait event functions

Last update: June 2021
Author: Carmen Ensink

"""
import numpy as np

def TSwOnset(gyroscopedata, indexToeOff):
    #      INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #                indexToeOff; index numbers of Toe Off points
    #      OUTPUT:   indexTerminalSwingOnset; index numbers of Terminal-Swing onset points
    
    #  Terminal-Swing Onset; zero-crossing (positive to negative)

    allTSwOns = np.array([])
    for i in range(0,(len(indexToeOff)-1)):
        databetweenTO = gyroscopedata[indexToeOff[i]:indexToeOff[i+1]]
        firstnegative = np.where(databetweenTO < 0)[0]
        if len(firstnegative) != 0:
            firstnegative = np.where(databetweenTO < 0)[0][0]
        else:
            firstnegative = 1
        lastpositive = firstnegative-1
        if lastpositive == -1:
            lastpositive = 0
        diffneg = abs(databetweenTO[firstnegative])
        diffpos = abs(databetweenTO[lastpositive])
        if diffneg<diffpos:
            idxTSwOns = indexToeOff[i]+firstnegative-1
        elif diffpos<diffneg:
            idxTSwOns = indexToeOff[i]+lastpositive-1
        elif diffpos == diffneg:
            idxTSwOns = indexToeOff[i]+firstnegative-1
    
        allTSwOns = np.append(allTSwOns, idxTSwOns)
    
    lastTO = indexToeOff[-1]
    dataafterLastTO = gyroscopedata[lastTO:]
    firstnegative = np.where(dataafterLastTO < 0)[0]
    if len(firstnegative) != 0:
        firstnegative = np.where(dataafterLastTO < 0)[0][0]
    else:
        firstnegative = 1
    
    lastpositive = firstnegative-1
    diffneg = abs(dataafterLastTO[firstnegative])
    diffpos = abs(dataafterLastTO[lastpositive])
    if diffneg < diffpos:
        idxTSwOns = indexToeOff[-1]+firstnegative-1
    elif diffpos < diffneg:
        idxTSwOns = indexToeOff[-1]+lastpositive-1
    elif diffpos == diffneg:
        idxTSwOns = indexToeOff[-1]+firstnegative-1

    allTSwOns = np.append(allTSwOns, idxTSwOns)
    indexTerminalSwingOnset = (np.sort(allTSwOns)).astype('int64')


    return indexTerminalSwingOnset
