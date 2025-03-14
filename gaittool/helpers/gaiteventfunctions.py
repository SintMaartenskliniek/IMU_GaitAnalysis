"""
Gait event functions

Last update: March 2025
Author: Carmen Ensink, MvM

"""
import numpy as np

def TSwOnset(gyroscopedata, indexToeOff):
    #      INPUT:    gyroscopedata; gyroscope data in medio-lateral direction
    #                indexToeOff; index numbers of Toe Off points
    #      OUTPUT:   indexTerminalSwingOnset; index numbers of Terminal-Swing onset points
    
    #   Old: Terminal-Swing Onset; zero-crossing (positive to negative)
    #   New 03-14-2025: Incorrect calculation Terminal-Swing Onset
    #   Update calculation Terminal-Swing Onset by MvM
    #   Defined as the peak in the gyroscopic data between the zero-crossing neg -> pos and 70% of the gaitcycle between TO-TO (Behboodi et al. 2019)

    allTSwOns = np.array([])
    cycle_len = np.array([])
    for i in range(0,(len(indexToeOff)-1)):
        databetweenTO = gyroscopedata[indexToeOff[i]:indexToeOff[i+1]]
        cycle_len = np.append(cycle_len,len(databetweenTO))
        firstnegative = np.where(databetweenTO < 0)[0]
        firstpositive = np.where(databetweenTO > 0)[0]
        if len(firstpositive)!=0:
            search_range_begin = firstpositive[0]
        else:
            search_range_begin = 0

        search_range_end = int(np.floor(0.7*cycle_len[-1]))
        idxTSwOns = indexToeOff[i] + np.where(databetweenTO == np.max(databetweenTO[search_range_begin:search_range_end]))
        allTSwOns = np.append(allTSwOns, int(idxTSwOns))
    
    lastTO = indexToeOff[-1]
    dataafterLastTO = gyroscopedata[lastTO:]
    if len(dataafterLastTO) > 0.5*np.median(cycle_len):
        firstpositive = np.where(dataafterLastTO > 0)[0]
        if len(firstpositive) != 0:
            search_range_begin = firstpositive[0]
        else:
            search_range_begin = 0
        idxTSwOns = lastTO + np.where(dataafterLastTO == np.max(dataafterLastTO[search_range_begin:]))
        allTSwOns = np.append(allTSwOns, int(idxTSwOns))

    indexTerminalSwingOnset = (np.sort(allTSwOns)).astype('int64')

    return indexTerminalSwingOnset
