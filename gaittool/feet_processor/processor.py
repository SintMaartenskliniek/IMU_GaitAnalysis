import numpy as np
import statistics as st

from ..helpers.preprocessor import data_filelist, data_preprocessor
from ..helpers.spatiotemporalfunctions import walkingsamples, ststransfers, nonactivesamples, stepcount, cadence, swingtime, stancetime, stridetime, steptime, doublesupport, velocity, gaitspeed, turnidentification, turnparameters, positionestimation, stridelength, asymmetry, leanangle, trunk_ROM, footAngle
from .analyzedata import analyzedata

def process(data, showfigure): 
    
    # Error handeling
    errors={}
    errors['Wrong sample frequency'] = False
    errors['High amount missing samples'] = False
    errors['No steady state gait'] = False
    errors['Trial type'] = False
    errors['No walking period'] = False
        
    # Find gait events and gait phases
    data, errors = analyzedata(data, errors, showfigure)

    # Determine Spatiotemporal parameters
    data['Spatiotemporals'] = dict()
    # Walkingtime / Non-active time
    data = walkingsamples(data, errors)  # data['Left foot']['Gait Phases']['Walking samples'] >> Right foot contains same data
    data['Spatiotemporals']['Walking time (s)'] = round((len(data['Left foot']['Gait Phases']['Walking samples'])/data['Sample Frequency (Hz)']),2)
        
    # Assumed is that a person is non-active when sample is not identified as walking.
    data = nonactivesamples(data)  # data['Spatiotemporals']['Non-Active samples']
    data['Spatiotemporals']['Non-Active time (s)'] = round((len(data['Left foot']['Gait Phases']['Non-Active samples'])/data['Sample Frequency (Hz)']),2)
    
    # Step count
    data = stepcount(data)  # data['Spatiotemporals']['Number of steps']
    
    # Cadence (steps per minute)
    data = cadence(data, errors)  # data['Spatiotemporals']['Cadence']

    # Swingtime
    # Time between Toe-Off and Heel-Strike of the same foot
    data = swingtime(data, errors)
    
    # Swingtimeasymmetry
    # For each measure, a value of 0.50 reflects perfect symmetry, >0.5 reflects left larger, <0.5 reflects right larger
    data['Spatiotemporals']['Swing time asymmetry'] = round(asymmetry(data['Spatiotemporals']['Swing time left (s)'], data['Spatiotemporals']['Swing time right (s)']),2)
    
    # Stancetime
    # Time between Heel-Strike and Toe-Off of the same foot
    data = stancetime(data, errors)
            
    # Stancetimeasymmetry
    # For each measure, a value of 0.50 reflects perfect symmetry, >0.5 reflects left larger, <0.5 reflects right larger
    data['Spatiotemporals']['Stance time asymmetry'] = round(asymmetry(data['Spatiotemporals']['Stance time left (s)'], data['Spatiotemporals']['Stance time right (s)']),2)
    
    # Stridetime
    # Time from Heel-Strike of one foot to next Heel-Strike of the same foot
    data = stridetime(data, errors)
    
    # Swing-Time as percentage of gaitcycle
    data['Spatiotemporals']['Swing time as percentage of gaitcycle left (%)'] = round((data['Spatiotemporals']['Swing time left (s)']/data['Spatiotemporals']['Stride time left (s)'])*100,2)
    data['Spatiotemporals']['Swing time as percentage of gaitcycle right (%)'] = round((data['Spatiotemporals']['Swing time right (s)']/data['Spatiotemporals']['Stride time right (s)'])*100,2)
    
    # Stance-Time as percentage of gaitcycle
    data['Spatiotemporals']['Stance time as percentage of gaitcycle left (%)'] = round((data['Spatiotemporals']['Stance time left (s)']/data['Spatiotemporals']['Stride time left (s)'])*100,2)
    data['Spatiotemporals']['Stance time as percentage of gaitcycle right (%)'] = round((data['Spatiotemporals']['Stance time right (s)']/data['Spatiotemporals']['Stride time right (s)'])*100,2)
    
    # Steptime
    # Time between Heel-Strike of one foot and Heel-Strike of the other foot
    data = steptime(data, errors)
        
    # Double-Support time
    # Time between Heel-Stike of one foot and Toe-Off of the other foot (time that both feet are in stance-phase)
    data = doublesupport(data, errors)

    # Double-Support time as percentage of gaitcycle
    data['Spatiotemporals']['Double support time as percentage of gaitcycle left (%)'] = round(data['Spatiotemporals']['Double support time left (s)']/data['Spatiotemporals']['Stride time left (s)']*100,1)
    data['Spatiotemporals']['Double support time as percentage of gaitcycle right (%)'] = round(data['Spatiotemporals']['Double support time right (s)']/data['Spatiotemporals']['Stride time right (s)']*100,1)
    
    # Single-limb support time
    data['Spatiotemporals']['Single-limb support time left (s)'] = data['Spatiotemporals']['Swing time right (s)']
    data['Spatiotemporals']['Single-limb support time right (s)'] = data['Spatiotemporals']['Swing time left (s)']
    
    # Single-limb support time as percentage of gaitcycle
    data['Spatiotemporals']['Single-limb support time as percentage of gaitcycle left (%)'] = round(data['Spatiotemporals']['Single-limb support time left (s)']/data['Spatiotemporals']['Stride time left (s)']*100,1)
    data['Spatiotemporals']['Single-limb support time as percentage of gaitcycle right (%)'] = round(data['Spatiotemporals']['Single-limb support time right (s)']/data['Spatiotemporals']['Stride time right (s)']*100,1)
        
    # Velocity estimation
    data = velocity(data, errors, showfigure)
    
    # Position estimation
    data = positionestimation(data, errors, showfigure)

    if 'Lumbar' not in data['Missing Sensors'] or 'Sternum' not in data['Missing Sensors']:
        # Turn identification
        data = turnidentification(data)
    
        # Peak angular velocity in turns
        data = turnparameters(data)
    
    # Stride length
    data = stridelength(data, errors, showfigure)
    data['Spatiotemporals']['Stride length - average steady state walking (m)'] = round(np.nanmedian(np.sort(np.append(data['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,2], data['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,2]) )), 2)
    data['Spatiotemporals']['Stride length - average all strides (m)'] = round(((data['Left foot']['derived']['Stride length - average all strides (m)'] + data['Right foot']['derived']['Stride length - average all strides (m)'])/2),2)

    # # Stride time
    # Part of gait speed function
     
    # Gait speed
    data = gaitspeed(data, errors)
        
    # Distance walked
    data['Spatiotemporals']['Walked distance (m)'] = round( np.mean([np.sum(data['Left foot']['derived']['Stride length - all strides (m)'][:,2]), np.sum(data['Right foot']['derived']['Stride length - all strides (m)'][:,2])])  , 2) #round((data['Spatiotemporals']['Gait speed (km/h)']/3.6 * data['Spatiotemporals']['Walking time (s)']), 2)
        
    # Calculate STS transfers if L-test is true
    if data['trialType'] == 'L-test' or data['trialType'] == 'STS-transfer':
        data, errors = ststransfers(data, errors)
        if 'Sternum' not in data['Missing Sensors']:
            ststime = round( np.sum(data['Sternum']['derived']['Transfers'][:,1]- data['Sternum']['derived']['Transfers'][:,0])/data['Sample Frequency (Hz)'] , 2 )
        elif 'Lumbar' not in data['Missing Sensors']:
            ststime = round( np.sum(data['Lumbar']['derived']['Transfers'][:,1] - data['Lumbar']['derived']['Transfers'][:,0])/data['Sample Frequency (Hz)'] , 2 ) 
        data['Spatiotemporals']['L-test time (s)'] = round( data['Spatiotemporals']['Walking time (s)']+ststime , 2)
    
    # Lean angle
    if 'Sternum' not in data['Missing Sensors'] and (data['trialType'] == 'L-test' or data['trialType'] == 'STS-transfer'):
        data = leanangle(data, showfigure)
    
    # Trunk coronal range of motion
    if 'Sternum' not in data['Missing Sensors']:
        data = trunk_ROM(data)
    
    # Check if sample frequency makes sense based on output:
    if errors['No walking period'] == False:
        if data['Spatiotemporals']['Gait speed (km/h)'] < 1 and data['Spatiotemporals']['Double support time as percentage of gaitcycle left (%)'] > 30 and data['Spatiotemporals']['Double support time as percentage of gaitcycle right (%)'] > 30:
            errors['Wrong sample frequency'] = True
        
    if errors['Wrong sample frequency'] == True:
        print('Please check sample frequency for this measurement in MTManager')
    if errors['No steady state gait'] == True:
        print('No steady state gait was detected, average gait speed was calculated over all strides including turns!')
    
    # Determine foot angle at IC and TC
    data = footAngle(data, errors)  
    
    return data, errors


def test_processor(datafolder, **kwargs):
    
    # Define visablitiy of debug-figures
    showfigure = 'hide'  # 'view'
    
    # Define list of filepaths and sensortype of the datarecordings
    filepaths, sensortype, sample_frequency = data_filelist(datafolder)
    
    if len(filepaths) > 0:
        # Define data dictionary with all sensordata
        if sample_frequency == False:
            data = data_preprocessor(filepaths, sensortype)
        else:
            data = data_preprocessor(filepaths, sensortype, sample_frequency=sample_frequency)
        
        # Determine trialType based on foldername or kwargs item
        # default is 'Unknown from foldername'
        data['trialType'] = 'Unknown from foldername'
        # Check if trialType in foldername
        if 'L-test' in datafolder:
            data['trialType'] = 'L-test'
        elif '2-minuten looptest' in datafolder:
            data['trialType'] = '2MWT'
        # Check if an overwrite in **kwargs items
        for key, value in kwargs.items():
            if key == 'trialType':
                data['trialType'] = value
            
        if len(data)>0:
            return process(data, showfigure)
    else:
        data = {}
        return data


