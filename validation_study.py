# -*- coding: utf-8 -*-
"""
Validation study - MOTOR project
Sint Maartenskliniek study ID: 0900_Smarten_the_Clinic_V2

Author:         C.J. Ensink, c.ensink@maartenskliniek.nl
Last update:    14-12-2022

Main script

"""

# Import dependencies for data handeling
import numpy as np
import matplotlib.pyplot as plt # Plots
import pickle # Save/load data
import pandas as pd
import os

# Import dependencies for data analysis
from functions_validationstudy.functions_dataimport import dataimport, sort_files
from functions_validationstudy.functions_validationstudy import analyze_OMCS, stride_by_stride, bland_altman_plot_spatiotemporals_2, bland_altman_plot_spatiotemporals_4, histogram_gait_events, post_hoc_forcedata

from scipy import stats

# ---------- USER INPUTS REQUIRED ---------- #
# Set trialtype you wish to analyze to 'True'
analyze_trialtypes = dict()
analyze_trialtypes['Healthy GRAIL'] = True
analyze_trialtypes['Healthy Lab'] = True
analyze_trialtypes['CVA GRAIL'] = True

# Set wether or not a saved .pkl file can be found in you directory (stroredfile = True / False)
storedfile = False

if storedfile == True:
    filename = 'validation_study_dataset.pkl'
elif storedfile == False:
    # Define filepaths for vicon and xsens data
    datafolder = os.path.abspath('data')
    # Set name for file to be saved
    save_as = 'validation_study_dataset.pkl' 
        
# ---------- END USER INPUTS REQUIRED ---------- #

# If there is no .pkl file of the data available: analyze from raw data
if storedfile == False:
        
    # Data import
    corresponding_files, trialnames, OMCS, IMU, errors = dataimport(datafolder, analyze_trialtypes)
    
    # Save file inbetween (in case of error, at least all raw data is stored and does not have to be loaded again)
    f = open(save_as,"wb")
    a = {'OMCS':OMCS, 'IMU':IMU, 'corresponding_files':corresponding_files, 'trialnames':trialnames, 'analyze_trialtypes':analyze_trialtypes}
    pickle.dump(a,f)
    f.close()
    
# If there is a .pkl file of the data available: analyze from .pkl file
elif storedfile == True:
    # Open data file with analyzed gait data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        IMU = data['IMU']
        OMCS = data['OMCS']
        trialnames = data['trialnames']
        corresponding_files = data['corresponding_files']
    f.close()
    

# Gait event detection and calculate spatiotemporals from OMCS data
OMCS_gait_events, OMCS_spatiotemporals = analyze_OMCS(OMCS, IMU, trialnames)
# Compare on a stride by stride basis
SbS_IMU, SbS_OMCS = stride_by_stride(IMU, OMCS_gait_events, OMCS_spatiotemporals)


# Sort files on task
allfilenames = list(SbS_IMU['Initial contact'].keys())
files = sort_files (allfilenames)

# Check trial mean and variability (cov)
mean_IMU = dict()
mean_OMCS = dict()
mean_IMU['Healthy'] = dict()
mean_OMCS['Healthy'] = dict()
mean_IMU['Healthy']['Regular'] = dict()
mean_OMCS['Healthy']['Regular'] = dict()
mean_IMU['Healthy']['Irregular'] = dict()
mean_OMCS['Healthy']['Irregular'] = dict()
mean_IMU['Healthy']['Overground'] = dict()
mean_OMCS['Healthy']['Overground'] = dict()

mean_IMU['Stroke'] = dict()
mean_OMCS['Stroke'] = dict()
mean_IMU['Stroke']['Regular'] = dict()
mean_OMCS['Stroke']['Regular'] = dict()
mean_IMU['Stroke']['Irregular'] = dict()
mean_OMCS['Stroke']['Irregular'] = dict()

variability_IMU = dict()
variability_OMCS = dict()
variability_IMU['Healthy'] = dict()
variability_OMCS['Healthy'] = dict()
variability_IMU['Healthy']['Regular'] = dict()
variability_OMCS['Healthy']['Regular'] = dict()
variability_IMU['Healthy']['Irregular'] = dict()
variability_OMCS['Healthy']['Irregular'] = dict()
variability_IMU['Healthy']['Overground'] = dict()
variability_OMCS['Healthy']['Overground'] = dict()

variability_IMU['Stroke'] = dict()
variability_OMCS['Stroke'] = dict()
variability_IMU['Stroke']['Regular'] = dict()
variability_OMCS['Stroke']['Regular'] = dict()
variability_IMU['Stroke']['Irregular'] = dict()
variability_OMCS['Stroke']['Irregular'] = dict()

# Append strides for each spatiotemporal parameter from each subjectgroup and devide between regular and irregular walking tasks
strides_regular_IMU = dict()
strides_regular_IMU['Healthy'] = dict()
strides_regular_IMU['Stroke'] = dict()
strides_regular_OMCS = dict()
strides_regular_OMCS['Healthy'] = dict()
strides_regular_OMCS['Stroke'] = dict()

strides_irregular_IMU = dict()
strides_irregular_IMU['Healthy'] = dict()
strides_irregular_IMU['Stroke'] = dict()
strides_irregular_OMCS = dict()
strides_irregular_OMCS['Healthy'] = dict()
strides_irregular_OMCS['Stroke'] = dict()

strides_overground_IMU = dict()
strides_overground_OMCS = dict()

variability_matrix = np.array([[],[],[],[]]).transpose()

for parameter in SbS_IMU:
    strides_regular_IMU['Healthy'][parameter] = np.array([])
    strides_regular_OMCS['Healthy'][parameter] = np.array([])
    
    strides_regular_IMU['Stroke'][parameter] = np.array([])
    strides_regular_OMCS['Stroke'][parameter] = np.array([])
    
    strides_irregular_IMU['Healthy'][parameter] = np.array([])
    strides_irregular_OMCS['Healthy'][parameter] = np.array([])
    
    strides_irregular_IMU['Stroke'][parameter] = np.array([])
    strides_irregular_OMCS['Stroke'][parameter] = np.array([])
    
    strides_overground_IMU[parameter] = np.array([])
    strides_overground_OMCS[parameter] = np.array([])
    
    if parameter != 'Initial contact' and parameter != 'Terminal contact':
        mean_IMU['Healthy']['Regular'][parameter] = np.array([])
        mean_IMU['Healthy']['Irregular'][parameter] = np.array([])
        mean_IMU['Healthy']['Overground'][parameter] = np.array([])
        mean_OMCS['Healthy']['Regular'][parameter] = np.array([])
        mean_OMCS['Healthy']['Irregular'][parameter] = np.array([])
        mean_OMCS['Healthy']['Overground'][parameter] = np.array([])
        
        mean_IMU['Stroke']['Regular'][parameter] = np.array([])
        mean_IMU['Stroke']['Irregular'][parameter] = np.array([])  
        mean_OMCS['Stroke']['Regular'][parameter] = np.array([])
        mean_OMCS['Stroke']['Irregular'][parameter] = np.array([])
        
        variability_IMU['Healthy']['Regular'][parameter] = np.array([])
        variability_IMU['Healthy']['Irregular'][parameter] = np.array([])
        variability_IMU['Healthy']['Overground'][parameter] = np.array([])
        variability_OMCS['Healthy']['Regular'][parameter] = np.array([])
        variability_OMCS['Healthy']['Irregular'][parameter] = np.array([])
        variability_OMCS['Healthy']['Overground'][parameter] = np.array([])
        
        variability_IMU['Stroke']['Regular'][parameter] = np.array([])
        variability_IMU['Stroke']['Irregular'][parameter] = np.array([])  
        variability_OMCS['Stroke']['Regular'][parameter] = np.array([])
        variability_OMCS['Stroke']['Irregular'][parameter] = np.array([])   

for parameter in SbS_IMU:
    for trial in files['GRAIL healthy regular']:
        strides_regular_IMU['Healthy'][parameter] = np.append(strides_regular_IMU['Healthy'][parameter], SbS_IMU[parameter][trial])
        strides_regular_OMCS['Healthy'][parameter] = np.append(strides_regular_OMCS['Healthy'][parameter], SbS_OMCS[parameter][trial])
        
        if parameter != 'Initial contact' and parameter != 'Terminal contact':
            m_stp_imu = round(np.nanmean(SbS_IMU[parameter][trial]), 4)
            sd_stp_imu = round(np.nanstd(SbS_IMU[parameter][trial]), 4)
            mean_IMU['Healthy']['Regular'][parameter] = np.append(mean_IMU['Healthy']['Regular'][parameter], m_stp_imu)
            variability_IMU['Healthy']['Regular'][parameter] = np.append(variability_IMU['Healthy']['Regular'][parameter], (sd_stp_imu/m_stp_imu)*100)
            m_stp_omcs = round(np.nanmean(SbS_OMCS[parameter][trial]), 4)
            sd_stp_omcs = round(np.nanstd(SbS_OMCS[parameter][trial]), 4)
            mean_OMCS['Healthy']['Regular'][parameter] = np.append(mean_OMCS['Healthy']['Regular'][parameter], m_stp_omcs)
            variability_OMCS['Healthy']['Regular'][parameter] = np.append(variability_OMCS['Healthy']['Regular'][parameter], (sd_stp_omcs/m_stp_omcs)*100)
            
            variability_matrix = np.vstack((variability_matrix, np.array([trial, (sd_stp_omcs/m_stp_omcs)*100, (sd_stp_imu/m_stp_imu)*100, parameter]).transpose()))
    
    for trial in files['GRAIL healthy irregular']:
        strides_irregular_IMU['Healthy'][parameter] = np.append(strides_irregular_IMU['Healthy'][parameter], SbS_IMU[parameter][trial])
        strides_irregular_OMCS['Healthy'][parameter] = np.append(strides_irregular_OMCS['Healthy'][parameter], SbS_OMCS[parameter][trial])
        
        if parameter != 'Initial contact' and parameter != 'Terminal contact':
            m_stp_imu = round(np.nanmean(SbS_IMU[parameter][trial]), 4)
            sd_stp_imu = round(np.nanstd(SbS_IMU[parameter][trial]), 4)
            mean_IMU['Healthy']['Irregular'][parameter] = np.append(mean_IMU['Healthy']['Irregular'][parameter], m_stp_imu)
            variability_IMU['Healthy']['Irregular'][parameter] = np.append(variability_IMU['Healthy']['Irregular'][parameter], (sd_stp_imu/m_stp_imu)*100)
            m_stp_omcs = round(np.nanmean(SbS_OMCS[parameter][trial]), 4)
            sd_stp_omcs = round(np.nanstd(SbS_OMCS[parameter][trial]), 4)
            mean_OMCS['Healthy']['Irregular'][parameter] = np.append(mean_OMCS['Healthy']['Irregular'][parameter], m_stp_omcs)
            variability_OMCS['Healthy']['Irregular'][parameter] = np.append(variability_OMCS['Healthy']['Irregular'][parameter], (sd_stp_omcs/m_stp_omcs)*100)
            
            variability_matrix = np.vstack((variability_matrix, np.array([trial, (sd_stp_omcs/m_stp_omcs)*100, (sd_stp_imu/m_stp_imu)*100, parameter]).transpose()))
            
    for trial in files['GRAIL stroke regular']:
        strides_regular_IMU['Stroke'][parameter] = np.append(strides_regular_IMU['Stroke'][parameter], SbS_IMU[parameter][trial])
        strides_regular_OMCS['Stroke'][parameter] = np.append(strides_regular_OMCS['Stroke'][parameter], SbS_OMCS[parameter][trial])
        
        if parameter != 'Initial contact' and parameter != 'Terminal contact':
            m_stp_imu = round(np.nanmean(SbS_IMU[parameter][trial]), 4)
            sd_stp_imu = round(np.nanstd(SbS_IMU[parameter][trial]), 4)
            mean_IMU['Stroke']['Regular'][parameter] = np.append(mean_IMU['Stroke']['Regular'][parameter], m_stp_imu)
            variability_IMU['Stroke']['Regular'][parameter] = np.append(variability_IMU['Stroke']['Regular'][parameter], (sd_stp_imu/m_stp_imu)*100)
            m_stp_omcs = round(np.nanmean(SbS_OMCS[parameter][trial]), 4)
            sd_stp_omcs = round(np.nanstd(SbS_OMCS[parameter][trial]), 4)
            mean_OMCS['Stroke']['Regular'][parameter] = np.append(mean_OMCS['Stroke']['Regular'][parameter], m_stp_omcs)
            variability_OMCS['Stroke']['Regular'][parameter] = np.append(variability_OMCS['Stroke']['Regular'][parameter], (sd_stp_omcs/m_stp_omcs)*100)
            
            variability_matrix = np.vstack((variability_matrix, np.array([trial, (sd_stp_omcs/m_stp_omcs)*100, (sd_stp_imu/m_stp_imu)*100, parameter]).transpose()))
            
    for trial in files['GRAIL stroke irregular']:
        strides_irregular_IMU['Stroke'][parameter] = np.append(strides_irregular_IMU['Stroke'][parameter], SbS_IMU[parameter][trial])
        strides_irregular_OMCS['Stroke'][parameter] = np.append(strides_irregular_OMCS['Stroke'][parameter], SbS_OMCS[parameter][trial])
        
        if parameter != 'Initial contact' and parameter != 'Terminal contact':
            m_stp_imu = round(np.nanmean(SbS_IMU[parameter][trial]), 4)
            sd_stp_imu = round(np.nanstd(SbS_IMU[parameter][trial]), 4)
            mean_IMU['Stroke']['Irregular'][parameter] = np.append(mean_IMU['Stroke']['Irregular'][parameter], m_stp_imu)
            variability_IMU['Stroke']['Irregular'][parameter] = np.append(variability_IMU['Stroke']['Irregular'][parameter], (sd_stp_imu/m_stp_imu)*100)
            m_stp_omcs = round(np.nanmean(SbS_OMCS[parameter][trial]), 4)
            sd_stp_omcs = round(np.nanstd(SbS_OMCS[parameter][trial]), 4)
            mean_OMCS['Stroke']['Irregular'][parameter] = np.append(mean_OMCS['Stroke']['Irregular'][parameter], m_stp_omcs)
            variability_OMCS['Stroke']['Irregular'][parameter] = np.append(variability_OMCS['Stroke']['Irregular'][parameter], (sd_stp_omcs/m_stp_omcs)*100)
            
            # print(trial, ' cov OMCS = ', (sd_stp_omcs/m_stp_omcs)*100, ' cov IMU = ', (sd_stp_imu/m_stp_imu)*100, parameter)
            variability_matrix = np.vstack((variability_matrix, np.array([trial, (sd_stp_omcs/m_stp_omcs)*100, (sd_stp_imu/m_stp_imu)*100, parameter]).transpose()))
            
    for trial in files['Overground']:
        strides_overground_IMU[parameter] = np.append(strides_overground_IMU[parameter], SbS_IMU[parameter][trial])
        strides_overground_OMCS[parameter] = np.append(strides_overground_OMCS[parameter], SbS_OMCS[parameter][trial])
        
        if parameter != 'Initial contact' and parameter != 'Terminal contact':
            m_stp_imu = round(np.nanmean(SbS_IMU[parameter][trial]), 4)
            sd_stp_imu = round(np.nanstd(SbS_IMU[parameter][trial]), 4)
            mean_IMU['Healthy']['Overground'][parameter] = np.append(mean_IMU['Healthy']['Overground'][parameter], m_stp_imu)
            variability_IMU['Healthy']['Overground'][parameter] = np.append(variability_IMU['Healthy']['Overground'][parameter], (sd_stp_imu/m_stp_imu)*100)
            m_stp_omcs = round(np.nanmean(SbS_OMCS[parameter][trial]), 4)
            sd_stp_omcs = round(np.nanstd(SbS_OMCS[parameter][trial]), 4)
            mean_OMCS['Healthy']['Overground'][parameter] = np.append(mean_OMCS['Healthy']['Overground'][parameter], m_stp_omcs)
            variability_OMCS['Healthy']['Overground'][parameter] = np.append(variability_OMCS['Healthy']['Overground'][parameter], (sd_stp_omcs/m_stp_omcs)*100)
        
# Bland-Altman-like analysis
bland_altman_plot_spatiotemporals_2(strides_regular_IMU['Healthy']['Stride length (m)'], strides_regular_OMCS['Healthy']['Stride length (m)'], strides_irregular_IMU['Healthy']['Stride length (m)'], strides_irregular_OMCS['Healthy']['Stride length (m)'], eventType = 'Healthy  -  Stride length', unit = '(m)', group='healthy')
bland_altman_plot_spatiotemporals_2(strides_regular_IMU['Healthy']['Stride time (s)'], strides_regular_OMCS['Healthy']['Stride time (s)'], strides_irregular_IMU['Healthy']['Stride time (s)'], strides_irregular_OMCS['Healthy']['Stride time (s)'], eventType = 'Healthy  -  Stride time', unit = '(s)', group='healthy')
bland_altman_plot_spatiotemporals_2(strides_regular_IMU['Healthy']['Stride velocity (m/s)'], strides_regular_OMCS['Healthy']['Stride velocity (m/s)'], strides_irregular_IMU['Healthy']['Stride velocity (m/s)'], strides_irregular_OMCS['Healthy']['Stride velocity (m/s)'], eventType = 'Healthy  -  Stride velocity', unit = '(m/s)', group='healthy')

bland_altman_plot_spatiotemporals_2(strides_regular_IMU['Stroke']['Stride length (m)'], strides_regular_OMCS['Stroke']['Stride length (m)'], strides_irregular_IMU['Stroke']['Stride length (m)'], strides_irregular_OMCS['Stroke']['Stride length (m)'], eventType = 'Stroke  -  Stride length', unit = '(m)', group='stroke')
bland_altman_plot_spatiotemporals_2(strides_regular_IMU['Stroke']['Stride time (s)'], strides_regular_OMCS['Stroke']['Stride time (s)'], strides_irregular_IMU['Stroke']['Stride time (s)'], strides_irregular_OMCS['Stroke']['Stride time (s)'], eventType = 'Stroke  -  Stride time', unit = '(s)', group='stroke')
bland_altman_plot_spatiotemporals_2(strides_regular_IMU['Stroke']['Stride velocity (m/s)'], strides_regular_OMCS['Stroke']['Stride velocity (m/s)'], strides_irregular_IMU['Stroke']['Stride velocity (m/s)'], strides_irregular_OMCS['Stroke']['Stride velocity (m/s)'], eventType = 'Stroke  -  Stride velocity', unit = '(m/s)', group='stroke')

bland_altman_plot_spatiotemporals_2(strides_overground_IMU['Stride length (m)'], strides_overground_OMCS['Stride length (m)'], np.array([]), np.array([]), eventType = 'Overground  -  Stride length', unit = '(m)', group='healthy')
bland_altman_plot_spatiotemporals_2(strides_overground_IMU['Stride time (s)'], strides_overground_OMCS['Stride time (s)'], np.array([]), np.array([]), eventType = 'Overground  -  Stride time', unit = '(s)', group='healthy')
bland_altman_plot_spatiotemporals_2(strides_overground_IMU['Stride velocity (m/s)'], strides_overground_OMCS['Stride velocity (m/s)'], np.array([]), np.array([]), eventType = 'Overground  -  Stride velocity', unit = '(m/s)', group='healthy')
      
histogram_gait_events(strides_regular_IMU['Healthy']['Initial contact'], strides_regular_OMCS['Healthy']['Initial contact'], strides_irregular_IMU['Healthy']['Initial contact'], strides_irregular_OMCS['Healthy']['Initial contact'], label='Healthy - Initial contact')
histogram_gait_events(strides_regular_IMU['Healthy']['Terminal contact'], strides_regular_OMCS['Healthy']['Terminal contact'], strides_irregular_IMU['Healthy']['Terminal contact'], strides_irregular_OMCS['Healthy']['Terminal contact'], label='Healthy - Terminal contact')
histogram_gait_events(strides_regular_IMU['Stroke']['Initial contact'], strides_regular_OMCS['Stroke']['Initial contact'], strides_irregular_IMU['Stroke']['Initial contact'], strides_irregular_OMCS['Stroke']['Initial contact'], label='Stroke - Initial contact')
histogram_gait_events(strides_regular_IMU['Stroke']['Terminal contact'], strides_regular_OMCS['Stroke']['Terminal contact'], strides_irregular_IMU['Stroke']['Terminal contact'], strides_irregular_OMCS['Stroke']['Terminal contact'], label='Stroke - Terminal contact')     
histogram_gait_events(strides_overground_IMU['Initial contact'], strides_overground_OMCS['Initial contact'], np.array([]), np.array([]), label = 'Overground  -  Initial contact', unit = '(s)')
histogram_gait_events(strides_overground_IMU['Terminal contact'], strides_overground_OMCS['Terminal contact'], np.array([]), np.array([]), label = 'Overground  -  Terminal contact', unit = '(s)')


for parameter in mean_IMU['Healthy']['Regular']:
    bland_altman_plot_spatiotemporals_4(mean_IMU['Healthy']['Regular'][parameter], mean_OMCS['Healthy']['Regular'][parameter], mean_IMU['Healthy']['Irregular'][parameter], mean_OMCS['Healthy']['Irregular'][parameter], mean_IMU['Stroke']['Regular'][parameter], mean_OMCS['Stroke']['Regular'][parameter], mean_IMU['Stroke']['Irregular'][parameter], mean_OMCS['Stroke']['Irregular'][parameter], eventType = str(parameter), unit='')
    bland_altman_plot_spatiotemporals_4(variability_IMU['Healthy']['Regular'][parameter], variability_OMCS['Healthy']['Regular'][parameter], variability_IMU['Healthy']['Irregular'][parameter], variability_OMCS['Healthy']['Irregular'][parameter], variability_IMU['Stroke']['Regular'][parameter], variability_OMCS['Stroke']['Regular'][parameter], variability_IMU['Stroke']['Irregular'][parameter], variability_OMCS['Stroke']['Irregular'][parameter], eventType = str(parameter), unit='')


meancovimuHR=dict()
meancovomcsHR=dict()
meancovimuHI=dict()
meancovomcsHI=dict()
meancovimuSR=dict()
meancovomcsSR=dict()
meancovimuSI=dict()
meancovomcsSI=dict()
sdcovimuHR=dict()
sdcovomcsHR=dict()
sdcovimuHI=dict()
sdcovomcsHI=dict()
sdcovimuSR=dict()
sdcovomcsSR=dict()
sdcovimuSI=dict()
sdcovomcsSI=dict()
meanmimuHR=dict()
meanmomcsHR=dict()
meanmimuHI=dict()
meanmomcsHI=dict()
meanmimuSR=dict()
meanmomcsSR=dict()
meanmimuSI=dict()
meanmomcsSI=dict()
sdmimuHR=dict()
sdmomcsHR=dict()
sdmimuHI=dict()
sdmomcsHI=dict()
sdmimuSR=dict()
sdmomcsSR=dict()
sdmimuSI=dict()
sdmomcsSI=dict()
for parameter in mean_IMU['Healthy']['Regular']:
    meancovimuHR[parameter] = np.nanmean(variability_IMU['Healthy']['Regular'][parameter])
    meancovomcsHR[parameter] = np.nanmean(variability_OMCS['Healthy']['Regular'][parameter])
    meancovimuHI[parameter] = np.nanmean(variability_IMU['Healthy']['Irregular'][parameter])
    meancovomcsHI[parameter] = np.nanmean(variability_OMCS['Healthy']['Irregular'][parameter])
    meancovimuSR[parameter] = np.nanmean(variability_IMU['Stroke']['Regular'][parameter])
    meancovomcsSR[parameter] = np.nanmean(variability_OMCS['Stroke']['Regular'][parameter])
    meancovimuSI[parameter] = np.nanmean(variability_IMU['Stroke']['Irregular'][parameter])
    meancovomcsSI[parameter] = np.nanmean(variability_OMCS['Stroke']['Irregular'][parameter])
for parameter in mean_IMU['Healthy']['Regular']:
    sdcovimuHR[parameter] = np.nanstd(variability_IMU['Healthy']['Regular'][parameter])
    sdcovomcsHR[parameter] = np.nanstd(variability_OMCS['Healthy']['Regular'][parameter])
    sdcovimuHI[parameter] = np.nanstd(variability_IMU['Healthy']['Irregular'][parameter])
    sdcovomcsHI[parameter] = np.nanstd(variability_OMCS['Healthy']['Irregular'][parameter])
    sdcovimuSR[parameter] = np.nanstd(variability_IMU['Stroke']['Regular'][parameter])
    sdcovomcsSR[parameter] = np.nanstd(variability_OMCS['Stroke']['Regular'][parameter])
    sdcovimuSI[parameter] = np.nanstd(variability_IMU['Stroke']['Irregular'][parameter])
    sdcovomcsSI[parameter] = np.nanstd(variability_OMCS['Stroke']['Irregular'][parameter])
for parameter in mean_IMU['Healthy']['Regular']:
    meanmimuHR[parameter] = np.nanmean(mean_IMU['Healthy']['Regular'][parameter])
    meanmomcsHR[parameter] = np.nanmean(mean_OMCS['Healthy']['Regular'][parameter])
    meanmimuHI[parameter] = np.nanmean(mean_IMU['Healthy']['Irregular'][parameter])
    meanmomcsHI[parameter] = np.nanmean(mean_OMCS['Healthy']['Irregular'][parameter])
    meanmimuSR[parameter] = np.nanmean(mean_IMU['Stroke']['Regular'][parameter])
    meanmomcsSR[parameter] = np.nanmean(mean_OMCS['Stroke']['Regular'][parameter])
    meanmimuSI[parameter] = np.nanmean(mean_IMU['Stroke']['Irregular'][parameter])
    meanmomcsSI[parameter] = np.nanmean(mean_OMCS['Stroke']['Irregular'][parameter])
for parameter in mean_IMU['Healthy']['Regular']:
    sdmimuHR[parameter] = np.nanstd(mean_IMU['Healthy']['Regular'][parameter])
    sdmomcsHR[parameter] = np.nanstd(mean_OMCS['Healthy']['Regular'][parameter])
    sdmimuHI[parameter] = np.nanstd(mean_IMU['Healthy']['Irregular'][parameter])
    sdmomcsHI[parameter] = np.nanstd(mean_OMCS['Healthy']['Irregular'][parameter])
    sdmimuSR[parameter] = np.nanstd(mean_IMU['Stroke']['Regular'][parameter])
    sdmomcsSR[parameter] = np.nanstd(mean_OMCS['Stroke']['Regular'][parameter])
    sdmimuSI[parameter] = np.nanstd(mean_IMU['Stroke']['Irregular'][parameter])
    sdmomcsSI[parameter] = np.nanstd(mean_OMCS['Stroke']['Irregular'][parameter])
    
# Length and weights of participants
heightsHealthy = np.array([168, 164,  166,  165,  183,  173,  168,  179, 186, 181,  180,  180,  170,  162,  166,  182, 174, 180,  179,  176])
weightsHealthy = np.array([72,  74.8, 76.8, 67.8, 77.2, 62.4, 63.6, 69,  93,  77.6, 78.2, 88.6, 68.4, 66.2, 70.4, 77,  70,  76.8, 89.2, 73.4])
heightsStroke  = np.array([162, 183,  178,  181,  171,  176,  184,  184, 180, 165])
weightsStroke  = np.array([70,  82,   71,   93,   91,   71,   95.4, 85,  76,  77])
agesHealthy    = np.array([54,  69,   68,   68,   63,   51,   41,  43,  55,   41,   45,   53,   41,   59,   61,   79,  73,  72,   76,   72])
agesStroke     = np.array([65,  57,   83,   60,   56,   45,   72,  68,  57,   49])
genderHealthy  = np.array([10, 10]) #list    (['f', 'f',  'm',  'f',  'm',  'f',  'f', 'm', 'm',  'm',  'f',  'm',  'f',  'f',  'f',  'm', 'f', 'm',  'm',  'm'])
genderStroke   = np.array([7,  3])  #list    (['f', 'm',  'm',  'm',  'm',  'f',  'm', 'm', 'm',  'f'])


meanHeightHealthy = np.mean(heightsHealthy)
sdHeightHealthy = np.std(heightsHealthy)
meanWeightHealthy = np.mean(weightsHealthy)
sdWeightHealthy = np.std(weightsHealthy)
meanHeightStroke = np.mean(heightsStroke)
sdHeightStroke = np.std(heightsStroke)
meanWeightStroke = np.mean(weightsStroke)
sdWeightStroke = np.std(weightsStroke)

# Check for differences between group characteristics
print('Mann-Whitney_U test for differences between height of the groups: ', stats.mannwhitneyu(heightsHealthy, heightsStroke))
print('Mann-Whitney_U test for differences between weight of the groups: ', stats.mannwhitneyu(weightsHealthy, weightsStroke))
print('Mann-Whitney_U test for differences between age of the groups: ', stats.mannwhitneyu(agesHealthy, agesStroke))
print('Chi-square test for differences between genderdistribution of the groups: ', stats.chisquare(f_obs=genderHealthy/20, f_exp=genderStroke/10))

# Uncomment this part if you wish to save an .xls file for further statistical analysis in R.
R_formatted_data = pd.DataFrame(columns=(['SubjectID', 'Trialtype', 'StrideNr', 'Stride time sensor', 'Stride length sensor', 'Stride velocity sensor', 'Stride time OMCS', 'Stride length OMCS', 'Stride velocity OMCS']))
for trial in SbS_IMU['Stride length (m)']:
    if trial in files['GRAIL healthy regular']:
        trialtype = 'GRAIL healthy regular'
    elif trial in files['GRAIL healthy irregular']:
        trialtype = 'GRAIL healthy irregular'
    elif trial in files['GRAIL stroke regular']:
        trialtype = 'GRAIL stroke regular'
    elif trial in files['GRAIL stroke irregular']:
        trialtype = 'GRAIL stroke irregular'
    elif trial in files['Overground']:
        trialtype = 'Overground'
        
    for i in range(0, len(SbS_IMU['Initial contact'][trial])):
        df2 = pd.DataFrame({'SubjectID': [trial],
                        'Trialtype': [trialtype],
                        'StrideNr': [i],
                        
                        'Stride time sensor': [SbS_IMU['Stride time (s)'][trial][i]],
                        'Stride length sensor': [SbS_IMU['Stride length (m)'][trial][i]],
                        'Stride velocity sensor': [SbS_IMU['Stride velocity (m/s)'][trial][i]],
                        'Initial contact sensor': [SbS_IMU['Initial contact'][trial][i]],
                        'Terminal contact sensor': [SbS_IMU['Terminal contact'][trial][i]],
                        
                        'Stride time OMCS': [SbS_OMCS['Stride time (s)'][trial][i]],
                        'Stride length OMCS': [SbS_OMCS['Stride length (m)'][trial][i]],
                        'Stride velocity OMCS': [SbS_OMCS['Stride velocity (m/s)'][trial][i]],
                        'Initial contact OMCS': [SbS_OMCS['Initial contact'][trial][i]],
                        'Terminal contact OMCS': [SbS_OMCS['Terminal contact'][trial][i]] })

        if df2.isnull().values.any() == False:
            # for appending df2 at the end of df1
            R_formatted_data = R_formatted_data.append(df2, ignore_index = True)
R_formatted_data.to_excel('R_dataset.xlsx', index=False)
print ('Saved .xls file for statistical analysis in R')



# Post-hoc analysis (forceplate data)
forceplate_gait_events = post_hoc_forcedata(OMCS, OMCS_spatiotemporals, IMU, files)
all_force_OMCS_IC = np.array([], dtype=int)
all_force_OMCS_TC = np.array([], dtype=int)
all_force_IMU_IC = np.array([], dtype=int)
all_force_IMU_TC = np.array([], dtype=int)
all_OMCS_force_IC = np.array([], dtype = int)
all_OMCS_force_TC = np.array([], dtype = int)
all_IMU_force_IC = np.array([], dtype = int)
all_IMU_force_TC = np.array([], dtype = int)
for trial in forceplate_gait_events['IC left force with OMCS']:
    all_force_OMCS_IC = np.append(all_force_OMCS_IC, forceplate_gait_events['IC left force with OMCS'][trial])
    all_force_OMCS_IC = np.append(all_force_OMCS_IC, forceplate_gait_events['IC right force with OMCS'][trial])
    all_force_OMCS_TC = np.append(all_force_OMCS_TC, forceplate_gait_events['TC left force with OMCS'][trial])
    all_force_OMCS_TC = np.append(all_force_OMCS_TC, forceplate_gait_events['TC right force with OMCS'][trial])
    all_force_IMU_IC = np.append(all_force_IMU_IC, forceplate_gait_events['IC left force with IMU'][trial])
    all_force_IMU_IC = np.append(all_force_IMU_IC, forceplate_gait_events['IC right force with IMU'][trial])
    all_force_IMU_TC = np.append(all_force_IMU_TC, forceplate_gait_events['TC left force with IMU'][trial])
    all_force_IMU_TC = np.append(all_force_IMU_TC, forceplate_gait_events['TC right force with IMU'][trial])
    all_OMCS_force_IC = np.append(all_OMCS_force_IC, forceplate_gait_events['IC left OMCS with force'][trial])
    all_OMCS_force_IC = np.append(all_OMCS_force_IC, forceplate_gait_events['IC right OMCS with force'][trial])
    all_OMCS_force_TC = np.append(all_OMCS_force_TC, forceplate_gait_events['TC left OMCS with force'][trial])
    all_OMCS_force_TC = np.append(all_OMCS_force_TC, forceplate_gait_events['TC right OMCS with force'][trial])
    all_IMU_force_IC = np.append(all_IMU_force_IC, forceplate_gait_events['IC left IMU with force'][trial])
    all_IMU_force_IC = np.append(all_IMU_force_IC, forceplate_gait_events['IC right IMU with force'][trial])
    all_IMU_force_TC = np.append(all_IMU_force_TC, forceplate_gait_events['TC left IMU with force'][trial])
    all_IMU_force_TC = np.append(all_IMU_force_TC, forceplate_gait_events['TC right IMU with force'][trial])

histogram_gait_events(all_OMCS_force_IC, all_force_OMCS_IC, all_IMU_force_IC, all_force_IMU_IC, label='Initial contact: OMCS/IMU - forceplate')
histogram_gait_events(all_OMCS_force_TC, all_force_OMCS_TC, all_IMU_force_TC, all_force_IMU_TC, label='Terminal contact: OMCS/IMU - forceplate')
mdIC_OMCS_force = round(np.nanmean(all_OMCS_force_IC/100 - all_force_OMCS_IC/100), 2)
mdIC_IMU_force = round(np.nanmean(all_IMU_force_IC/100 - all_force_IMU_IC/100), 2)
mdTC_OMCS_force = round(np.nanmean(all_OMCS_force_TC/100 - all_force_OMCS_TC/100), 2)
mdTC_IMU_force = round(np.nanmean(all_IMU_force_TC/100 - all_force_IMU_TC/100), 2)

sdIC_OMCS_force = round(np.nanstd(all_OMCS_force_IC/100 - all_force_OMCS_IC/100), 2)
sdIC_IMU_force = round(np.nanstd(all_IMU_force_IC/100 - all_force_IMU_IC/100), 2)
sdTC_OMCS_force = round(np.nanstd(all_OMCS_force_TC/100 - all_force_OMCS_TC/100), 2)
sdTC_IMU_force = round(np.nanstd(all_IMU_force_TC/100 - all_force_IMU_TC/100), 2)

loaIC_OMCS_force = np.array([round(mdIC_OMCS_force -1.96*sdIC_OMCS_force, 2) , round(mdIC_OMCS_force +1.96*sdIC_OMCS_force, 2)])
loaIC_IMU_force = np.array([round(mdIC_IMU_force -1.96*sdIC_IMU_force , 2) , round(mdIC_IMU_force +1.96*sdIC_IMU_force , 2)])
loaTC_OMCS_force = np.array([round(mdTC_OMCS_force -1.96*sdTC_OMCS_force, 2) , round(mdTC_OMCS_force +1.96*sdTC_OMCS_force, 2)])
loaTC_IMU_force = np.array([round(mdTC_IMU_force -1.96*sdTC_IMU_force , 2) , round(mdTC_IMU_force +1.96*sdTC_IMU_force , 2)])
# histogram_gait_events(all_IMU_force_IC, all_force_IMU_IC, np.array([]), np.array([]), label='Initial contact: IMU - forceplate')
# histogram_gait_events(all_IMU_force_TC, all_force_IMU_TC, np.array([]), np.array([]), label='Terminal contact: IMU - forceplate')


# Calculate mean, sd and LoA for stride length, gait speed, and stride time
md_healthy_regular = {}
md_healthy_irregular = {}
md_stroke_regular = {}
md_stroke_irregular = {}
md_overground = {}

sd_healthy_regular = {}
sd_healthy_irregular = {}
sd_stroke_regular = {}
sd_stroke_irregular = {}
sd_overground = {}

loa_healthy_regular = {}
loa_healthy_irregular = {}
loa_stroke_regular = {}
loa_stroke_irregular = {}
loa_overground = {}

for subjectgroup in strides_regular_IMU:
    if subjectgroup == 'Healthy':
        for par in strides_regular_IMU[subjectgroup]:
            if par =='Initial contact' or par == 'Terminal contact': 
                md_healthy_regular[par] = round(np.nanmean(strides_regular_IMU[subjectgroup][par]/100 - strides_regular_OMCS[subjectgroup][par]/100), 3)
                md_healthy_irregular[par] = round(np.nanmean(strides_irregular_IMU[subjectgroup][par]/100 - strides_irregular_OMCS[subjectgroup][par]/100), 3)
                sd_healthy_regular[par] = round(np.nanstd(strides_regular_IMU[subjectgroup][par]/100 - strides_regular_OMCS[subjectgroup][par]/100), 3)
                sd_healthy_irregular[par] = round(np.nanstd(strides_irregular_IMU[subjectgroup][par]/100 - strides_irregular_OMCS[subjectgroup][par]/100), 3)
                
            else:
                md_healthy_regular[par] = round(np.nanmean(strides_regular_IMU[subjectgroup][par] - strides_regular_OMCS[subjectgroup][par]), 2)
                md_healthy_irregular[par] = round(np.nanmean(strides_irregular_IMU[subjectgroup][par] - strides_irregular_OMCS[subjectgroup][par]), 2)
                sd_healthy_regular[par] = round(np.nanstd(strides_regular_IMU[subjectgroup][par] - strides_regular_OMCS[subjectgroup][par]), 2)
                sd_healthy_irregular[par] = round(np.nanstd(strides_irregular_IMU[subjectgroup][par] - strides_irregular_OMCS[subjectgroup][par]), 2)
            loa_healthy_regular[par] = np.array([ round(md_healthy_regular[par] - 1.96*sd_healthy_regular[par], 2)  ,  round(md_healthy_regular[par] + 1.96*sd_healthy_regular[par], 2) ])
            loa_healthy_irregular[par] = np.array([ round(md_healthy_irregular[par] - 1.96*sd_healthy_irregular[par], 2)  ,  round(md_healthy_irregular[par] + 1.96*sd_healthy_irregular[par], 2) ])
    elif subjectgroup == 'Stroke':
        for par in strides_regular_IMU[subjectgroup]:
            if par =='Initial contact' or par == 'Terminal contact':
                md_stroke_regular[par] = round(np.nanmean(strides_regular_IMU[subjectgroup][par]/100 - strides_regular_OMCS[subjectgroup][par]/100), 3)
                md_stroke_irregular[par] = round(np.nanmean(strides_irregular_IMU[subjectgroup][par]/100 - strides_irregular_OMCS[subjectgroup][par]/100), 3)
                sd_stroke_regular[par] = round(np.nanstd(strides_regular_IMU[subjectgroup][par]/100 - strides_regular_OMCS[subjectgroup][par]/100), 3)
                sd_stroke_irregular[par] = round(np.nanstd(strides_irregular_IMU[subjectgroup][par]/100 - strides_irregular_OMCS[subjectgroup][par]/100), 3)
            else:
                md_stroke_regular[par] = round(np.nanmean(strides_regular_IMU[subjectgroup][par] - strides_regular_OMCS[subjectgroup][par]), 2)
                md_stroke_irregular[par] = round(np.nanmean(strides_irregular_IMU[subjectgroup][par] - strides_irregular_OMCS[subjectgroup][par]), 2)
                sd_stroke_regular[par] = round(np.nanstd(strides_regular_IMU[subjectgroup][par] - strides_regular_OMCS[subjectgroup][par]), 2)
                sd_stroke_irregular[par] = round(np.nanstd(strides_irregular_IMU[subjectgroup][par] - strides_irregular_OMCS[subjectgroup][par]), 2)
            loa_stroke_regular[par] = np.array([ round(md_stroke_regular[par] - 1.96*sd_stroke_regular[par], 2)  ,  round(md_stroke_regular[par] + 1.96*sd_stroke_regular[par], 2) ])
            loa_stroke_irregular[par] = np.array([ round(md_stroke_irregular[par] - 1.96*sd_stroke_irregular[par], 2)  ,  round(md_stroke_irregular[par] + 1.96*sd_stroke_irregular[par], 2) ])

for par in strides_overground_IMU:
    if par =='Initial contact' or par == 'Terminal contact':
        md_overground[par] = round(np.nanmean(strides_overground_IMU[par]/100 - strides_overground_OMCS[par]/100), 3)
        sd_overground[par] = round(np.nanstd(strides_overground_IMU[par]/100 - strides_overground_OMCS[par]/100), 3)
    else:
        md_overground[par] = round(np.nanmean(strides_overground_IMU[par] - strides_overground_OMCS[par]), 2)
        sd_overground[par] = round(np.nanstd(strides_overground_IMU[par] - strides_overground_OMCS[par]), 2)
    loa_overground[par] = np.array([ round(md_overground[par] - 1.96*sd_overground[par], 2)  ,  round(md_overground[par] + 1.96*sd_overground[par], 2) ])
    

# Calculate mean, sd and LoA for stride length, gait speed, and stride time per subject 
meansubjects = dict()
sdsubjects = dict()
meansubjects['Regular'] = pd.DataFrame([], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridetime', 'meanstridelength', 'meanstridevelocity', 'ic', 'tc'])
meansubjects['Irregular'] = pd.DataFrame([], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridetime', 'meanstridelength', 'meanstridevelocity', 'ic', 'tc'])
sdsubjects['Regular'] = pd.DataFrame([], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridevelocity', 'ic', 'tc'])
sdsubjects['Irregular'] = pd.DataFrame([], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridevelocity', 'ic', 'tc'])
for trial in files['GRAIL regular']:
    mst = round(np.nanmean(SbS_IMU['Stride time (s)'][trial] - SbS_OMCS['Stride time (s)'][trial]), 8)
    msl = round(np.nanmean(SbS_IMU['Stride length (m)'][trial] - SbS_OMCS['Stride length (m)'][trial]), 8)
    msv = round(np.nanmean(SbS_IMU['Stride velocity (m/s)'][trial] - SbS_OMCS['Stride velocity (m/s)'][trial]), 8)
    meanst = np.nanmean(np.nanmean(np.array([SbS_IMU['Stride time (s)'][trial], SbS_OMCS['Stride time (s)'][trial]]), axis=0))
    meansl = np.nanmean(np.nanmean(np.array([SbS_IMU['Stride length (m)'][trial], SbS_OMCS['Stride length (m)'][trial]]), axis=0))
    meansv = np.nanmean(np.nanmean(np.array([SbS_IMU['Stride velocity (m/s)'][trial], SbS_OMCS['Stride velocity (m/s)'][trial]]), axis=0))
    mic = round(np.nanmean((SbS_IMU['Initial contact'][trial] - SbS_OMCS['Initial contact'][trial])/100), 8)
    mtc = round(np.nanmean((SbS_IMU['Terminal contact'][trial] - SbS_OMCS['Terminal contact'][trial])/100), 8)
    df1 = pd.DataFrame([[trial, mst, msl, msv, meanst, meansl, meansv, mic, mtc]], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridetime', 'meanstridelength', 'meanstridevelocity', 'ic', 'tc'])
    meansubjects['Regular'] = pd.concat([meansubjects['Regular'], df1], ignore_index=True)
    
    sdst = round(np.nanstd(SbS_IMU['Stride time (s)'][trial] - SbS_OMCS['Stride time (s)'][trial]), 8)
    sdsl = round(np.nanstd(SbS_IMU['Stride length (m)'][trial] - SbS_OMCS['Stride length (m)'][trial]), 8)
    sdsv = round(np.nanstd(SbS_IMU['Stride velocity (m/s)'][trial] - SbS_OMCS['Stride velocity (m/s)'][trial]), 8)
    sdic = round(np.nanstd((SbS_IMU['Initial contact'][trial] - SbS_OMCS['Initial contact'][trial])/100), 8)
    sdtc = round(np.nanstd((SbS_IMU['Terminal contact'][trial] - SbS_OMCS['Terminal contact'][trial])/100), 8)
    df2 = pd.DataFrame([[trial, sdst, sdsl, sdsv, meansv, sdic, sdtc]], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridevelocity', 'ic', 'tc'])
    sdsubjects['Regular'] = pd.concat([sdsubjects['Regular'], df2], ignore_index=True)

for trial in files['GRAIL irregular']:
    mst = round(np.nanmean(SbS_IMU['Stride time (s)'][trial] - SbS_OMCS['Stride time (s)'][trial]), 8)
    msl = round(np.nanmean(SbS_IMU['Stride length (m)'][trial] - SbS_OMCS['Stride length (m)'][trial]), 8)
    msv = round(np.nanmean(SbS_IMU['Stride velocity (m/s)'][trial] - SbS_OMCS['Stride velocity (m/s)'][trial]), 8)
    meanst = np.nanmean(np.nanmean(np.array([SbS_IMU['Stride time (s)'][trial], SbS_OMCS['Stride time (s)'][trial]]), axis=0))
    meansl = np.nanmean(np.nanmean(np.array([SbS_IMU['Stride length (m)'][trial], SbS_OMCS['Stride length (m)'][trial]]), axis=0))
    meansv = np.nanmean(np.nanmean(np.array([SbS_IMU['Stride velocity (m/s)'][trial], SbS_OMCS['Stride velocity (m/s)'][trial]]), axis=0))
    mic = round(np.nanmean((SbS_IMU['Initial contact'][trial] - SbS_OMCS['Initial contact'][trial])/100), 8)
    mtc = round(np.nanmean((SbS_IMU['Terminal contact'][trial] - SbS_OMCS['Terminal contact'][trial])/100), 8)
    df1 = pd.DataFrame([[trial, mst, msl, msv, meanst, meansl, meansv, mic, mtc]], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridetime', 'meanstridelength', 'meanstridevelocity', 'ic', 'tc'])
    meansubjects['Irregular'] = pd.concat([meansubjects['Irregular'], df1], ignore_index=True)
    
    sdst = round(np.nanstd(SbS_IMU['Stride time (s)'][trial] - SbS_OMCS['Stride time (s)'][trial]), 8)
    sdsl = round(np.nanstd(SbS_IMU['Stride length (m)'][trial] - SbS_OMCS['Stride length (m)'][trial]), 8)
    sdsv = round(np.nanstd(SbS_IMU['Stride velocity (m/s)'][trial] - SbS_OMCS['Stride velocity (m/s)'][trial]), 8)
    sdic = round(np.nanstd((SbS_IMU['Initial contact'][trial] - SbS_OMCS['Initial contact'][trial])/100), 8)
    sdtc = round(np.nanstd((SbS_IMU['Terminal contact'][trial] - SbS_OMCS['Terminal contact'][trial])/100), 8)
    df2 = pd.DataFrame([[trial, sdst, sdsl, sdsv, meansv, sdic, sdtc]], columns=['trialname', 'stridetime', 'stridelength', 'stridevelocity', 'meanstridevelocity', 'ic', 'tc'])
    sdsubjects['Irregular'] = pd.concat([sdsubjects['Irregular'], df2], ignore_index=True)



cm = 1/2.54
# Healthy gait events
healthy_irr_m = meansubjects['Irregular'][meansubjects['Irregular']['trialname'].str.contains("_V_")]
healthy_reg_m = meansubjects['Regular'][meansubjects['Regular']['trialname'].str.contains("_V_")]
healthy_irr_sd = sdsubjects['Irregular'][sdsubjects['Irregular']['trialname'].str.contains("_V_")]
healthy_reg_sd = sdsubjects['Regular'][sdsubjects['Regular']['trialname'].str.contains("_V_")]
healthy_irr_m=healthy_irr_m.sort_values(by='meanstridevelocity', ignore_index=True)
healthy_reg_m=healthy_reg_m.sort_values(by='meanstridevelocity', ignore_index=True)
healthy_irr_sd=healthy_irr_sd.sort_values(by='meanstridevelocity', ignore_index=True)
healthy_reg_sd=healthy_reg_sd.sort_values(by='meanstridevelocity', ignore_index=True)

# Initial contact
j=0
fig = plt.subplots(nrows=1, ncols=1, figsize=(40*cm, 25*cm))
for i in range(0,len(healthy_irr_m['trialname'])):
    j+=1
    plt.scatter(healthy_irr_m['ic'][i], np.array([j]), edgecolor = '#E79419', facecolor='None', marker = 'o', s=100, label='Irregular gait') # SMK ORANGE
    plt.plot(np.array([healthy_irr_m['ic'][i]-healthy_irr_sd['ic'][i], healthy_irr_m['ic'][i]+healthy_irr_sd['ic'][i]]), np.array([j,j]), linestyle='-', color='#E79419')
for i in range(0,len(healthy_reg_m['trialname'])):
    j+=1
    plt.scatter(healthy_reg_m['ic'][i], np.array([j]), edgecolor = '#004D43', facecolor='None', marker = 'o', s=100, label='Regular gait') # SMK GREEN
    plt.plot(np.array([healthy_reg_m['ic'][i]-healthy_reg_sd['ic'][i], healthy_reg_m['ic'][i]+healthy_reg_sd['ic'][i]]), np.array([j,j]), linestyle='-', color='#004D43')
plt.yticks(ticks=[], labels=[])
plt.xticks(ticks=list(np.arange(start=-0.2, stop=0.21, step=0.02)), labels=list(np.arange(start=-0.2, stop=0.21, step=0.02).round(decimals=2)), fontsize=15)
    
# Terminal contact
j=0
fig = plt.subplots(nrows=1, ncols=1, figsize=(40*cm, 25*cm))
for i in range(0,len(healthy_irr_m['trialname'])):
    j+=1
    plt.scatter(healthy_irr_m['tc'][i], np.array([j]), edgecolor = '#E79419', facecolor='None', marker = 'o', s=100, label='Irregular gait') # SMK ORANGE
    plt.plot(np.array([healthy_irr_m['tc'][i]-healthy_irr_sd['tc'][i], healthy_irr_m['tc'][i]+healthy_irr_sd['tc'][i]]), np.array([j,j]), linestyle='-', color='#E79419')
for i in range(0,len(healthy_reg_m['trialname'])):
    j+=1
    plt.scatter(healthy_reg_m['tc'][i], np.array([j]), edgecolor = '#004D43', facecolor='None', marker = 'o', s=100, label='Regular gait') # SMK GREEN
    plt.plot(np.array([healthy_reg_m['tc'][i]-healthy_reg_sd['tc'][i], healthy_reg_m['tc'][i]+healthy_reg_sd['tc'][i]]), np.array([j,j]), linestyle='-', color='#004D43')
plt.yticks(ticks=[], labels=[])
plt.xticks(ticks=list(np.arange(start=-0.2, stop=0.21, step=0.02)), labels=list(np.arange(start=-0.2, stop=0.21, step=0.02).round(decimals=2)), fontsize=15)


# Stroke gait events
stroke_irr_m = meansubjects['Irregular'][meansubjects['Irregular']['trialname'].str.contains("_CVA_")]
stroke_reg_m = meansubjects['Regular'][meansubjects['Regular']['trialname'].str.contains("_CVA_")]
stroke_irr_sd = sdsubjects['Irregular'][sdsubjects['Irregular']['trialname'].str.contains("_CVA_")]
stroke_reg_sd = sdsubjects['Regular'][sdsubjects['Regular']['trialname'].str.contains("_CVA_")]
stroke_irr_m=stroke_irr_m.sort_values(by='meanstridevelocity', ignore_index=True)
stroke_reg_m=stroke_reg_m.sort_values(by='meanstridevelocity', ignore_index=True)
stroke_irr_sd=stroke_irr_sd.sort_values(by='meanstridevelocity', ignore_index=True)
stroke_reg_sd=stroke_reg_sd.sort_values(by='meanstridevelocity', ignore_index=True)

# Initial contact
j=0
fig = plt.subplots(nrows=1, ncols=1, figsize=(40*cm, 25*cm))
for i in range(0,len(stroke_irr_m['trialname'])):
    j+=1
    plt.scatter(stroke_irr_m['ic'][i], np.array([j]), edgecolor = '#E79419', facecolor='None', marker = 'o', s=100, label='Irregular gait') # SMK ORANGE
    plt.plot(np.array([stroke_irr_m['ic'][i]-stroke_irr_sd['ic'][i], stroke_irr_m['ic'][i]+stroke_irr_sd['ic'][i]]), np.array([j,j]), linestyle='-', color='#E79419')
for i in range(0,len(stroke_reg_m['trialname'])):
    j+=1
    plt.scatter(stroke_reg_m['ic'][i], np.array([j]), edgecolor = '#004D43', facecolor='None', marker = 'o', s=100, label='Regular gait') # SMK GREEN
    plt.plot(np.array([stroke_reg_m['ic'][i]-stroke_reg_sd['ic'][i], stroke_reg_m['ic'][i]+stroke_reg_sd['ic'][i]]), np.array([j,j]), linestyle='-', color='#004D43')
plt.yticks(ticks=[], labels=[])
plt.xticks(ticks=list(np.arange(start=-0.2, stop=0.21, step=0.02)), labels=list(np.arange(start=-0.2, stop=0.21, step=0.02).round(decimals=2)), fontsize=15)
    
# Terminal contact
j=0
fig = plt.subplots(nrows=1, ncols=1, figsize=(40*cm, 25*cm))
for i in range(0,len(stroke_irr_m['trialname'])):
    j+=1
    plt.scatter(stroke_irr_m['tc'][i], np.array([j]), edgecolor = '#E79419', facecolor='None', marker = 'o', s=100, label='Irregular gait') # SMK ORANGE
    plt.plot(np.array([stroke_irr_m['tc'][i]-stroke_irr_sd['tc'][i], stroke_irr_m['tc'][i]+stroke_irr_sd['tc'][i]]), np.array([j,j]), linestyle='-', color='#E79419')
for i in range(0,len(stroke_reg_m['trialname'])):
    j+=1
    plt.scatter(stroke_reg_m['tc'][i], np.array([j]), edgecolor = '#004D43', facecolor='None', marker = 'o', s=100, label='Regular gait') # SMK GREEN
    plt.plot(np.array([stroke_reg_m['tc'][i]-stroke_reg_sd['tc'][i], stroke_reg_m['tc'][i]+stroke_reg_sd['tc'][i]]), np.array([j,j]), linestyle='-', color='#004D43')
plt.yticks(ticks=[], labels=[])
plt.xticks(ticks=list(np.arange(start=-0.2, stop=0.21, step=0.02)), labels=list(np.arange(start=-0.2, stop=0.21, step=0.02).round(decimals=2)), fontsize=15)
        


# if storedfile == False:
#     # Save data
#     f = open(save_as,"wb")
#     a = {'OMCS':OMCS, 'OMCS_gait_events':OMCS_gait_events, 'OMCS_spatiotemporals':OMCS_spatiotemporals, 'IMU':IMU, 'corresponding_files':corresponding_files, 'trialnames':trialnames, 'analyze_trialtypes':analyze_trialtypes}
#     pickle.dump(a,f)
#     f.close()
#     print ('Saved file locally')
