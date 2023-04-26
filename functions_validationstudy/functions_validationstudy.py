"""
Validation study - MOTOR project
Sint Maartenskliniek study ID: 0900_Smarten_the_Clinic_V2

Author:         C.J. Ensink, c.ensink@maartenskliniek.nl
Last update:    26-10-2022

Functions for running the scripts for the validation study.
This file:
    - check events
    - check spatiotemporal parameters

"""

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt # Plots

from VICON_functions.zenieventdetection import zenieventdetection
from VICON_functions.spatiotemporalsvicon import spatiotemporalsGRAIL, spatiotemporalsGBA


def analyze_OMCS(OMCS, IMU, trialnames):
    
    VideoFrameRate = 100 # Hz
    # Prepare datastructure
    OMCS_gait_events = dict()
    OMCS_spatiotemporals = dict()
    
    # Detect gait events in vicon data
    for f in OMCS:
        if IMU[f]['trialType'] == 'GRAIL':
            # Detect gait events GRAIL vicon data
            OMCS_gait_events[f] = zenieventdetection(OMCS[f], VideoFrameRate, algorithmtype ='velocity', trialtype='treadmill')
            
            # Determine spatiotemporal parameters from vicon data
            try:
                OMCS_spatiotemporals[f] = spatiotemporalsGRAIL(OMCS[f], OMCS_gait_events[f], VideoFrameRate)
            except:
                print('Cannot calculate OMCS based spatiotemporals for trial ', f) 
            
        if IMU[f]['trialType'] == '2MWT' or IMU[f]['trialType'] == 'L-test':
            # Detect gait events overground vicon data
            OMCS_gait_events[f] = zenieventdetection(OMCS[f], VideoFrameRate, algorithmtype ='velocity', trialtype='overground')
            
            # Determine spatiotemporal parameters from vicon data
            try:
                OMCS_spatiotemporals[f] = spatiotemporalsGBA(OMCS[f], OMCS_gait_events[f], VideoFrameRate)
            except:
                print('Cannot calculate OMCS based spatiotemporals for trial ', f)
    
    return OMCS_gait_events, OMCS_spatiotemporals





def checkevents (data1, data2, window):
    # Assumed that gait events that do not have a matching event are falsely detected or missed events. These are saved in wrongdata1 and wrongdata2
    rightdata1 = np.array([], dtype = int)
    rightdata2 = np.array([], dtype = int)
    wrongdata1 = np.array([], dtype = int)
    wrongdata2 = np.array([], dtype = int)
    
    #Identify gait events within window of frames with a matching event in the other dataset
    for i in range(0, len(data1)):
        windowvaluesdata1 = np.array(range(int(data1[i])-window, int(data1[i])+window))
        
        okayvaluedata2 = data2[np.in1d(data2, windowvaluesdata1)]
        # if len(okayvaluedata2)>0:
            
        if len(okayvaluedata2) > 1:
            okayvaluedata2 = okayvaluedata2[np.argmin(data1[i]-okayvaluedata2)]
        if okayvaluedata2.size>0:
            rightdata1 = np.append(rightdata1, int(data1[i]))
            rightdata2 = np.append(rightdata2, int(okayvaluedata2))
        
    # Identify wich gait events do not have a matching event in the other dataset
    for i in range(0, len(data1)):
        if data1[i] not in rightdata1:
            wrongdata1 = np.append(wrongdata1, data1[i])
    for i in range(0, len(data2)):
        if data2[i] not in rightdata2:
            wrongdata2 = np.append(wrongdata2, data2[i])
            
    return (rightdata1), (rightdata2), (wrongdata1), (wrongdata2)





def stride_by_stride(IMU, OMCS_gait_events, OMCS_spatiotemporals):
    
    parameters = ['Initial contact', 'Terminal contact', 'Stride time (s)', 'Stride length (m)', 'Stride velocity (m/s)']
    
    # Analyze parameters only if part of a full stride
    VideoFrameRate = 100 # Hz
    # Create empty variables to be filled for gait event comparison
    BA_ICL_IMU = dict()
    BA_ICL_OMCS = dict()
    wrongICL_IMU = dict()
    missedICL_OMCS = dict()
    BA_TCL_IMU = dict()
    BA_TCL_OMCS = dict()
    wrongTCL_IMU = dict()
    missedTCL_OMCS = dict()
    BA_ICR_IMU = dict()
    BA_ICR_OMCS = dict()
    wrongICR_IMU = dict()
    missedICR_OMCS = dict()
    BA_TCR_IMU = dict()
    BA_TCR_OMCS = dict()
    wrongTCR_IMU = dict()
    missedTCR_OMCS = dict()
    
    window = int(0.2*VideoFrameRate)
    
    for f in OMCS_spatiotemporals:
        BA_ICL_IMU[f], BA_ICL_OMCS[f], wrongICL_IMU[f], missedICL_OMCS[f] = checkevents(IMU[f]['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,1], OMCS_spatiotemporals[f]['Stridelength left (mm)'][:,1], window)
        BA_TCL_IMU[f], BA_TCL_OMCS[f], wrongTCL_IMU[f], missedTCL_OMCS[f] = checkevents(IMU[f]['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0], OMCS_spatiotemporals[f]['Stridelength left (mm)'][:,0], window)
        BA_ICR_IMU[f], BA_ICR_OMCS[f], wrongICR_IMU[f], missedICR_OMCS[f] = checkevents(IMU[f]['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,1], OMCS_spatiotemporals[f]['Stridelength right (mm)'][:,1], window)
        BA_TCR_IMU[f], BA_TCR_OMCS[f], wrongTCR_IMU[f], missedTCR_OMCS[f] = checkevents(IMU[f]['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0], OMCS_spatiotemporals[f]['Stridelength right (mm)'][:,0], window)
    
    
    
    # Prepare data structure
    SbS_OMCS = dict()
    SbS_IMU = dict()
    
    for p in parameters:
        SbS_OMCS[p] = dict()
        SbS_IMU[p] = dict()
        
        for trial in OMCS_spatiotemporals:
            SbS_OMCS[p][trial] = np.array([])
            SbS_IMU[p][trial] = np.array([])
            
            if p == 'Initial contact':
                # Left
                for j in range(0,len(IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICL_IMU[trial] and IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCL_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,1])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength left (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,1] in BA_ICL_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,0] in BA_TCL_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,1])
                # Right
                for j in range(0,len(IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICR_IMU[trial] and IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCR_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,1])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength right (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,1] in BA_ICR_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,0] in BA_TCR_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,1])

                
            elif p == 'Terminal contact':
                # Left
                for j in range(0,len(IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICL_IMU[trial] and IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCL_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,0])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength left (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,1] in BA_ICL_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,0] in BA_TCL_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,0])
                # Right
                for j in range(0,len(IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICR_IMU[trial] and IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCR_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,0])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength right (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,1] in BA_ICR_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,0] in BA_TCR_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,0])

                
            elif p == 'Stride length (m)':
                # Left
                for j in range(0,len(IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICL_IMU[trial] and IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCL_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,2])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength left (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,1] in BA_ICL_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,0] in BA_TCL_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,2]/1000)
                # Right
                for j in range(0,len(IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICR_IMU[trial] and IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCR_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,2])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength right (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,1] in BA_ICR_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,0] in BA_TCR_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,2]/1000)
            
            
            elif p == 'Stride time (s)':
                # Left
                for j in range(0,len(IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICL_IMU[trial] and IMU[trial]['Left foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCL_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Left foot']['derived']['Stride time per stride - all strides (s)'][j])
                          
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength left (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,1] in BA_ICL_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,0] in BA_TCL_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Gait Cycle duration left (s)'][j,2])
                # Right          
                for j in range(0,len(IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'])):
                    if IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,1] in BA_ICR_IMU[trial] and IMU[trial]['Right foot']['derived']['Stride length - all strides (m)'][j,0] in BA_TCR_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Right foot']['derived']['Stride time per stride - all strides (s)'][j])
                         
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength right (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,1] in BA_ICR_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,0] in BA_TCR_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Gait Cycle duration right (s)'][j,2])
                       
                
            elif p == 'Stride velocity (m/s)':
                # Left
                for j in range(0,len(IMU[trial]['Left foot']['derived']['Gait speed per stride (m/s)'])):
                    if IMU[trial]['Left foot']['derived']['Gait speed per stride (m/s)'][j,1] in BA_ICL_IMU[trial] and IMU[trial]['Left foot']['derived']['Gait speed per stride (m/s)'][j,0] in BA_TCL_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Left foot']['derived']['Gait speed per stride (m/s)'][j,2])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength left (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,1] in BA_ICL_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength left (mm)'][j,0] in BA_TCL_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Gait speed left strides (m/s)'][j,2])
                # Right
                for j in range(0,len(IMU[trial]['Right foot']['derived']['Gait speed per stride (m/s)'])):
                    if IMU[trial]['Right foot']['derived']['Gait speed per stride (m/s)'][j,1] in BA_ICR_IMU[trial] and IMU[trial]['Right foot']['derived']['Gait speed per stride (m/s)'][j,0] in BA_TCR_IMU[trial]:
                        SbS_IMU[p][trial] = np.append(SbS_IMU[p][trial], IMU[trial]['Right foot']['derived']['Gait speed per stride (m/s)'][j,2])
                for j in range(0,len(OMCS_spatiotemporals[trial]['Stridelength right (mm)'])):
                    if OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,1] in BA_ICR_OMCS[trial] and OMCS_spatiotemporals[trial]['Stridelength right (mm)'][j,0] in BA_TCR_OMCS[trial]:
                        SbS_OMCS[p][trial] = np.append(SbS_OMCS[p][trial], OMCS_spatiotemporals[trial]['Gait speed right strides (m/s)'][j,2])
                    
    
    return SbS_IMU, SbS_OMCS




# Function for Bland Altman plots of gait events
def bland_altman_plot_gait_events(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    # mean      = np.mean([data1, data2], axis=0)
    mean=np.array([])                                 # Instead of mean, add all data point after eachother (so per gait event a difference between xsens and vicon becomes clear)
    for i in range(0, len(data1)):
        mean = np.append(mean, i)
        
    diff      = (data1 - data2)/100                   # Difference between data1 and data2
    md        = np.mean(diff)                         # Mean of the difference
    md_string = 'mean difference: ' + round(md, 3).astype(str)
    sd        = np.std(diff, axis=0)                  # Standard deviation of the difference
    ub_string = '+ 1.96*SD: ' + round(md + 1.96*sd, 3).astype(str)
    lb_string = '- 1.96*SD: ' + round(md - 1.96*sd, 3).astype(str)
    
    # Check for inputname in **kwargs items
    eventType = str()
    for key, value in kwargs.items():
        if key == 'eventType':
            eventType = value
    
    fig = plt.subplots()
    plt.title('Bland-Altman Plot ' + eventType)
    plt.scatter(mean, diff)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.text(max(mean), md, md_string, fontsize=10)
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.text(max(mean), md + 1.96*sd, ub_string, fontsize=10)
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.text(max(mean), md - 1.96*sd, lb_string, fontsize=10)
    plt.xlabel("Event #")
    plt.ylabel("Difference between measures (s)")
    
    return


# Function for Bland Altman plots
def bland_altman_plot_spatiotemporals_2(regular_IMU, regular_OMCS, irregular_IMU, irregular_OMCS, *args, **kwargs):
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    cm = 1/2.54
    # Regular walking
    regular_IMU     = np.asarray(regular_IMU)
    regular_OMCS     = np.asarray(regular_OMCS)
    mean_regular      = np.nanmean([regular_IMU, regular_OMCS], axis=0)
    diff_regular      = (regular_IMU - regular_OMCS)                   # Difference between data1 and data2
    md_regular        = np.nanmean(diff_regular)                         # Mean of the difference
    md_regular_string = round(md_regular, 2).astype(str) # 'mean difference: ' + 
    sd_regular        = np.nanstd(diff_regular, axis=0)                  # Standard deviation of the difference
    ub_string_regular = round(md_regular + 1.96*sd_regular, 2).astype(str) # '+ 1.96*SD: ' + 
    lb_string_regular = round(md_regular - 1.96*sd_regular, 2).astype(str) # '- 1.96*SD: ' + 
    
    if mean_regular.size > 0:
        mask_reg = ~np.isnan(mean_regular) & ~np.isnan(diff_regular)
        slope, intercept, r, p, std_err = stats.linregress(mean_regular[mask_reg], diff_regular[mask_reg])
        predictions_regular = np.transpose(np.array([slope*mean_regular[mask_reg] + intercept])).flatten()
    
    # Irregular walking
    irregular_IMU     = np.asarray(irregular_IMU)
    irregular_OMCS     = np.asarray(irregular_OMCS)
    mean_irregular      = np.nanmean([irregular_IMU, irregular_OMCS], axis=0)
    diff_irregular      = (irregular_IMU - irregular_OMCS)                   # Difference between data1 and data2
    md_irregular        = np.nanmean(diff_irregular)                         # Mean of the difference
    md_irregular_string = round(md_irregular, 2).astype(str) #'mean difference: ' + 
    sd_irregular        = np.nanstd(diff_irregular, axis=0)                  # Standard deviation of the difference
    ub_string_irregular = round(md_irregular + 1.96*sd_irregular, 2).astype(str) #'+ 1.96*SD: ' + 
    lb_string_irregular = round(md_irregular - 1.96*sd_irregular, 2).astype(str) #'- 1.96*SD: ' + 
    
    if mean_irregular.size > 0:
        mask_irreg = ~np.isnan(mean_irregular) & ~np.isnan(diff_irregular)
        slope, intercept, r, p, std_err = stats.linregress(mean_irregular[mask_irreg], diff_irregular[mask_irreg])
        predictions_irregular = np.transpose(np.array([slope*mean_irregular[mask_irreg] + intercept])).flatten()
    
    # Check for inputname in **kwargs items
    eventType = str()
    unit = str()
    for key, value in kwargs.items():
        if key == 'eventType':
            eventType = value
        if key == 'unit':
            unit = value
    
    fig = plt.subplots(figsize=(30*cm, 25*cm))
    
    # plt.title('Bland-Altman Plot   -   ' + eventType, fontsize=20)
    # plt.title(eventType, fontsize=20)
    # plt.subplots_adjust(left=0.100, right=0.740, top=0.900, bottom=0.105)
    
    # Regular walking
    plt.scatter(mean_regular, diff_regular, edgecolor = '#004D43', facecolor='None', marker = 'o', s=100, label='Regular gait') # SMK GREEN
    
    # plt.axhline(md_regular,           color='#337068', linestyle='-', linewidth = 5)
    if mean_regular.size > 0:
        plt.plot(mean_regular[mask_reg], predictions_regular, color='#337068', linestyle='-', linewidth = 5)
    # plt.text(min(mean_regular), md_regular, md_regular_string, fontsize=14)
    
    # plt.axhline(md_regular + 1.96*sd_regular, color='#337068', linestyle='--', linewidth = 5)
    # plt.text(min(mean_regular), md_regular + 1.96*sd_regular, ub_string_regular, fontsize=12)
    
    # plt.axhline(md_regular - 1.96*sd_regular, color='#337068', linestyle='--', linewidth = 5)
    # plt.text(min(mean_regular), md_regular - 1.96*sd_regular, lb_string_regular, fontsize=12)
    
    # Irregular walking
    if irregular_IMU.size > 0:
        plt.scatter(mean_irregular, diff_irregular, edgecolor='#E79419', facecolor='None', marker='^', s=100, label = 'Irregular, precision stepping, gait') # SMK ORANGE
    
        # plt.axhline(md_irregular,           color='#ECA947', linestyle='-', linewidth = 5)
        plt.plot(mean_irregular[mask_irreg], predictions_irregular, color='#ECA947', linestyle='-', linewidth = 5)
        # plt.text(max(mean_irregular)-0.05*np.nanmean(mean_irregular), md_irregular, md_irregular_string, fontsize=16)
        # plt.axhline(md_irregular + 1.96*sd_irregular, color='#ECA947', linestyle='--', linewidth = 5)
        # plt.text(max(mean_irregular)-0.02*np.nanmean(mean_irregular), md_irregular + 1.96*sd_irregular, ub_string_irregular, fontsize=16)
        # plt.axhline(md_irregular - 1.96*sd_irregular, color='#ECA947', linestyle='--', linewidth = 5)
        # plt.text(max(mean_irregular)-0.02*np.nanmean(mean_irregular), md_irregular - 1.96*sd_irregular, lb_string_irregular, fontsize=16)
        
    # plt.plot(mean_all, model, color = 'k', label = 'Linear regression')
        
    # plt.xlabel("Mean of measures "+ unit, fontsize=18)
    # plt.ylabel("Difference between measures " + unit, fontsize=18)
    
    plt.xticks(fontsize=22) #16   
    # set_xticklabels(fontsize=16)
    plt.yticks(fontsize=22) #16
    
    if 'Stride time' in eventType:
        plt.ylim(-0.35, 0.9)
    elif 'Stride length' in eventType:
        plt.ylim(-0.5, 0.6)
    elif 'Stride velocity' in eventType:
        plt.ylim(-0.35, 0.35)
    # plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)  
    
    return





# Function for Bland Altman plots
def bland_altman_plot_spatiotemporals_4(HCregular_IMU, HCregular_OMCS, HCirregular_IMU, HCirregular_OMCS, STRregular_IMU, STRregular_OMCS, STRirregular_IMU, STRirregular_OMCS, *args, **kwargs):
    cm = 1/2.54
    # HC Regular walking
    HCregular_IMU     = np.asarray(HCregular_IMU)
    HCregular_OMCS     = np.asarray(HCregular_OMCS)
    mean_HCregular      = np.nanmean([HCregular_IMU, HCregular_OMCS], axis=0)
    diff_HCregular      = (HCregular_IMU - HCregular_OMCS)                   # Difference between data1 and data2
    md_HCregular        = np.nanmean(diff_HCregular)                         # Mean of the difference
    md_HCregular_string = round(md_HCregular, 2).astype(str) # 'mean difference: ' + 
    sd_HCregular        = np.nanstd(diff_HCregular, axis=0)                  # Standard deviation of the difference
    ub_string_HCregular = round(md_HCregular + 1.96*sd_HCregular, 2).astype(str) # '+ 1.96*SD: ' + 
    lb_string_HCregular = round(md_HCregular - 1.96*sd_HCregular, 2).astype(str) # '- 1.96*SD: ' + 
    
    # HC Irregular walking
    HCirregular_IMU     = np.asarray(HCirregular_IMU)
    HCirregular_OMCS     = np.asarray(HCirregular_OMCS)
    mean_HCirregular      = np.nanmean([HCirregular_IMU, HCirregular_OMCS], axis=0)
    diff_HCirregular      = (HCirregular_IMU - HCirregular_OMCS)                   # Difference between data1 and data2
    md_HCirregular        = np.nanmean(diff_HCirregular)                         # Mean of the difference
    md_HCirregular_string = round(md_HCirregular, 2).astype(str) #'mean difference: ' + 
    sd_HCirregular        = np.nanstd(diff_HCirregular, axis=0)                  # Standard deviation of the difference
    ub_string_HCirregular = round(md_HCirregular + 1.96*sd_HCirregular, 2).astype(str) #'+ 1.96*SD: ' + 
    lb_string_HCirregular = round(md_HCirregular - 1.96*sd_HCirregular, 2).astype(str) #'- 1.96*SD: ' + 

    # STR Regular walking
    STRregular_IMU     = np.asarray(STRregular_IMU)
    STRregular_OMCS     = np.asarray(STRregular_OMCS)
    mean_STRregular      = np.nanmean([STRregular_IMU, STRregular_OMCS], axis=0)
    diff_STRregular      = (STRregular_IMU - STRregular_OMCS)                   # Difference between data1 and data2
    md_STRregular        = np.nanmean(diff_STRregular)                         # Mean of the difference
    md_STRregular_string = round(md_STRregular, 2).astype(str) # 'mean difference: ' + 
    sd_STRregular        = np.nanstd(diff_STRregular, axis=0)                  # Standard deviation of the difference
    ub_string_STRregular = round(md_STRregular + 1.96*sd_STRregular, 2).astype(str) # '+ 1.96*SD: ' + 
    lb_string_STRregular = round(md_STRregular - 1.96*sd_STRregular, 2).astype(str) # '- 1.96*SD: ' + 
    
    # STR Irregular walking
    irregular_IMU     = np.asarray(STRirregular_IMU)
    irregular_OMCS     = np.asarray(STRirregular_OMCS)
    mean_STRirregular      = np.nanmean([STRirregular_IMU, STRirregular_OMCS], axis=0)
    diff_STRirregular      = (STRirregular_IMU - STRirregular_OMCS)                   # Difference between data1 and data2
    md_STRirregular        = np.nanmean(diff_STRirregular)                         # Mean of the difference
    md_STRirregular_string = round(md_STRirregular, 2).astype(str) #'mean difference: ' + 
    sd_STRirregular        = np.nanstd(diff_STRirregular, axis=0)                  # Standard deviation of the difference
    ub_string_STRirregular = round(md_STRirregular + 1.96*sd_STRirregular, 2).astype(str) #'+ 1.96*SD: ' + 
    lb_string_STRirregular = round(md_STRirregular - 1.96*sd_STRirregular, 2).astype(str) #'- 1.96*SD: ' + 
    
    
    # Check for inputname in **kwargs items
    eventType = str()
    unit = str()
    for key, value in kwargs.items():
        if key == 'eventType':
            eventType = value
        if key == 'unit':
            unit = value
    
    fig = plt.subplots(figsize=(30*cm, 25*cm))
    # plt.title('Bland-Altman Plot   -   ' + eventType, fontsize=20)
    # plt.title(eventType, fontsize=20)
    
    # HC Regular walking
    plt.scatter(mean_HCregular, diff_HCregular, edgecolor = '#004D43', facecolor='None', marker = 'o', s=100, label='Healthy regular gait') # SMK GREEN
    
    # plt.axhline(md_HCregular,           color='#337068', linestyle='-', linewidth = 5)
    # plt.text(min(mean_HCregular), md_HCregular, md_HCregular_string, fontsize=14)
    
    # plt.axhline(md_HCregular + 1.96*sd_HCregular, color='#337068', linestyle='--', linewidth = 5)
    # plt.text(min(mean_HCregular), md_HCregular + 1.96*sd_HCregular, ub_string_HCregular, fontsize=12)
    
    # plt.axhline(md_HCregular - 1.96*sd_HCregular, color='#337068', linestyle='--', linewidth = 5)
    # plt.text(min(mean_HCregular), md_HCregular - 1.96*sd_HCregular, lb_string_HCregular, fontsize=12)
    
    # HC Irregular walking
    if HCirregular_IMU.size > 0:
        plt.scatter(mean_HCirregular, diff_HCirregular, edgecolor='#004D43', facecolor='None', marker='^', s=100, label = 'Healthy irregular gait') # SMK ORANGE
    
        # plt.axhline(md_HCirregular,           color='#ECA947', linestyle='-', linewidth = 5)
        # # plt.text(max(mean_irregular)-0.05*np.nanmean(mean_irregular), md_irregular, md_irregular_string, fontsize=16)
        # plt.axhline(md_HCirregular + 1.96*sd_HCirregular, color='#ECA947', linestyle='--', linewidth = 5)
        # # plt.text(max(mean_irregular)-0.02*np.nanmean(mean_irregular), md_irregular + 1.96*sd_irregular, ub_string_irregular, fontsize=16)
        # plt.axhline(md_HCirregular - 1.96*sd_HCirregular, color='#ECA947', linestyle='--', linewidth = 5)
        # # plt.text(max(mean_irregular)-0.02*np.nanmean(mean_irregular), md_irregular - 1.96*sd_irregular, lb_string_irregular, fontsize=16)
    
    # STR Regular walking
    plt.scatter(mean_STRregular, diff_STRregular, edgecolor = '#E79419', facecolor='None', marker = 'o', s=100, label='Stroke regular gait') # SMK GREEN
    
    # plt.axhline(md_STRregular,           color='#337068', linestyle='-', linewidth = 5)
    # plt.text(min(mean_STRregular), md_STRregular, md_STRregular_string, fontsize=14)
    
    # plt.axhline(md_STRregular + 1.96*sd_STRregular, color='#337068', linestyle='--', linewidth = 5)
    # plt.text(min(mean_STRregular), md_STRregular + 1.96*sd_STRregular, ub_string_STRregular, fontsize=12)
    
    # plt.axhline(md_STRregular - 1.96*sd_STRregular, color='#337068', linestyle='--', linewidth = 5)
    # plt.text(min(mean_STRregular), md_STRregular - 1.96*sd_STRregular, lb_string_STRregular, fontsize=12)
    
    # STR Irregular walking
    if STRirregular_IMU.size > 0:
        plt.scatter(mean_STRirregular, diff_STRirregular, edgecolor='#E79419', facecolor='None', marker='^', s=100, label = 'Stroke irregular gait') # SMK ORANGE
    
        # plt.axhline(md_STRirregular,           color='#ECA947', linestyle='-', linewidth = 5)
        # # plt.text(max(mean_irregular)-0.05*np.nanmean(mean_irregular), md_irregular, md_irregular_string, fontsize=16)
        # plt.axhline(md_STRirregular + 1.96*sd_STRirregular, color='#ECA947', linestyle='--', linewidth = 5)
        # # plt.text(max(mean_irregular)-0.02*np.nanmean(mean_irregular), md_irregular + 1.96*sd_irregular, ub_string_irregular, fontsize=16)
        # plt.axhline(md_STRirregular - 1.96*sd_STRirregular, color='#ECA947', linestyle='--', linewidth = 5)
        # # plt.text(max(mean_irregular)-0.02*np.nanmean(mean_irregular), md_irregular - 1.96*sd_irregular, lb_string_irregular, fontsize=16)
        
    # plt.xlabel("Mean of measures "+ unit, fontsize=18)
    # plt.ylabel("Difference between measures " + unit, fontsize=18)
    
    plt.xticks(fontsize=22) #16   
    # set_xticklabels(fontsize=16)
    plt.yticks(fontsize=22) #16
    
    # plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)  
    
    return




def histogram_gait_events(IMU_regular, OMCS_regular, IMU_irregular, OMCS_irregular, **kwargs):
    cm = 1/2.54
    width = 0.01 # the width of the bars 
    
    # Regular gait
    diff_regular = np.round((IMU_regular/100 - OMCS_regular/100),2)      # Difference between data1 and data2
    
    count = 0
    for j in range(len(diff_regular)):
        if np.abs(diff_regular[j])>0.1:
            count+=1
    
    md_regular        = np.nanmean(diff_regular)                         # Mean of the difference
    sd_regular        = np.nanstd(diff_regular, axis=0)                  # Standard deviation of the difference
    
    fig, ax = plt.subplots(figsize=(40*cm, 25*cm))    
    
    diffvalues_regular = np.unique(diff_regular)
    y = np.zeros((1,len(diffvalues_regular)))
    for i in range(0,len(diffvalues_regular)):
        y[0,i] = int(np.count_nonzero(diff_regular == diffvalues_regular[i]))
    height_regular = y[0,:]
    ax.bar(diffvalues_regular, height_regular, width, color='#004D43', alpha=0.8, label = 'Normal gait') # SMK GREEN

    ax.plot([md_regular, md_regular], [0, np.max(height_regular)+0.1*max(height_regular)], color='#3F897E', label = 'mean', linewidth = 4) # Mean difference line
    ax.plot([md_regular-1.96*sd_regular, md_regular-1.96*sd_regular], [0, np.max(height_regular)+0.1*max(height_regular)], color='#3F897E', linestyle = 'dashed', linewidth = 4, label = 'mean+-1.96sd')# Lower bound 1.96*sd
    ax.plot([md_regular+1.96*sd_regular, md_regular+1.96*sd_regular], [0, np.max(height_regular)+0.1*max(height_regular)], color='#3F897E', linestyle = 'dashed', linewidth = 4)# Upper bound 1.96*sd
    
    # ax.set_xticks(diffvalues_regular)
    # ax.set_xticklabels(labels = diffvalues_regular, fontsize=15) 
    # # ax.set_yticks(np.sort(height_regular[height_regular>50]))
    # ax.set_yticklabels(labels=np.sort(height_regular[height_regular>50]).astype(int), fontsize=15, minor=False) #16
    for key, value in kwargs.items():
        if key == 'label':
            labelname = value
            print(labelname)
        else:
            labelname = 'Gait event'
    
    # Irregular gait (stepping stones)
    if IMU_irregular.size > 0:
        diff_irregular     = np.round((IMU_irregular/100 - OMCS_irregular/100),2)                       # Difference between data1 and data2
        count=0
        for i in range(len(diff_irregular)):
            if np.abs(diff_irregular[i])>0.1:
                count+=1
        md_irregular = np.mean(diff_irregular)                         # Mean of the difference
        sd_irregular = np.std(diff_irregular, axis=0)                  # Standard deviation of the difference
        

        diffvalues_irregular = np.unique(diff_irregular)
        y_irregular = np.zeros((1,len(diffvalues_irregular)))
        for i in range(0,len(diffvalues_irregular)):
            y_irregular[0,i] = int(np.count_nonzero(diff_irregular == diffvalues_irregular[i]))
        height_irregular = y_irregular[0,:]
        ax.bar(diffvalues_irregular, height_irregular, width, color='#E79419', alpha=0.8, label='Irregular gait') # SMK ORANGE
        
        ax.plot([md_irregular,md_irregular], [0,np.max(height_regular)+0.1*max(height_regular)], color='#ECA947', label = 'mean', linewidth = 4) # Mean difference line
        ax.plot([md_irregular-1.96*sd_irregular, md_irregular-1.96*sd_irregular], [0,np.max(height_regular)+0.1*max(height_regular)], color='#ECA947', linestyle = 'dashed', linewidth = 4, label = 'mean+-1.96sd')# Lower bound 1.96*sd
        ax.plot([md_irregular+1.96*sd_irregular, md_irregular+1.96*sd_irregular], [0,np.max(height_regular)+0.1*max(height_regular)], color='#ECA947', linestyle = 'dashed', linewidth = 4)# Upper bound 1.96*sd
    
    ax.set_xticks(np.arange(start=-0.2, stop=0.21, step=0.02))
    l = np.arange(start=-0.2, stop=0.21, step=0.02)
    l=l.round(decimals=2)
    ax.set_xticklabels(labels = l, fontsize=16) 
    # ax.set_xticks(diffvalues_regular[np.argwhere(diffvalues_regular*100 % 2 == 0)].flatten()) #
    # ax.set_xticklabels(labels = diffvalues_regular[np.argwhere(diffvalues_regular*100 % 2 == 0)].flatten(), fontsize=15) 
    
    if 'Healthy' in labelname:
        stopval = 4501
        stepval = 250
    elif 'Stroke' in labelname:
        stopval = 601
        stepval = 100
    else:
        stopval=(max(height_regular)+0.1*max(height_regular))
        if max(height_regular) > 1000:
            stepval = 250
        elif max(height_regular) > 500:
            stepval = 100
        else:
            stepval = 50
    lbls = np.arange(start=0, stop=stopval, step=stepval)
    ax.set_yticks(lbls)
    ax.set_yticklabels(labels=lbls.astype(int), fontsize=16, minor=False) #16
            
    # plt.title(label=labelname, fontsize=20)
    # plt.xlabel('Difference between measures (s)', fontsize=18)
    # plt.ylabel('Count (#)', fontsize=18) 
    # plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    
    plt.show()
    
    return





def pearson_correlation(IMU, OMCS):
    from scipy import stats # Pearson correlation
    nans = np.logical_or(np.isnan(IMU), np.isnan(OMCS))
    IMU = IMU[~nans]
    OMCS = OMCS[~nans]
    
    pearson_rho, pearson_pval = stats.pearsonr(IMU, OMCS)
    pearson_rho = round(pearson_rho, 2)
    pearson_pval = round(pearson_pval, 2)
    
    return pearson_rho, pearson_pval



def post_hoc_forcedata(OMCS, OMCS_spatiotemporals, IMU, files):
    # Check which walking trials have focedata (vertical direction)
    forcedata = dict()
    for trial in files['GRAIL']:
        try:
            if len(OMCS[trial]['Analog data']['Force.Fz1'])>0:
                forcedata[trial] = dict()
                forcedata[trial]['Fz1'] = OMCS[trial]['Analog data']['Force.Fz1']
                forcedata[trial]['Fz2'] = OMCS[trial]['Analog data']['Force.Fz2']
        except KeyError:
                pass
    
    
    # Downsample and lowpass filter forceplate data according to: Roberts (2019); https://doi.org/10.1016/j.medengphy.2019.02.012
    # Raw forceplate data was down-sampled to 100 Hz and filtered
    # using a fourth-order, zero phase-shift, low-pass Butterworth filter
    # with a cut-off frequency of 5 Hz [15,16].
    from scipy import signal
    filtered_forcedata = dict()
    
    # Filter design
    fs_forceplates = 100 # sample frequency of the forceplates after downsampling
    fc = 20  # Cut-off frequency of the filter
    w = fc / (fs_forceplates / 2) # Normalize the frequency
    N = 4 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, w, filter_type)

    # Downsample and apply filter on data
    for trial in forcedata:
        filtered_forcedata[trial] = dict()
        nsamps_old = len(forcedata[trial]['Fz1'])
        fs_old = 1000
        t = nsamps_old/fs_old
        nsamps_new = int(t*fs_forceplates)
        
        filtered_forcedata[trial]['Fz1 sampled'] = signal.resample(forcedata[trial]['Fz1'], nsamps_new)
        filtered_forcedata[trial]['Fz2 sampled'] = signal.resample(forcedata[trial]['Fz2'], nsamps_new)

        filtered_forcedata[trial]['Fz left filtered'] = signal.filtfilt(b, a, filtered_forcedata[trial]['Fz1 sampled'])
        filtered_forcedata[trial]['Fz right filtered'] = signal.filtfilt(b, a, filtered_forcedata[trial]['Fz2 sampled'])

    # Gait event detection based on GRF data
    # IC: GRF < -10 Newton for a width of at least 0.40 seconds
    # TC: GRC > -10 Newon for a width of at least  0.40 seconds
    threshold = -10
    widththres = int(0.40*fs_forceplates)

    gait_events_forceplate = dict()
    gait_events_forceplate['IC left'] = dict()
    gait_events_forceplate['IC right'] = dict()
    gait_events_forceplate['TC left'] = dict()
    gait_events_forceplate['TC right'] = dict()

    for trial in filtered_forcedata:
        gait_events_forceplate['IC left'][trial] = np.array([], dtype=int)
        gait_events_forceplate['IC right'][trial] = np.array([], dtype=int)
        gait_events_forceplate['TC left'][trial] = np.array([], dtype=int)
        gait_events_forceplate['TC right'][trial] = np.array([], dtype=int)
        
        # Left IC and TC events
        above_thres = np.argwhere(filtered_forcedata[trial]['Fz left filtered'] > threshold).flatten()
        ic_temp = above_thres[np.argwhere(np.diff(above_thres)>widththres).flatten()]
        tc_temp = above_thres[np.argwhere(np.diff(above_thres)>widththres).flatten()+1]
        try:
            for i in range(0, len(ic_temp)):
                if np.all(filtered_forcedata[trial]['Fz left filtered'][ic_temp[i]+1 : ic_temp[i]+1+widththres] < threshold) and np.all(filtered_forcedata[trial]['Fz left filtered'][ic_temp[i]-1-widththres : ic_temp[i]-1] > threshold):
                    gait_events_forceplate['IC left'][trial] = np.append(gait_events_forceplate['IC left'][trial], int(ic_temp[i]))
        except IndexError:
            pass
        try:
            for i in range(0, len(tc_temp)):
                if np.all(filtered_forcedata[trial]['Fz left filtered'][tc_temp[i]+1 : tc_temp[i]+1+widththres] > threshold) and np.all(filtered_forcedata[trial]['Fz left filtered'][tc_temp[i]-1-widththres : tc_temp[i]-1] < threshold):
                    gait_events_forceplate['TC left'][trial] = np.append(gait_events_forceplate['TC left'][trial], int(tc_temp[i]))
        except IndexError:
            pass        

        # Right IC and TC events
        above_thres = np.argwhere(filtered_forcedata[trial]['Fz right filtered'] > threshold).flatten()
        ic_temp = above_thres[np.argwhere(np.diff(above_thres)>widththres).flatten()]
        tc_temp = above_thres[np.argwhere(np.diff(above_thres)>widththres).flatten()+1]
        try:
            for i in range(0, len(ic_temp)):
                if np.all(filtered_forcedata[trial]['Fz right filtered'][ic_temp[i]+1 : ic_temp[i]+1+widththres] < threshold) and np.all(filtered_forcedata[trial]['Fz right filtered'][ic_temp[i]-1-widththres : ic_temp[i]-1] > threshold):
                    gait_events_forceplate['IC right'][trial] = np.append(gait_events_forceplate['IC right'][trial], int(ic_temp[i]))
        except IndexError:
            pass
        try:
            for i in range(0, len(tc_temp)):
                if np.all(filtered_forcedata[trial]['Fz right filtered'][tc_temp[i]+1 : tc_temp[i]+1+widththres] > threshold) and np.all(filtered_forcedata[trial]['Fz right filtered'][tc_temp[i]-1-widththres : tc_temp[i]-1] < threshold):
                    gait_events_forceplate['TC right'][trial] = np.append(gait_events_forceplate['TC right'][trial], int(tc_temp[i]))
        except IndexError:
            pass
        
    # 900_CVA_04_SP01.c3d >> walking mostly on one of the treadmill bands, not viable for gait event detection.
    gait_events_forceplate['IC left']['900_CVA_04_SP01.c3d'] = np.array([], dtype=int)
    gait_events_forceplate['TC left']['900_CVA_04_SP01.c3d'] = np.array([], dtype=int)
    gait_events_forceplate['IC right']['900_CVA_04_SP01.c3d'] = np.array([], dtype=int)
    gait_events_forceplate['TC right']['900_CVA_04_SP01.c3d'] = np.array([], dtype=int)


    # for trial in files['GRAIL stroke regular']:
    #     fig = plt.figure()
    #     plt.title(label = trial)
    #     plt.plot(filtered_forcedata[trial]['Fz left filtered'], 'black', label='Left')
    #     plt.plot(filtered_forcedata[trial]['Fz right filtered'], 'orange', label='Right')
    #     # plt.plot(OMCS[trial]['LHEE'][:,2], 'black', linestyle = 'dashed', label = 'left')
    #     # plt.plot(OMCS[trial]['RHEE'][:,2], 'orange', linestyle = 'dashed', label = 'right')
    #     plt.plot(gait_events_forceplate['IC left'][trial], filtered_forcedata[trial]['Fz left filtered'][gait_events_forceplate['IC left'][trial]], 'gv', label='IC')
    #     plt.plot(gait_events_forceplate['IC right'][trial], filtered_forcedata[trial]['Fz right filtered'][gait_events_forceplate['IC right'][trial]], 'gv')
    #     plt.plot(gait_events_forceplate['TC left'][trial], filtered_forcedata[trial]['Fz left filtered'][gait_events_forceplate['TC left'][trial]], 'r^', label='TC')
    #     plt.plot(gait_events_forceplate['TC right'][trial], filtered_forcedata[trial]['Fz right filtered'][gait_events_forceplate['TC right'][trial]], 'r^')


    # Compare only gait events that are part of a full stride
    comparable_gait_events_forceplate = dict()
    window = int(0.2*fs_forceplates)
    comparable_gait_events_forceplate['IC left force with OMCS'] = dict()
    comparable_gait_events_forceplate['IC left force with IMU'] = dict()
    comparable_gait_events_forceplate['IC left OMCS with force'] = dict()
    comparable_gait_events_forceplate['IC left IMU with force'] = dict()
    # comparable_gait_events_forceplate['IC left force wrong'] = dict()
    # comparable_gait_events_forceplate['IC left vicon missed'] = dict()
    # comparable_gait_events_forceplate['IC left xsens missed'] = dict()
    comparable_gait_events_forceplate['TC left force with OMCS'] = dict()
    comparable_gait_events_forceplate['TC left force with IMU'] = dict()
    comparable_gait_events_forceplate['TC left OMCS with force'] = dict()
    comparable_gait_events_forceplate['TC left IMU with force'] = dict()
    # comparable_gait_events_forceplate['TC left force wrong'] = dict()
    # comparable_gait_events_forceplate['TC left vicon missed'] = dict()
    # comparable_gait_events_forceplate['TC left xsens missed'] = dict()
    comparable_gait_events_forceplate['IC right force with OMCS'] = dict()
    comparable_gait_events_forceplate['IC right force with IMU'] = dict()
    comparable_gait_events_forceplate['IC right OMCS with force'] = dict()
    comparable_gait_events_forceplate['IC right IMU with force'] = dict()
    # comparable_gait_events_forceplate['IC right force wrong'] = dict()
    # comparable_gait_events_forceplate['IC right vicon missed'] = dict()
    # comparable_gait_events_forceplate['IC right xsens missed'] = dict()
    comparable_gait_events_forceplate['TC right force with OMCS'] = dict()
    comparable_gait_events_forceplate['TC right force with IMU'] = dict()
    comparable_gait_events_forceplate['TC right OMCS with force'] = dict()
    comparable_gait_events_forceplate['TC right IMU with force'] = dict()
    # comparable_gait_events_forceplate['TC right force wrong'] = dict()
    # comparable_gait_events_forceplate['TC right vicon missed'] = dict()
    # comparable_gait_events_forceplate['TC right xsens missed'] = dict()

    for f in gait_events_forceplate['IC left']:
        comparable_gait_events_forceplate['IC left force with OMCS'][f] = np.array([])
        comparable_gait_events_forceplate['IC left force with IMU'][f] = np.array([])
        comparable_gait_events_forceplate['IC left OMCS with force'][f] = np.array([])
        comparable_gait_events_forceplate['IC left IMU with force'][f] = np.array([])
        # comparable_gait_events_forceplate['IC left force wrong'][f] = np.array([])
        # comparable_gait_events_forceplate['IC left vicon missed'][f] = np.array([])
        # comparable_gait_events_forceplate['IC left xsens missed'][f] = np.array([])
        comparable_gait_events_forceplate['TC left force with OMCS'][f] = np.array([])
        comparable_gait_events_forceplate['TC left force with IMU'][f] = np.array([])
        comparable_gait_events_forceplate['TC left OMCS with force'][f] = np.array([])
        comparable_gait_events_forceplate['TC left IMU with force'][f] = np.array([])
        # comparable_gait_events_forceplate['TC left force wrong'][f] = np.array([])
        # comparable_gait_events_forceplate['TC left vicon missed'][f] = np.array([])
        # comparable_gait_events_forceplate['TC left xsens missed'][f] = np.array([])
        comparable_gait_events_forceplate['IC right force with OMCS'][f] = np.array([])
        comparable_gait_events_forceplate['IC right force with IMU'][f] = np.array([])
        comparable_gait_events_forceplate['IC right OMCS with force'][f] = np.array([])
        comparable_gait_events_forceplate['IC right IMU with force'][f] = np.array([])
        # comparable_gait_events_forceplate['IC right force wrong'][f] = np.array([])
        # comparable_gait_events_forceplate['IC right vicon missed'][f] = np.array([])
        # comparable_gait_events_forceplate['IC right xsens missed'][f] = np.array([])
        comparable_gait_events_forceplate['TC right force with OMCS'][f] = np.array([])
        comparable_gait_events_forceplate['TC right force with IMU'][f] = np.array([])
        comparable_gait_events_forceplate['TC right OMCS with force'][f] = np.array([])
        comparable_gait_events_forceplate['TC right IMU with force'][f] = np.array([])
        # comparable_gait_events_forceplate['TC right force wrong'][f] = np.array([])
        # comparable_gait_events_forceplate['TC right vicon missed'][f] = np.array([])
        # comparable_gait_events_forceplate['TC right xsens missed'][f] = np.array([])
        try:
            comparable_gait_events_forceplate['IC left force with OMCS'][f], comparable_gait_events_forceplate['IC left OMCS with force'][f], a, b = checkevents(OMCS_spatiotemporals[f]['Stridelength left (mm)'][:,1], gait_events_forceplate['IC left'][f].astype(int), window) #Stride length - all strides (m)
            comparable_gait_events_forceplate['TC left force with OMCS'][f], comparable_gait_events_forceplate['TC left OMCS with force'][f], a, b = checkevents(OMCS_spatiotemporals[f]['Stridelength left (mm)'][:,0], gait_events_forceplate['TC left'][f].astype(int), window)
            comparable_gait_events_forceplate['IC right force with OMCS'][f], comparable_gait_events_forceplate['IC right OMCS with force'][f], a, b = checkevents(OMCS_spatiotemporals[f]['Stridelength right (mm)'][:,1], gait_events_forceplate['IC right'][f].astype(int), window)
            comparable_gait_events_forceplate['TC right force with OMCS'][f], comparable_gait_events_forceplate['TC right OMCS with force'][f], a, b = checkevents(OMCS_spatiotemporals[f]['Stridelength right (mm)'][:,0], gait_events_forceplate['TC right'][f].astype(int), window)
            comparable_gait_events_forceplate['IC left force with IMU'][f], comparable_gait_events_forceplate['IC left IMU with force'][f], a, b = checkevents(IMU[f]['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,1], gait_events_forceplate['IC left'][f].astype(int), window) #Stride length - all strides (m)
            comparable_gait_events_forceplate['TC left force with IMU'][f], comparable_gait_events_forceplate['TC left IMU with force'][f], a, b = checkevents(IMU[f]['Left foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0], gait_events_forceplate['TC left'][f].astype(int), window)
            comparable_gait_events_forceplate['IC right force with IMU'][f], comparable_gait_events_forceplate['IC right IMU with force'][f], a, b = checkevents(IMU[f]['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,1], gait_events_forceplate['IC right'][f].astype(int), window)
            comparable_gait_events_forceplate['TC right force with IMU'][f], comparable_gait_events_forceplate['TC right IMU with force'][f], a, b = checkevents(IMU[f]['Right foot']['derived']['Stride length - no 2 steps around turn (m)'][:,0], gait_events_forceplate['TC right'][f].astype(int), window)
        except:
            pass
    
    return comparable_gait_events_forceplate 