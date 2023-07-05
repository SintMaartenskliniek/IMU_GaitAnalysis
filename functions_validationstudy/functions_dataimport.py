"""
Validation study - MOTOR project
Sint Maartenskliniek study ID: 0900_Smarten_the_Clinic_V2

Author:         C.J. Ensink, c.ensink@maartenskliniek.nl
Last update:    04-07-2023

Functions for running the scripts for the validation study.
This file:
    - dataimport

"""
# Import dependencies
import numpy as np
import os # Scan directories
import samplerate

# Import dependencies to analyze gait data
import gaittool.feet_processor.processor as feet
from gaittool.helpers.preprocessor import data_filelist, data_preprocessor

from VICON_functions.readmarkerdata import readmarkerdata


def dataimport(datafolder, trialtype):
    showfigure = 'hide'
        
    # Prepare datastructure
    vicon = dict()
    xsens = dict()
    errors = dict()
       
    # Set subfolder for xsens data
    subfolderxsens = 'Xsens/exported'
    
    # Define if vicon data is from GRAIL (../GRAIL/..) or overground lab (../GBA/..) trials
    subfolderviconGRAIL = 'Vicon/GRAIL'
    subfolderviconGBA = 'Vicon/GBA'
    
    # Define xsens trialnumber with corresponding vicon measurement
    corresponding_files = dict()
    # All files
    files = dict()
    
    # HEALTHY GRAIL TRIALS
    if trialtype['Healthy GRAIL'] == True:
        subfolder = '/Healthy_controls'
        mainpath = datafolder + subfolder
        dirnames = os.listdir(mainpath)
        dirnames = [item for item in dirnames if item.startswith('900_V')]
        ppfolders = []
        ppfoldersvicon = []
        ppfoldersxsens = []
        for i in range(0, len(dirnames)):
            ppfolders.append(mainpath + '/' + dirnames[i])
        for i in range(0, len(ppfolders)):
            # date = os.listdir(ppfolders[i])
            ppfoldersvicon.append(ppfolders[i] + '/' + subfolderviconGRAIL) # + '/' + date[0]
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) # + '/' + date[0]
            
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            with os.scandir(ppfoldersvicon[i]) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_file():
                        files[entry.name] = (ppfoldersvicon[i] + '/' + entry.name)
                        
                        # Define xsens exports
                        if entry.name == '900_V_pp01_SP01.c3d':
                            xsensnum[entry.name] = '005'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_01' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp01_FS_SS01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp03_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp03_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp04_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp04_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp05_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp05_SS01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp06_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp06_FS_SS01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp07_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp07_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp08_SP02.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp08_FS_SS02.c3d':
                            xsensnum[entry.name] = '013'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp09_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp09_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp10_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp10_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp11_SP01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp11_FS_SS01.c3d':
                            xsensnum[entry.name] = '013'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp12_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp12_FS_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                                        
                        elif entry.name == '900_V_pp13_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp13_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp14_SP01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp14_FS_SS01.c3d':
                            xsensnum[entry.name] = '012'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp15_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp15_FS_SS01.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name] 
                        
                        elif entry.name == '900_V_pp16_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp16_FS_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp18_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp18_FS_SS02.c3d':
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_V_pp19_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp19_FS_SS01.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp20_SP01.c3d':
                            xsensnum[entry.name] = '009'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp20_FS_SS01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp21_SP01.c3d':
                            xsensnum[entry.name] = '011'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp21_FS_SS01.c3d':
                            xsensnum[entry.name] = '013'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_V_pp22_SP01.c3d':
                            xsensnum[entry.name] = '008'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_V_pp22_FS_SS01.c3d':
                            xsensnum[entry.name] = '010'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
        
        corresponding_files['Healthy GRAIL'] = dict()
        corresponding_files['Healthy GRAIL']['xsensnum'] = xsensnum
        corresponding_files['Healthy GRAIL']['xsensfilepaths'] = xsensfilepaths
        
    # CVA GRAIL TRIALS
    if trialtype['CVA GRAIL'] == True:
        subfolder = '/CVA'
        mainpath = datafolder + subfolder
        dirnames = os.listdir(mainpath)
        dirnames = [item for item in dirnames if item.startswith('900_CVA')]
        ppfolders = []
        ppfoldersvicon = []
        ppfoldersxsens = []
        for i in range(0, len(dirnames)):
            ppfolders.append(mainpath + '/' + dirnames[i])
        for i in range(0, len(ppfolders)):
            # date = os.listdir(ppfolders[i])
            ppfoldersvicon.append(ppfolders[i] + '/' + subfolderviconGRAIL) # + '/' + date[0]
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) # + '/' + date[0]
            
        # files=dict()
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            with os.scandir(ppfoldersvicon[i]) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_file():
                        files[entry.name] = (ppfoldersvicon[i]+'/'+entry.name)
                        
                        # Define xsens exports
                        if entry.name == '900_CVA_01_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_01' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_01_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_01' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_pp02_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_02' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_pp02_FS_SS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_02' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_03_FS01.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_03' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_03_FS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_03' in item][0]+xsensnum[entry.name]
                    
                        elif entry.name == '900_CVA_04_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_04' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_04_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_04' in item][0]+xsensnum[entry.name]
                            
                        elif entry.name == '900_CVA_05_SP01.c3d':
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_05' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_05_FS_SS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_05' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_06_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_06' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_06_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_06' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_07_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_07' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_07_FS_SS02.c3d': # Fixed speed
                            xsensnum[entry.name] = '004'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_07' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_08_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_08' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_08_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_08' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_09_FS01.c3d':
                            xsensnum[entry.name] = '002'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_09' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_09_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '003'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_09' in item][0]+xsensnum[entry.name]
                        
                        elif entry.name == '900_CVA_10_SP01.c3d':
                            xsensnum[entry.name] = '000'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_10' in item][0]+xsensnum[entry.name]
                        elif entry.name == '900_CVA_10_FS_SS01.c3d': # Fixed speed
                            xsensnum[entry.name] = '001'
                            xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_CVA_10' in item][0]+xsensnum[entry.name]
        corresponding_files['CVA GRAIL'] = dict()
        corresponding_files['CVA GRAIL']['xsensnum'] = xsensnum
        corresponding_files['CVA GRAIL']['xsensfilepaths'] = xsensfilepaths
    
    # HEALTHY LAB TRIALS
    if trialtype['Healthy Lab'] == True:
        subfolder = '/Healthy_controls'
        mainpath = datafolder + subfolder
        dirnames = os.listdir(mainpath)
        dirnames = [item for item in dirnames if item.startswith('900_V')]
        ppfolders = []
        ppfoldersvicon = []
        ppfoldersxsens = []
        for i in range(0, len(dirnames)):
            ppfolders.append(mainpath + '/' + dirnames[i])
        for i in range(0, len(ppfolders)):
            # date = os.listdir(ppfolders[i])
            ppfoldersvicon.append(ppfolders[i] + '/' + subfolderviconGBA) #+ '/' + date[0] 
            ppfoldersxsens.append(ppfolders[i] + '/' + subfolderxsens) #+ '/' + date[0]
            
        # files=dict()
        xsensnum = dict()
        xsensfilepaths = dict()
        for i in range(0, len(ppfoldersvicon)):
            try:
                with os.scandir(ppfoldersvicon[i]) as it:        
                    for entry in it:
                        if not entry.name.startswith('.') and entry.is_file():
                            files[entry.name] = (ppfoldersvicon[i]+'/'+entry.name)
                            
                            # Define xsens exports
                            if entry.name == '900_V_pp01_2MWT01.c3d':
                                xsensnum[entry.name] = '004'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_01' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp03_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_03' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp04_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_04' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp05_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_05' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp06_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_06' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp07_SW03.c3d':
                                xsensnum[entry.name] = '003'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_07' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp08_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_08' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp09_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_09' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp10_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_10' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp11_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_11' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp12_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_12' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp13_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_13' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp14_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_14' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp15_SW01.c3d':
                                xsensnum[entry.name] = '006'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_15' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp16_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_16' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp18_SW01.c3d':
                                xsensnum[entry.name] = '005'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_18' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp19_SW01.c3d':
                                xsensnum[entry.name] = '004'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_19' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp20_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_20' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp21_SW01.c3d':
                                xsensnum[entry.name] = '002'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_21' in item][0]+xsensnum[entry.name]
                            
                            elif entry.name == '900_V_pp22_SW01.c3d':
                                xsensnum[entry.name] = '001'
                                xsensfilepaths[entry.name] = [item for item in ppfoldersxsens if '900_V_22' in item][0]+xsensnum[entry.name]
                           
            except FileNotFoundError:
                xsensnum[ppfolders[i]] = 'Unavailable'
                xsensfilepaths[ppfolders[i]] = 'Unavailable'
        
        corresponding_files['Healthy Lab'] = dict()
        corresponding_files['Healthy Lab']['xsensnum'] = xsensnum
        corresponding_files['Healthy Lab']['xsensfilepaths'] = xsensfilepaths
    
    # 3 sets
    if trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths']}
    # 2 sets
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == False:
        xsensfilepaths = {**corresponding_files['Healthy GRAIL']['xsensfilepaths'], **corresponding_files['Healthy Lab']['xsensfilepaths']}
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == True:
        xsensfilepaths = {**corresponding_files['Healthy Lab']['xsensfilepaths'], **corresponding_files['CVA GRAIL']['xsensfilepaths']}
    # 1 set
    elif trialtype['Healthy GRAIL'] == True and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == False:
        xsensfilepaths = corresponding_files['Healthy GRAIL']['xsensfilepaths']
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == True and trialtype['CVA GRAIL'] == False:
        xsensfilepaths = corresponding_files['Healthy Lab']['xsensfilepaths']
    elif trialtype['Healthy GRAIL'] == False and trialtype['Healthy Lab'] == False and trialtype['CVA GRAIL'] == True:
        xsensfilepaths = corresponding_files['CVA GRAIL']['xsensfilepaths']
    
    # Sort files on task
    filesGRAIL = dict()
    filesREG = dict()
    filesIRREG = dict()
    filesGBA = dict()
    filesSW = dict() # Straight ahead walking in measurement volume

    removekeys=[]
    for key in files:
        if ('V_pp02' or 'V_pp17' or '900_V_pp11_LT03') in key: # exclusion of these test persons
            removekeys.append(key)
    for key in removekeys:
        files.pop(key)

    for key in xsensfilepaths:
        # GRAIL trials
        if '_FS0' in key:
            filesREG[key] = files[key]
            filesGRAIL[key] = files[key]
        if '_SP0' in key:
            if key == '900_V_pp01_SP03.c3d': # Fixed speed trial, accidentily wrongly named
                pass
            else:
                filesREG[key] = files[key]
                filesGRAIL[key] = files[key]
        if '_SS' in key:
            filesIRREG[key] = files[key]
            filesGRAIL[key] = files[key]
        # Overground trials
        if '_SW' in key:
            filesSW[key] = files[key]
            filesGBA[key] = files[key]
        if '_2MWT' in key:
            filesSW[key] = files[key]
            filesGBA[key] = files[key]
    
    # Set trialnames to be analyzed
    trialnames = list()
    if trialtype['Healthy GRAIL'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '_V_' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['CVA GRAIL'] == True:
        trialnames.extend( [string for string in list(filesGRAIL.keys()) if '_CVA_' in string] )
        # trialnames.extend(list(filesGRAIL.keys()))
    if trialtype['Healthy Lab'] == True:
        # trialnames.extend( [string for string in list(filesGBA.keys()) if '_V_' in string] )
        trialnames.extend(list(filesSW.keys()))
    
    trialnames = list(set(trialnames))
    trialnames.remove('900_CVA_03_FS02.c3d') # This person performed 2 regular walking trials, remove one for further analysis
        
    # Read markerdata vicon        
    for i in range(0,len(trialnames)):
        print('Start vicon import of trial: ', trialnames[i], ' (',i,'/',len(trialnames),')')
        datavicon, VideoFrameRate, analogdata = readmarkerdata( files[trialnames[i]], analogdata=True ) #ParameterGroup, 
    
        # Check the markernames
        dataviconfilt = {}
        for key in datavicon:
            if 'LHEE' in key:
                dataviconfilt['LHEE'] = datavicon[key]
            elif 'LTOE' in key:
                dataviconfilt['LTOE'] = datavicon[key]
            elif 'RHEE' in key:
                dataviconfilt['RHEE'] = datavicon[key]  
            elif 'RTOE' in key:
                dataviconfilt['RTOE'] = datavicon[key]
            elif 'LASI' in key:
                dataviconfilt['LASI'] = datavicon[key]
            elif 'RASI' in key:
                dataviconfilt['RASI'] = datavicon[key]
            elif 'LPSI' in key:
                dataviconfilt['LPSI'] = datavicon[key]
            elif 'RPSI' in key:
                dataviconfilt['RPSI'] = datavicon[key]
            elif 'LANK' in key:
                dataviconfilt['LANK'] = datavicon[key]
            elif 'RANK' in key:
                dataviconfilt['RANK'] = datavicon[key]
        
        # Two trials with some part 'flickering' markers; set these time periods to missing markerdata
        # if trialnames[i] == '900_V_pp12_FS01.c3d': # no data labeling (bad dataquality)
        #     for key in dataviconfilt:
        #         dataviconfilt[key][5522:5651,:] = 0
        if trialnames[i] == '900_V_pp21_FS_SS01.c3d': # no data labeling (bad dataquality)
            for key in dataviconfilt:
                dataviconfilt[key][10800:10855,:] = 0
        
        # Interpolate missing values
        if trialnames[i] == '900_V_pp08_SP02.c3d': # Gap fill (3 x 1 sample)
            for key in dataviconfilt:
                missingvalues = np.unique(np.where(dataviconfilt[key] == 0)[0])
                nonmissingvalues = (np.where(dataviconfilt[key] != 0)[0])
                dataviconfilt[key][missingvalues,0] = np.interp(missingvalues, nonmissingvalues, dataviconfilt[key][nonmissingvalues,0])
                dataviconfilt[key][missingvalues,1] = np.interp(missingvalues, nonmissingvalues, dataviconfilt[key][nonmissingvalues,1])
                dataviconfilt[key][missingvalues,2] = np.interp(missingvalues, nonmissingvalues, dataviconfilt[key][nonmissingvalues,2])
        
        dataviconfilt['Analog data'] = analogdata
        
        vicon[trialnames[i]] = dataviconfilt
                
    # Analyze xsens data
    for i in range(0,len(trialnames)):
        print('Start xsens import of trial: ', trialnames[i], ' (',i,'/',len(trialnames),')')
        filepaths, sensortype, fs = data_filelist(xsensfilepaths[trialnames[i]])
        if len(filepaths) > 0:
            # Define data dictionary with all sensordata
            data_dict = data_preprocessor(filepaths, sensortype)
            
            # Determine trialType based on foldername or kwargs item
            if 'L-test' in xsensfilepaths[trialnames[i]]:
                data_dict['trialType'] = 'L-test'
            elif '2-minuten looptest' in xsensfilepaths[trialnames[i]]:
                data_dict['trialType'] = '2MWT'
            elif trialnames[i] in filesSW.keys():
                data_dict['trialType'] = '2MWT'
            else:
                data_dict['trialType'] = 'GRAIL'
                    
            if '900_V_pp15' in trialnames[i]:
                data_dict['L'] = data_dict['Right foot']
                data_dict['Right foot'] = data_dict['Left foot']
                data_dict['Left foot'] = data_dict['L']
            
            # 900_V_pp01 data collected at 40 Hz sample frequency, correct for that
            if '900_V_01' in xsensfilepaths[trialnames[i]] and data_dict['trialType'] == 'GRAIL':
                wrongfs = 40
                for key in data_dict:
                    if key == 'Timestamp':
                        data_dict[key] = samplerate.resample(data_dict[key], 100/wrongfs, 'sinc_best')
                    elif key == 'Sample Frequency (Hz)':
                        data_dict[key] = data_dict[key]
                    elif key == 'Left foot' or key == 'Right foot' or key == 'Lumbar' or key == 'Sternum':
                        for subkey in data_dict[key]['raw']:
                            if np.shape(data_dict[key]['raw'][subkey])[1] == 3:
                                a = samplerate.resample(data_dict[key]['raw'][subkey][:,0], 100/wrongfs, 'sinc_best')
                                b = samplerate.resample(data_dict[key]['raw'][subkey][:,1], 100/wrongfs, 'sinc_best')
                                c = samplerate.resample(data_dict[key]['raw'][subkey][:,2], 100/wrongfs, 'sinc_best')
                                data_dict[key]['raw'][subkey] = np.vstack((a,b,c))
                                data_dict[key]['raw'][subkey] = np.swapaxes(data_dict[key]['raw'][subkey], 0, 1)
                            elif np.shape(data_dict[key]['raw'][subkey])[1] == 4:
                                a = samplerate.resample(data_dict[key]['raw'][subkey][:,0], 100/wrongfs, 'sinc_best')
                                b = samplerate.resample(data_dict[key]['raw'][subkey][:,1], 100/wrongfs, 'sinc_best')
                                c = samplerate.resample(data_dict[key]['raw'][subkey][:,2], 100/wrongfs, 'sinc_best')
                                d = samplerate.resample(data_dict[key]['raw'][subkey][:,3], 100/wrongfs, 'sinc_best')
                                data_dict[key]['raw'][subkey] = np.vstack((a,b,c,d))
                                data_dict[key]['raw'][subkey] = np.swapaxes(data_dict[key]['raw'][subkey], 0, 1)
            
            
            xsens[trialnames[i]], errors[trialnames[i]] = feet.process(data_dict, showfigure)
        
    return corresponding_files, trialnames, vicon, xsens, errors

def sort_files(allfilenames):
    # Sort files on task
    files= dict()
    files['GRAIL'] = []
    files['GRAIL regular'] = []
    files['GRAIL stroke regular'] = []
    files['GRAIL healthy regular'] = []
    files['GRAIL irregular'] = []
    files['GRAIL stroke irregular'] = []
    files['GRAIL healthy irregular'] = []
    files['Overground'] = [] # Overground lab
    
    for key in allfilenames:
        if '_FS0' in key:
            if 'CVA' in key:
                files['GRAIL stroke regular'].append(key)
                files['GRAIL regular'].append(key)
                files['GRAIL'].append(key)
        if '_SP0' in key:
            if key == '900_V_pp01_SP03.c3d': # Fixed speed trial, accidentily wrongly named
                pass
                # files['GRAIL healthy regular'].append(key)
                # files['GRAIL regular'].append(key)
                # files['GRAIL'].append(key)
            elif key == '900_V_pp16_SP02.c3d': # Turn trial, accidentily wrongly named
                pass
            else:
                if 'CVA' in key:
                    files['GRAIL stroke regular'].append(key)
                    files['GRAIL regular'].append(key)
                    files['GRAIL'].append(key)
                elif '_V_' in key:
                    files['GRAIL healthy regular'].append(key)
                    files['GRAIL regular'].append(key)
                    files['GRAIL'].append(key)
                
        if '_SS' in key:
            if 'CVA' in key:
                files['GRAIL stroke irregular'].append(key)
                files['GRAIL irregular'].append(key)
                files['GRAIL'].append(key)
            elif '_V_' in key:
                files['GRAIL healthy irregular'].append(key)
                files['GRAIL irregular'].append(key)
                files['GRAIL'].append(key)
        
        if '_SW' in key or '2MWT' in key:
            files['Overground'].append(key)
    return files