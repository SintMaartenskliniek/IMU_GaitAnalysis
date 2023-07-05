# -*- coding: utf-8 -*-
"""
Script to read c3d markerdata

Version:
    2023-01-10: C.J. Ensink - check if analog data available
    2022-01-20: C.J. Ensink - add analog data
    2022-12-23: Bart Nienhuis
"""

import c3d
import numpy as np
# import tkinter as tk
# from tkinter import filedialog

# # diverse modules om te kunnen plotten  
# import matplotlib.pyplot as plt
# from matplotlib.textpath import TextPath
# from matplotlib.patches import Rectangle, PathPatch
# import mpl_toolkits.mplot3d.art3d as art3d
# from matplotlib.transforms import Affine2D

def readmarkerdata (filepath, **kwargs):
    
    # definieeer een reader opject welke velden er zijn kun je terug vinden en de c3d.py file
    reader = c3d.Reader(open(filepath, 'rb'))

    #  Lees sample frequentie markerdata
    fs_markerdata = reader.point_rate
    
    # De marker labels zitten in een apparte groep in de c3D file deze kun uitlezen m.b.v. het point_labels veld 
    # De volgorde van de labels is dezelfde als de volgorde in de markerdata.
    markerlabels=reader.point_labels
    
    # Controleer of analoge data (forceplates) beschikbaar is
    analog_available = reader.analog_used>0
    
    analog_wanted = False
    for key, value in kwargs.items():
        if key == 'analogdata' and value == True:
            analog_wanted = True
        elif key == 'analogdata' and value == False:
            analog_wanted = False
    
    # Lees analoge data in
    analoglabels=[]
    if analog_wanted == True:
        if analog_available == True:
            analog_per_frame=reader.header.analog_per_frame
            # fs_analogdata = reader.analog_rate
            # print(fs_analogdata)
            analoglabels = list(reader.analog_labels)
        elif analog_available == False:
            analoglabels = 'None available'
    
    # dotloc = np.array([],dtype=int)
    # for char in range(0,len(analog_labels)):
    #     if analog_labels[char] == '.':
    #         dotloc = np.append(dotloc, char)
    
    # for i in range(0,len(dotloc)):
    #     lastuppercase_beforedot = [idx for idx in range(0,dotloc[i]) if analog_labels[idx].isupper()][-1]
    #     firstspace_afterdot = [idx for idx in range(dotloc[i],len(analog_labels)) if analog_labels[idx] == " "][0]
        
    #     if dotloc[i] < dotloc[-1]:
    #         if firstspace_afterdot > dotloc[i+1]:
    #             firstspace_afterdot = [idx for idx in range(dotloc[i],len(analog_labels)) if analog_labels[idx].isupper()][1]
                
    #     label = analog_labels [lastuppercase_beforedot : firstspace_afterdot]
    #     analoglabels.append(label)

    # Dit zijn de lists waarin de makerdata en analog_data  worden ingelezen
    markerdata_list=[]
    analog_data_list=[]
    analog_per_frame = reader.header.analog_per_frame
    # analog_count = reader.header.analog_count

    for i, points, analog in reader.read_frames():
    #            frames : sequence of (frame number, points, analog)
    #            This method generates a sequence of (frame number, points, analog)
    #            tuples, one tuple per frame. The first element of each tuple is the
    #            frame number. The second is a numpy array of parsed, 5D point data
    #            and the third element of each tuple is a numpy array of analog
    #            values that were recorded during the frame. (Often the analog data
    #            are sampled at a higher frequency than the 3D point data, resulting
    #            in multiple analog frames per frame of point data.)
    #
    #            The first three columns in the returned point data are the (x, y, z)
    #            coordinates of the observed motion capture point. The fourth column
    #            is an estimate of the error for this particular point, and the fifth
    #            column is the number of cameras that observed the point in question.
    #            Both the fourth and fifth values are -1 if the point is considered
    #            to be invalid.
          
        points2=points[:,0:3] # lees alleen de x,y,z, cordinaat 
        markerdata_list.append(points2.T)

        #  LET OP voor de analoge moet de matrix gereshaped worden en dat hangt af van het aantal analoge kanalen in de C3D file
        try:
            analog_data_list.append(analog.T)
        # analog_data_list.append(analog.reshape(analog_per_frame, int(reader.analog_used))) #int(analog_count/analog_per_frame)))
        # analog_data_list.append(analog.reshape(analog_per_frame,36))
        # analog_data_list.append(analog.reshape(analog_per_frame,42))
        except ZeroDivisionError:
            # print('No analog data available')
            continue
        except:
            # print('Failed to read analog data')
            continue
        
        
    # maak van de twee list numpy array
    marker_data=np.stack(markerdata_list, axis=2)
    try:
        analog_data=np.vstack(analog_data_list)   
    except ValueError: # Empty analog_data_list
        analog_data=np.array([]) 
    
    markerdata = dict()
    # Sla merkerdata op onder label naam
    for i in range(0, len(markerlabels)):
            marker_x = marker_data[0,i,:]
            marker_y = marker_data[1,i,:]
            marker_z = marker_data[2,i,:]
            markerdata[markerlabels[i].split(' ')[0]] = np.transpose(np.array([marker_x, marker_y,marker_z]))
    
    analogdata=dict()
    try:
        for i in range(0,len(analoglabels)):
            analogdata[analoglabels[i]] = analog_data[:,i]
    except:
        pass
    
    actual_start_frame = reader.first_frame
    actual_stop_frame = reader.last_frame
    
    return markerdata, fs_markerdata, analogdata
        
    # fig1 = plt.figure()
    # ax3 = fig1.add_subplot(111, projection='3d') 
    # #
    # marker1_x = marker_data[0,0,:]
    # marker1_y = marker_data[1,0,:]
    # marker1_z = marker_data[2,0,:]

    # marker2_x = marker_data[0,1,:]
    # marker2_y = marker_data[1,1,:]
    # marker2_z = marker_data[2,1,:]
    
    # marker3_x = marker_data[0,2,:]
    # marker3_y = marker_data[1,2,:]
    # marker3_z = marker_data[2,2,:]
        
    # # plot de drie markers in een 3D plot
    # ax3.scatter(marker1_x,marker1_y,marker1_z, c='r', marker='o')   
    # ax3.scatter(marker2_x,marker2_y,marker2_z, c='g', marker='o')      
    # ax3.scatter(marker3_x,marker3_y,marker3_z, c='m', marker='o')      


    # ## Position graph x, y z
    # plt.style.use('classic')
    # plt.rcParams['font.size'] =8.0
    # example_3D_plot, axarr=plt.subplots(3, sharex=True)
    # example_3D_plot.subplots_adjust(hspace=0.4)
    
    # l1,=axarr[0].plot(marker1_x)
    # axarr[0].set_title('X')
    # axarr[0].set_ylabel('mm')
    # l2,=axarr[1].plot(marker1_y)
    # axarr[1].set_title('Y')
    # axarr[1].set_ylabel('mm')
    # l3,=axarr[2].plot(marker1_z)
    # axarr[2].set_title('Z')
    # axarr[2].set_ylabel('mm')
    
    # plt.show()