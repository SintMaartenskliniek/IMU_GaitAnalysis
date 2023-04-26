The data folder has the following structure:

- data
	- CVA (data of stroke participants)
		- 900_CVA_## (folder for participant based on study ID)
			- Vicon (folder with vicon, gold standard, data)
				- GRAIL (folder with vicon data from GRAIL trials)
			- Xsens (folder with IMU sensordata)
				- exported### (.txt files per sensor)
		- ...

	- Healthy controls (data of healthy participants)
		- 900_V_## (folder for participant based on study ID)
			- Vicon (folder with vicon, gold standard, data)
				- GBA (folder with vicon data from overground walking trials)
				- GRAIL (folder with vicon data from GRAIL trials)
			- Xsens (folder with IMU sensordata)
				- exported### (.txt files per sensor)
				- ...
		- ...

The vicon data is saved as .c3d files
The xsens sensordata is initially saved as an .mtb file for each trial, 
all trials are exported with MTManager to .txt files in the different "exported###" folders for each trial (see "MT Manager export settings.PNG" figure for the export settings).


