Explanation of the code and the order of running the code is commented through the scripts.

In case you are only interested in the gait analysis pipeline, only the code in the folder "gaittool" is of importance.
The "gaittool" folder contains functions to import, preprocess and process IMU data, and can be run with the following commands:
import gaittool.feet_processor.test_processor as process
data, errors = process("full path to your dataset here")

Winthin the \helpers\preprocessor.py file, the exported xsens Awinda files (.txt files in the "\Xsens\exported### folders in "data"), and other datastructures dependent on the type of IMU's used, data will be imported and structured for further processing.
See this file for further details of the data structure that is needed for further analysis.

Within \feet_processor\processor.py, the function test_processor is the main function to use. This function will import the other functions needed for the data analysis.
The output of the test_processor function contains two dictionaries, one with the analyzed data and the other with possible errors. The analyzed data dictionary has the following structure:

Left foot (dict) > dictionay with raw data as provided by the sensor, and derived data as calculated by the pipeline from this sensor
Right foot (dict) > dictionay with raw data as provided by the sensor, and derived data as calculated by the pipeline from this sensor
Lumbar (dict) > dictionay with raw data as provided by the sensor, and derived data as calculated by the pipeline from this sensor
Missing Sensors (dict) > dictionary describing possible missing sensors
Sample Frequency (Hz) (int) > sample frequency of the data (default in preprocessor for different sensortypes, but can manually be changed using the keyword argument "sample_frequency" in the test_processor function
Spatiotemporals (dict) > dictionary with all gait parameters derived from this trial (list of parameters and units provided in the comments of the feet_processor\processor.py file)
Sternum (dict) > dictionay with raw data as provided by the sensor, and derived data as calculated by the pipeline from this sensor
Timestamp (float) > timestamps of the data samples
trialType (str) > type of walking test, can be changed using the keyword argument "trialType" in the test_processor function