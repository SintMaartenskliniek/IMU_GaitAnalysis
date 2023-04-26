Gait analysis with inertial measurement units

** PUBLICATION OF THIS VALIDATION WILL FOLLOW **


*Version of the code used in this publication can be found in the **Validation study** release.*

 
 
 
Run the **validation_study.py** code to analyze the spatiotemporal parameters of the gait data used in this study.
- The folder **data** contains the data used in this study.
- The folder **functions_validationstudy** contains the functions the main script requires to run the validation and import all data in a structured way.
- The folder **gaittool** contains functions to import, preprocess and process the data of a single trial.
- The folder **VICON_functions** contains functions to analyze the optical motion capture data.

Explanation of the code and the order of running the code is commented through the scripts.
Explanation of the data is provided in the ReadMe file in the folder "data".

In case you are only interested in the gait analysis pipeline, only the code in the folder "gaittool" is of importance.
The "gaittool" folder contains functions to import, preprocess and process IMU data, and can be run with the following commands:
- import gaittool.feet_processor.test_processor as process
- data, errors = process(*"full path to your dataset here"*)
