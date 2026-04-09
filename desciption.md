Counter-Surveillance App

Summary

An app using video files to extract and present data based on observations in the video and saving the data to a database.

System

The app shall be made to run natively on a Mac with Apple Silicon, utilizing hardware acceleration such as neural engine and GPU-cores for the object detection and other ML tasks described below. The application should be accessible through, and interacted with, a web browser using a local web server.


Database

A database shall be created with the following fields:

- observation_id
- date_time_first_observation
- date_time_last_observation
- vehicle_make
- vehicle_model
- vehicle_color
- vehicle_licence_plate
- vehicle_licence_plate_color
- vehicle_licence_plate_nationality
- category

The database shall be used during the run time and each observation shall be updated when the information is extracted from the video file.


Function

The data extraction is based on a video file selected by the user as described under the headline "User interaction" below.
The current date and time shall be calculated during playback and be displayed as described under "User interaction" below. This date and time value will also be used to set time stamps for the observations as described below. This time shall be based on the date and time given in the metadata from the file as a starting date and time, with the elapsed time added to this value continuosly as the video is being processed.

The app should perform object detection to identify the outlines, make and model of motor vehicles. These make and model values shall be saved in the database under the field named "vehicle_make" and "vehicle_model" respectively. The color should also be identified if possible and saved in the database in the field "vehicle_color". The first observation of a vehicle in the video frame shall result in the creation of a new observation in ther databsae. Each observation shall be date and time stamped based on the metadata and elapsed time from above and stored under "date_time_first_observation". At the same time the observation shall be given a observation id, which shall be written to the database field "observation_id". An observation id consists of the file name of the video file followed by underscore and a number starting with 1 for the first observation, 2 for the second, and so on.

After the object detection, the vehicle shall be tracked until it is no longer observed. When it is no longer observed the date and time shall be stored in the database in the field named "date_time_last_observation".

Once the vehicle have been detected, an Automatic Number Plate Reader (ANPR) should be applied. If the background color of the licence plate is anything but white with black text it should be saved to the as a "taxi" if the background color is yellow or a "diplomatic" if the background color is blue. This value shall bestored in the database described below under the field "category". If the background color is white with dark text, the field "category" should be set to "normal". The nationality of the licence plate shall be identified if possible and stored in the "vehicle_licence_plate_nationality" field in the database.

If stationary vehicles is present in the beginning of the video, they should only be identified and have its licence plate number identified once, have the box drawn its outlines and only followed if they start to move. In this case, the "date_time_first_observation" shall be set as the time given in the metadata of the video files. Any vehicles that are still observed in the video when it ends shall have the "date_time_first_observation" set to the date and time from the metadata plus the elapsed time of the video.

A photograph of each observed vehicle shall be saved in jpg format to the folder "observation_images". The file name shall consist of the value for "observation_id" followed by ".jpg".

When the session ends, i.e. when the video file has been played from start to finish, the database shall be saved as a csv file in the folder "session_data". The file name is the same as the name of the video file.


User interaction

The app should initially prompt the user to select a video file from the folder named "video_input" by displaying a list of files currently in the folder. When a file have been selected it should be played for the user from start to finish with the date and time displayed on the screen below the displayed video.

A box should be drawn to visualize the confines of each vehicle and also tag the box with the licence plate number.

The right side of the screen should contain a panel with list of boxes, each describing one observed vehicle. The title of this panel shall be "Current observations". Each time a vehicle is being detected, a box containing information about the vehicle should appear in the top of the list by animating the box being pushed down from the top. Each box shall contain a visualization of the number plate with the correct background color from the database, make, model, color, nationalty and the date and time of the firstlast detection and how how long time has elapsed since the first observation. When a vehicle is no longer observed, the corresponding box shall be dropped from the list.

When the video ends, the boxes representing the vehicles that were still observed when the video ended shall remain on the screen, as shall the last frame of the video.