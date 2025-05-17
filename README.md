Our UI contains three main pages:
1.	Start Page: Brief introduction with navigation to the upload page.
2.	Upload Page: Upload folder path containing MRI images for analysis.
3.	Results Page: View the classification results with visualizations and predicted severity levels.
1. Start Page
The Start Page introduces the app's purpose, giving a brief overview of its functionality.
Navigation Button: Click on the Start button to move to the next page.

2. Upload Page
Follow the on-screen instructions to upload your file.
 	
Input Folder: Enter folder path to select your file. Ensure it contains DICOM (.dcm) images of MRI scans.
Analyze Button: After uploading the file, you will see the Analyze button. Click this to initiate the model’s classification process.
The application may take a few moments to process, and a spinner will display while the model is analyzing the images. After analysis, you will be automatically redirected to the Results Page to view the outcomes.

3. Results Page
View the classification results with visualizations and the predicted severity levels.

Cropped Image with Key Regions:
Images for each view (Sagittal T2/STIR, Sagittal T1, Axial T2) are shown with highlighted bounding boxes around areas of interest.


Prediction Table:
Displays predictions for each condition at specific spinal levels, showing severity as Normal/Mild, Moderate, or Severe. The table includes columns for Condition, Label, and Severity to make it easy to interpret each classification.
Refresh Button: If you want to try another analysis or start over, click the Try Another button to return to the Upload Page.

4. Code Structure of Streamlit UI
 Assets/: Store static resources used in the project
Style.css: Style files
Utilities/: Mainly tool functions and model files used for processing data, model inference, etc. Specifically, it includes:
Config. py: A configuration file that stores some global parameters.
Croper. py: Used to crop images and display crop boxes.
Data.py: Processing related to data.
Model_utils. py: Contains utility functions related to model inference, loading, and more.
Model2.LSTM.Py: The framework of the LSTM model.
Model3.VIT.Py: The framework of the VIT model.
Resnet18_1_input3Channels.pt and other PT files: Save the trained model for object detection.
Sample_Subsmission. csv and other CSV files: CSV files used in the project.
App. py: The main application entry file that controls navigation between pages.
Page_1_start. py, page_2_upload. py, and page_3. results. py correspond to the three pages of the application (start, upload, and results pages), respectively.

