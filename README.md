# Lumbar Spine Degenerative Disease Classification

This project provides a Streamlit-based web interface for automated classification of lumbar spine degenerative diseases from MRI images.  
It supports an intuitive workflow from image upload to model prediction and visualization, making it suitable for clinical applications.

---

## Features

- Upload and analyze DICOM MRI images of the lumbar spine  
- Deep learning models including EfficientNet + LSTM and ResNet + Transformer  
- Per-level prediction of 10 spinal conditions with 3 severity levels  
- Visualization of key regions from sagittal and axial views  
- Interactive result interface compatible with hospital workflows

---

## How to Run

Ensure Python ≥ 3.8 is installed. Then follow these steps:

```bash
git clone https://github.com/your-username/lumbar-spine-ui.git
cd lumbar-spine-ui

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py


## Application Workflow

Start Page
Introduces the application's purpose

Click the Start button to proceed to the Upload Page

Upload Page
Enter the folder path containing .dcm images for analysis

Click the Analyze button to trigger model inference

A loading spinner will appear during processing

Upon completion, the app navigates to the Results Page automatically

Results Page
Displays cropped images for each view (Sagittal T1, T2/STIR, Axial T2) with key regions highlighted

Presents a prediction table showing conditions and severity levels (Normal/Mild, Moderate, Severe)

Includes a Try Another button to restart the workflow


├── app.py                    # Main application controller
├── page_1_start.py           # Start page layout and routing
├── page_2_upload.py          # Upload and analysis logic
├── page_3_results.py         # Results rendering and display

├── assets/
│   └── style.css             # Custom styling

├── utilities/
│   ├── config.py             # Configuration parameters
│   ├── croper.py             # Functions for ROI cropping and visualization
│   ├── data.py               # DICOM file loading and preprocessing
│   ├── model_utils.py        # Model inference and utility functions
│   ├── model2_LSTM.py        # LSTM-based classification model
│   ├── model3_VIT.py         # Transformer-based classification model

├── *.pt                      # Pretrained model weights
├── *.csv                     # Sample outputs or helper files

System Code
The backend system code handles data preprocessing, keypoint detection, and model training.
Although included in the project, it is not the main focus of this repository and is not required to run the UI application.


