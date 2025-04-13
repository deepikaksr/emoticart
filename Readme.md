# EmotiCart: Real-Time Emotion-Based Product Detection System

EmotiCart is a smart shopping assistant that leverages real-time emotion and object detection technologies to provide personalized product recommendations in retail settings. By capturing a live webcam feed, the system detects both the product being viewed and the customer's mood. Based on the analysis, EmotiCart suggests tailored products that match the customer’s current emotional state.

## Features

- **Real-Time Emotion Detection:**  
  Uses the FER library to analyze facial expressions and detect emotions such as Happy, Neutral, Curious, Angry, or Sad.

- **Object Detection:**  
  Utilizes the YOLO model to identify products in the camera’s field of view.

- **Personalized Recommendations:**  
  Based on the detected emotion, the system provides smart suggestions. For example, if a customer’s mood is positive (happy, neutral, or curious) when viewing a product, the app offers product details from a CSV file containing curated recommendations.

- **Interactive UI:**  
  Built with Streamlit, the interface displays the live camera feed, smart suggestion panels, and exploration UI with Yes/No options to trigger detailed product views or display supportive messages.

## Technologies Used

- **Python**: Programming language used for implementation.
- **Streamlit**: Framework for building interactive web apps.
- **OpenCV**: Library for real-time computer vision tasks.
- **YOLO (Ultralytics)**: Model for object detection.
- **FER**: Library for facial emotion recognition.
- **Pandas**: Data manipulation library used to read product data from CSV files.

## Project Structure

```bash
EMOTICART/
├── __pycache__/         # Compiled Python cache files
├── app.py               # Main Streamlit application
├── detect_utils.py      # Utility functions for object and emotion detection
├── product_data.csv     # CSV file containing product details
├── Readme.md            # Project documentation
├── requirements.txt     # List of required Python packages
└── yolov8n.pt           # YOLO model weights for object detection
```

## Installation & Setup

### Clone this repository.
git clone https://github.com/deepikaksr/emoticart.git cd emoticart

## Requirements & Dependencies

Make sure you have Python 3.7 or later installed. All necessary Python packages are listed in the `requirements.txt` file. Some of the key dependencies include:
- `streamlit`
- `opencv-python`
- `numpy`
- `pandas`
- `ultralytics`
- `fer`

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## Prepare the Product Data:

Ensure that your `product_data.csv` file is in the project root. The CSV should include the following columns:
-`Category`
-`Title`
-`Price (INR)`
-`Description`
-`Image URL`

## Running the App

### Start Streamlit:
```bash
streamlit run app.py
```

### Open Your Browser:
Navigate to the local URL shown in the terminal (http://localhost:8501).

## Usage

### Start the Camera
Click on **"Start Camera"** to initiate the live webcam feed.

### Live Detection
The app continuously captures frames, detecting both your emotion and the product in view.

### Smart Suggestions
If a positive emotion (happy, neutral, or curious) is detected while viewing a product, an exploration UI appears prompting you to:
- **Click Yes** to view detailed product recommendations.
- **Click No** to see a supportive message.

### Feedback
When a response is given, the video feed stops immediately and the chosen outcome is displayed.
