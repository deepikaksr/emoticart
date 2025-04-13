import cv2
from fer import FER
from ultralytics import YOLO

# Load the YOLO model for object detection
yolo_model = YOLO("yolov8n.pt")
# Initialize the FER emotion detector (with MTCNN for face detection)
emotion_detector = FER(mtcnn=True)

def detect_objects_and_emotions(frame):
    # Run object detection on the frame
    results = yolo_model(frame)[0]
    object_labels = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        object_labels.append(label)

    # Run emotion detection on the frame
    emotion_results = emotion_detector.detect_emotions(frame)
    emotion = None
    if emotion_results:
        top_emotion = emotion_results[0]["emotions"]
        # Determine the emotion with the highest probability
        emotion = max(top_emotion, key=top_emotion.get)

    return object_labels, emotion
