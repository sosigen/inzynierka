import cv2
import numpy as np
from tensorflow import keras
import sys

# -------------------------
# 1. Load Your Model and Define Paths
# -------------------------
model_path = "./fer2013plus_79%.keras"
face_cascade_path = "./haarcascade_frontalface_alt2.xml"
skip_n = 5


target_size = (48, 48) # Change this to match model's input layer shape
desired_channels = 1  # Set to 1 for single-channel input; set to 3 to duplicate grayscale into 3 channels

# Load the model and emotion classes
model = keras.models.load_model(model_path)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load the face cascade
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Error: Failed to load face cascade classifier.")
    exit()

# -------------------------
# 2. Camera Setup for Processing
# -------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cv2.namedWindow("Live Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Camera Feed", 1280, 740)

# -------------------------
# 3. Functions for Detection, Annotation, and Tracking
# -------------------------
def compute_annotations(frame):
    """
    Detect faces and compute annotations (bounding boxes and emotion predictions).
    Returns a list of dictionaries with 'bbox' and 'texts'.
    """
    annotations = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Process face region for prediction
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, target_size)
        # Convert to grayscale for prediction
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_resized = face_resized / 255.0
        
        # Duplicate channels if required
        if desired_channels > 1:
            face_resized = np.repeat(face_resized[..., np.newaxis], desired_channels, axis=-1)
        else:
            face_resized = np.expand_dims(face_resized, axis=-1)
        
        face_resized = np.expand_dims(face_resized, axis=0)  # shape: (1, target_size[1], target_size[0], channels)
        
        predictions = model.predict(face_resized, verbose=0)[0]
        emotion_probs = list(zip(emotion_classes, predictions))
        emotion_probs.sort(key=lambda x: x[1], reverse=True)
        
        texts = []
        text_x = x + w + 10
        text_y = y + 20
        for i, (label, prob) in enumerate(emotion_probs):
            line_text = f"{label}: {prob * 100:.2f}%"
            font_scale = 0.7 if i == 0 else 0.5
            texts.append((line_text, (text_x, text_y + i*20), font_scale))
        
        annotations.append({
            "bbox": (x, y, w, h),
            "texts": texts
        })
    return annotations

def initialize_trackers(frame, annotations):
    """
    Initializes a CSRT tracker for each detected face.
    Returns a list of dictionaries with keys: 'tracker' and 'texts'.
    """
    trackers = []
    for ann in annotations:
        bbox = ann["bbox"]
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        trackers.append({
            "tracker": tracker,
            "texts": ann["texts"]
        })
    return trackers

def apply_annotations(frame, annotations):
    """
    Draws bounding boxes and texts on the frame using the provided annotations.
    Each annotation is expected to have a 'bbox' and associated 'texts'.
    """
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        for text, pos, font_scale in ann["texts"]:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

# -------------------------
# 4. Processing Phase with Live Camera Feed
# -------------------------
frame_count = 0
current_trackers = []  # List of dictionaries with 'tracker' and 'texts'

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if frame_count % skip_n == 0:
        # On key frames, perform face detection and emotion prediction,
        # then initialize trackers for each detected face.
        annotations = compute_annotations(frame)
        current_trackers = initialize_trackers(frame, annotations)
        annotated_frame = apply_annotations(frame.copy(), annotations)
    else:
        # On non-key frames, update each tracker to get new bounding boxes.
        updated_annotations = []
        for ann in current_trackers:
            tracker = ann["tracker"]
            success, bbox = tracker.update(frame)
            if success:
                updated_annotations.append({
                    "bbox": bbox,
                    "texts": ann["texts"]
                })
        annotated_frame = apply_annotations(frame.copy(), updated_annotations)
    
    cv2.imshow("Live Camera Feed", annotated_frame)
    
    # Exit if 'Esc' or 'q' is pressed
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
