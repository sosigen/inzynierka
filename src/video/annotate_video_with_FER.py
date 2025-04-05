import cv2
import numpy as np
from tensorflow import keras
import sys

# -------------------------
# 1. Load Your Model and Define Paths
# -------------------------
model_path = "./fer2013plus_79%.keras"
input_video_path = "./input_no_headphones.mp4"
face_cascade_path = "./haarcascade_frontalface_alt2.xml"
skip_n = 4  # Run model prediction every 4th frame

# Parameters for model input
target_size = (48, 48)       # Change this to match your model's expected resolution
desired_channels = 1         # Set to 1 for single-channel input; set to 3 to duplicate grayscale into 3 channels

model = keras.models.load_model(model_path)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# -------------------------
# 2. Video Setup for Processing
# -------------------------
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file '{input_video_path}'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Info -> FPS: {fps}, Size: ({width}x{height}), Total Frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "annotated_output.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
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
# 4. Processing Phase with Key Frame Prediction and Tracking
# -------------------------
frame_count = 0
current_trackers = []  # List of dictionaries with 'tracker' and 'texts'
bar_length = 40

while True:
    ret, frame = cap.read()
    if not ret:
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
    
    out.write(annotated_frame)
    frame_count += 1

    # Update terminal progress bar
    progress = frame_count / total_frames
    filled_length = int(bar_length * progress)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\rProcessing: [{bar}] {frame_count}/{total_frames} frames")
    sys.stdout.flush()

print("\nProcessing complete.")
cap.release()
out.release()

print(f"Video saved as {output_path}")

# -------------------------
# 5. Playback of Annotated Video
# -------------------------
cap_annotated = cv2.VideoCapture(output_path)
cv2.namedWindow("Annotated Video", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap_annotated.read()
    if not ret:
        break
    cv2.imshow("Annotated Video", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF in [27, ord('q')]:
        break
cap_annotated.release()
cv2.destroyAllWindows()
