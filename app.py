import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import time
import imutils

# Load serialized face detector model
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load face mask detector model
try:
    maskNet = load_model("mask_detector.h5", compile=False)
    # Recompile the model with the current TensorFlow version
    maskNet.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Function to detect faces and predict mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # skip empty faces
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Streamlit UI
st.title("😷 Real-Time Face Mask Detection")

# Initialize session state for video capture
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Add simple camera selection
camera_index = st.selectbox("Select Camera", [0, 1, 2], format_func=lambda x: f"Camera {x}")

# Add webcam control
run = st.checkbox("Start Webcam")

# Create a placeholder for the webcam feed with specific size
FRAME_WINDOW = st.image([], width=400)

if run:
    if st.session_state.camera is None:
        try:
            # Initialize video capture with default backend
            st.session_state.camera = cv2.VideoCapture(camera_index)
            
            # Set camera properties
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if not st.session_state.camera.isOpened():
                st.error(f"Could not open camera {camera_index}. Please try a different camera.")
                st.session_state.camera = None
                run = False
            else:
                st.success("Camera started successfully!")
        except Exception as e:
            st.error(f"Error initializing camera: {str(e)}")
            st.session_state.camera = None
            run = False

    while run and st.session_state.camera is not None:
        try:
            ret, frame = st.session_state.camera.read()
            if not ret or frame is None:
                st.error("Failed to read from camera")
                st.session_state.camera.release()
                st.session_state.camera = None
                break

            # Convert BGR to RGB immediately after reading
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for better performance
            frame = imutils.resize(frame, width=400)
            
            # Make a copy for detection
            detection_frame = frame.copy()
            
            # Detect faces and predict mask
            (locs, preds) = detect_and_predict_mask(detection_frame, faceNet, maskNet)

            # Draw bounding boxes and labels
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # Draw on the RGB frame
                cv2.putText(frame, label, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Display the frame using st.image
            FRAME_WINDOW.image(frame, channels="RGB", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            break

else:
    # Release camera when checkbox is unchecked
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
