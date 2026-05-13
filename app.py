import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

st.title("Smart Waste Segregation System")
st.write("AI-powered waste detection using YOLOv8")

uploaded_video = st.file_uploader("Upload Waste Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.flush()

    cap = cv2.VideoCapture(temp_video.name)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, conf=0.3)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    st.success("Video Processing Completed!")
