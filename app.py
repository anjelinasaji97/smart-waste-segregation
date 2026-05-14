import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from collections import Counter
import tempfile
import time

model = YOLO("yolov8m.pt")

waste_classes = {
    "bottle": "Plastic",
    "cup": "Plastic",
    "wine glass": "Glass",
    "bowl": "Plastic",
    "banana": "Organic",
    "apple": "Organic",
    "orange": "Organic",
    "cell phone": "E-Waste",
    "laptop": "E-Waste",
    "tv": "E-Waste",
    "keyboard": "E-Waste",
    "mouse": "E-Waste",
    "book": "Paper",
    "scissors": "Metal",
    "fork": "Metal",
    "knife": "Metal",
    "spoon": "Metal"
}

st.title("Smart Waste Segregation System")
uploaded_video = st.file_uploader("Upload Waste Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.flush()

    cap = cv2.VideoCapture(temp_video.name)
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    counted_ids = {}
    total_counts = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, conf=0.3)
        annotated_frame = results[0].plot()
        detected_types = []

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i, cls in enumerate(boxes.cls):
                class_name = model.names[int(cls)]
                if class_name not in waste_classes:
                    continue
                waste_type = waste_classes[class_name]
                detected_types.append(waste_type)
                if boxes.id is not None:
                    track_id = int(boxes.id[i])
                    if track_id not in counted_ids:
                        counted_ids[track_id] = waste_type
                        total_counts[waste_type] += 1

        contamination = "Organic" in detected_types and "Plastic" in detected_types

        if contamination:
            cv2.putText(annotated_frame, "CONTAMINATION DETECTED!", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        with stats_placeholder.container():
            st.write("### Live Statistics")
            st.metric("Total Unique Objects", sum(total_counts.values()))
            if contamination:
                st.error("Contamination! Organic and Plastic mixed!")
            else:
                st.success("No contamination detected")

        time.sleep(0.03)

    cap.release()
    st.success("Video Processing Completed!")
