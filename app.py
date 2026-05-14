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

COLORS = {
    "Plastic": (255, 165, 0),
    "Glass": (0, 255, 255),
    "Organic": (0, 200, 0),
    "E-Waste": (255, 0, 0),
    "Paper": (255, 255, 0),
    "Metal": (180, 180, 180)
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
        annotated_frame = frame.copy()
        detected_types = []

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i, cls in enumerate(boxes.cls):
                class_name = model.names[int(cls)]
                if class_name not in waste_classes:
                    continue
                waste_type = waste_classes[class_name]
                detected_types.append(waste_type)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                color = COLORS.get(waste_type, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, waste_type, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if boxes.id is not None:
                    track_id = int(boxes.id[i])
                    if track_id not in counted_ids:
                        counted_ids[track_id] = waste_type
                        total_counts[waste_type] += 1

        contamination = "Organic" in detected_types and "Plastic" in detected_types

        recommendation = "No Waste Detected"
        if detected_types:
            if "Plastic" in detected_types or "Glass" in detected_types or "Metal" in detected_types:
                recommendation = "Recyclable Waste"
            if "Paper" in detected_types:
                recommendation = "Dry Waste"
            if "Organic" in detected_types:
                recommendation = "Biodegradable Waste"
            if "E-Waste" in detected_types:
                recommendation = "Hazardous / E-Waste"

        cv2.putText(annotated_frame, f"Bin: {recommendation}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if contamination:
            cv2.putText(annotated_frame, "CONTAMINATION DETECTED!", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        stats_df = pd.DataFrame(total_counts.items(), columns=["Waste Type", "Count"])
        with stats_placeholder.container():
            st.write("### Live Statistics")
            st.metric("Total Unique Objects", sum(total_counts.values()))
            st.metric("Bin Recommendation", recommendation)
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True)
            if contamination:
                st.error("Contamination! Organic and Plastic mixed!")
            else:
                st.success("No contamination detected")

        time.sleep(0.03)

    cap.release()
    st.success("Video Processing Completed!")
