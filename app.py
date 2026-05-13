import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO

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
        detected_types = []

        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                class_name = model.names[int(cls)]
                if class_name in waste_classes:
                    waste_type = waste_classes[class_name]
                    detected_types.append(waste_type)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        if detected_types:
            st.write(f"Detected: {set(detected_types)}")

    cap.release()
    st.success("Video Processing Completed!")
