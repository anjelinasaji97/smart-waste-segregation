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
    "broccoli": "Organic",
    "carrot": "Organic",
    "sandwich": "Organic",
    "pizza": "Organic",
    "cake": "Organic",
    "cell phone": "E-Waste",
    "laptop": "E-Waste",
    "tv": "E-Waste",
    "keyboard": "E-Waste",
    "mouse": "E-Waste",
    "remote": "E-Waste",
    "book": "Paper",
    "scissors": "Metal",
    "fork": "Metal",
    "knife": "Metal",
    "spoon": "Metal",
    "can": "Metal",
    "toothbrush": "Plastic",
    "vase": "Glass",
    "suitcase": "Plastic",
    "backpack": "Plastic",
    "handbag": "Plastic"
}

COLORS = {
    "Plastic": (255, 165, 0),
    "Glass": (0, 255, 255),
    "Organic": (0, 200, 0),
    "E-Waste": (255, 0, 0),
    "Paper": (255, 255, 0),
    "Metal": (180, 180, 180),
    "Cardboard": (139, 90, 43),
    "General Waste": (128, 128, 128)
}

st.set_page_config(page_title="Smart Waste Segregation", layout="wide")

st.title("♻️ Smart Waste Segregation System")
st.markdown("### AI-Powered Waste Detection and Classification")

uploaded_video = st.file_uploader("Upload Waste Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.flush()

    cap = cv2.VideoCapture(temp_video.name)

    if not cap.isOpened():
        st.error("Could not open video.")

    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            frame_placeholder = st.empty()

        with col2:
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

                    label = f"{waste_type}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
                if "Paper" in detected_types or "Cardboard" in detected_types:
                    recommendation = "Dry Waste"
                if "Organic" in detected_types:
                    recommendation = "Biodegradable Waste"
                if "E-Waste" in detected_types:
                    recommendation = "Hazardous / E-Waste"
                if contamination:
                    recommendation = "CONTAMINATED - Sort Before Disposal"

            cv2.putText(annotated_frame, f"Bin: {recommendation}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if contamination:
                cv2.putText(annotated_frame, "CONTAMINATION DETECTED!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            total_items = sum(total_counts.values())
            most_common = total_counts.most_common(1)[0][0] if total_counts else "None"
            stats_df = pd.DataFrame(total_counts.items(), columns=["Waste Type", "Count"])

            with stats_placeholder.container():
                st.markdown("### 📊 Live Statistics")
                st.metric("Total Unique Objects", total_items)
                st.metric("Most Common Waste", most_common)
                st.metric("Bin Recommendation", recommendation)

                if not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True)

                if contamination:
                    st.error("⚠️ Contamination! Organic + Plastic mixed!")
                else:
                    st.success("✅ No contamination detected")

            time.sleep(0.03)

        cap.release()

        st.success("✅ Video Processing Completed!")

        final_df = pd.DataFrame(total_counts.items(), columns=["Waste Type", "Count"])

        if not final_df.empty:
            st.markdown("## 📋 Final Waste Report")
            st.dataframe(final_df, use_container_width=True)
            st.bar_chart(final_df.set_index("Waste Type"))
            csv = final_df.to_csv(index=False)
            st.download_button("📥 Download CSV Report", csv, "waste_report.csv", "text/csv")
