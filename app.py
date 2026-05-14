import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from collections import Counter
import tempfile
import time
import plotly.express as px
from fpdf import FPDF
import os

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

            color_map = {
                "Plastic": "🟠",
                "Glass": "🔵",
                "Organic": "🟢",
                "E-Waste": "🔴",
                "Paper": "🟡",
                "Metal": "⚪",
                "Cardboard": "🟤",
                "General Waste": "⬜"
            }

            cols = st.columns(len(final_df))
            for idx, row in final_df.iterrows():
                emoji = color_map.get(row["Waste Type"], "⬛")
                cols[idx].metric(
                    label=f"{emoji} {row['Waste Type']}",
                    value=int(row["Count"])
                )

            st.dataframe(final_df, use_container_width=True)

            fig_bar = px.bar(
                final_df,
                x="Waste Type",
                y="Count",
                title="Waste Distribution - Bar Chart",
                text="Count"
            )
            fig_bar.update_traces(
                textposition="outside",
                marker_color="#1f77b4",
                textfont=dict(color="black", size=14)
            )
            fig_bar.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="black",
                title_font_color="black",
                showlegend=False,
                xaxis=dict(color="black"),
                yaxis=dict(color="black")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            pie_colors = {
                "Plastic": "#FFA500",
                "Glass": "#00BFFF",
                "Organic": "#00C800",
                "E-Waste": "#FF4444",
                "Paper": "#FFD700",
                "Metal": "#A0A0A0",
                "Cardboard": "#8B5A2B",
                "General Waste": "#808080"
            }

            fig_pie = px.pie(
                final_df,
                names="Waste Type",
                values="Count",
                title="Waste Distribution - Pie Chart",
                color="Waste Type",
                color_discrete_map=pie_colors
            )
            fig_pie.update_traces(
                textfont=dict(color="black", size=14)
            )
            fig_pie.update_layout(
                paper_bgcolor="white",
                font_color="black",
                title_font_color="black",
                legend_font_color="black"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Save charts as large images
            bar_path = "waste_bar.png"
            pie_path = "waste_pie.png"
            fig_bar.write_image(bar_path, width=1800, height=700)
            fig_pie.write_image(pie_path, width=1800, height=700)

            # Build PDF - A4 is 210 x 297 mm
            pdf = FPDF()
            pdf.add_page()

            # Green header with black text
            pdf.set_fill_color(34, 139, 34)
            pdf.rect(0, 0, 210, 20, "F")
            pdf.set_font("Arial", "B", 18)
            pdf.set_text_color(0, 0, 0)
            pdf.set_xy(0, 0)
            pdf.cell(210, 20, "Smart Waste Segregation Report", ln=True, align="C")

            # Summary boxes
            pdf.set_fill_color(220, 240, 255)
            pdf.rect(5, 23, 95, 16, "F")
            pdf.set_xy(5, 23)
            pdf.set_font("Arial", "B", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(95, 8, "Total Unique Objects", ln=False, align="C")
            pdf.set_xy(5, 31)
            pdf.set_font("Arial", "B", 13)
            pdf.cell(95, 8, str(total_items), ln=False, align="C")

            pdf.set_fill_color(220, 255, 220)
            pdf.rect(110, 23, 95, 16, "F")
            pdf.set_xy(110, 23)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(95, 8, "Most Common Waste", ln=False, align="C")
            pdf.set_xy(110, 31)
            pdf.set_font("Arial", "B", 13)
            pdf.cell(95, 8, str(most_common), ln=False, align="C")

            # Table
            pdf.set_xy(5, 42)
            pdf.set_font("Arial", "B", 11)
            pdf.set_fill_color(34, 139, 34)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(100, 8, "Waste Type", border=1, fill=True, align="C")
            pdf.cell(100, 8, "Count", border=1, fill=True, align="C")
            pdf.ln()

            row_colors = {
                "Plastic": (255, 220, 150),
                "Glass": (180, 240, 255),
                "Organic": (180, 255, 180),
                "E-Waste": (255, 180, 180),
                "Paper": (255, 255, 180),
                "Metal": (220, 220, 220),
                "Cardboard": (210, 180, 140),
                "General Waste": (200, 200, 200)
            }

            pdf.set_font("Arial", "", 11)
            for _, row in final_df.iterrows():
                r, g, b = row_colors.get(row["Waste Type"], (240, 240, 240))
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(100, 8, str(row["Waste Type"]), border=1, fill=True, align="C")
                pdf.cell(100, 8, str(int(row["Count"])), border=1, fill=True, align="C")
                pdf.ln()

            table_end_y = pdf.get_y() + 3

            # Bar chart full width
            if os.path.exists(bar_path):
                pdf.image(bar_path, x=5, y=table_end_y, w=200)
                bar_end_y = table_end_y + 65
                os.remove(bar_path)
            else:
                bar_end_y = table_end_y

            # Pie chart full width below bar
            if os.path.exists(pie_path):
                pdf.image(pie_path, x=5, y=bar_end_y, w=200)
                os.remove(pie_path)

            # Footer
            pdf.set_y(-10)
            pdf.set_font("Arial", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 10, "Generated by Smart Waste Segregation System", align="C")

            pdf_output = bytes(pdf.output())

            st.download_button(
                "📥 Download PDF Report",
                data=pdf_output,
                file_name="waste_report.pdf",
                mime="application/pdf"
            )
