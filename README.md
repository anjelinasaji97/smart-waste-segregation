# Smart Waste Segregation System

AI-powered waste detection and classification using YOLOv8 and Streamlit.

## Features
- Real-time waste detection from video
- Classifies waste into Plastic, Glass, Paper, Metal, Organic, E-Waste
- Colored bounding boxes around detected waste
- Bin recommendation system
- Contamination detection (Organic + Plastic mixed)
- Live statistics with waste counts
- Final report with bar chart
- CSV download of waste report

## How to Run
1. Install dependencies:
pip install ultralytics streamlit opencv-python pandas

2. Run the app:
streamlit run app.py

## Technologies Used
- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- Python
