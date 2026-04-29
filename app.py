import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IP102 Pest Detection", page_icon="🐛", layout="wide")

st.title("🐛 IP102 Multi-Crop Multi-Pest Detection")
st.markdown("""
    Upload an image of a crop to detect and identify pests using the **YOLOv8** model.
    This system is trained to recognize 102 different types of agricultural pests.
""")

# --- MODEL LOADING ---
# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Path to your trained model weights
# Change 'best.pt' to your actual model file path
model_path = "best.pt" 
model = load_model(model_path)

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Model Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45)

st.sidebar.markdown("---")
st.sidebar.info("This app uses a YOLOv8 model trained on the IP102 dataset.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # --- INFERENCE ---
    if model is not None:
        with st.spinner("Detecting pests..."):
            # Run inference
            results = model.predict(
                source=image, 
                conf=conf_threshold, 
                iou=iou_threshold,
                save=False
            )
            
            # Plot results
            # results[0].plot() returns a BGR numpy array
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB for Streamlit/PIL
            res_plotted_rgb = res_plotted[:, :, ::-1] 
            
            with col2:
                st.subheader("Detection Result")
                st.image(res_plotted_rgb, use_column_width=True)

        # --- RESULTS SUMMARY ---
        st.subheader("Detection Details")
        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"- **Detected:** `{label}` with `{conf:.2f}` confidence")
        else:
            st.warning("No pests detected with the current confidence threshold.")
    else:
        st.error("Model not found. Please ensure 'best.pt' is in the directory.")

else:
    st.info("Please upload an image to start detection.")
