import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import tempfile
import os
import json
import pandas as pd

# Google Drive file IDs for model, class labels, and descriptions
MODEL_DRIVE_ID = "10J34f1PsyqYXfj6axuvGAQVu7R-lJwZ-"
LABELS_DRIVE_ID = "your_labels_json_id_here"  # Replace with actual file ID
DESCRIPTIONS_DRIVE_ID = "your_descriptions_csv_id_here"  # Replace with actual file ID

# Local paths
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"
DESCRIPTIONS_PATH = "class_descriptions.csv"

# Function to download files from Google Drive
def download_file(drive_id, output_path):
    if not os.path.exists(output_path):
        st.info(f"Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={drive_id}", output_path, quiet=False)

# Download model, labels, and descriptions
download_file(MODEL_DRIVE_ID, MODEL_PATH)
download_file(LABELS_DRIVE_ID, LABELS_PATH)
download_file(DESCRIPTIONS_DRIVE_ID, DESCRIPTIONS_PATH)

# Load model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
model = load_model()

# Load class mapping
with open(LABELS_PATH, "r") as file:
    class_mapping = json.load(file)  # Read JSON (Image filename → Class name)

# Extract only unique class names
unique_class_names = sorted(set(class_mapping.values()))  # Remove duplicates

# Create class index mapping
CLASS_NAMES = {i: class_name for i, class_name in enumerate(unique_class_names)}



# Load class descriptions from CSV
df = pd.read_csv(DESCRIPTIONS_PATH)
class_descriptions = dict(zip(df['Class'], df['Description']))
label_names = df['Class'].tolist()

# Streamlit app title
st.title("Image Classification App 📸")
st.subheader("Upload an image and let the AI predict what it sees!")

# Friendly Information
st.markdown(
    """
    ### 🎉 About This App
    - This app can currently classify **a limited number of objects**.
    - The available classes are:
    """
)
# Display class names in a stylish way
st.markdown("#### 🏷️ Supported Classes:")

st.markdown("✅ " + "\n✅ ".join(label_names))

st.markdown("👀 Try uploading an image and see if the model gets it right! Have fun!")
# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

def preprocess_image(img_path):
    """Preprocess image same as in notebook"""
    img = image.load_img(img_path, target_size=(150, 150))  # Match trained model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch
    return img_array

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    # Preprocess image
    img_array = preprocess_image(temp_path)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

      # Debugging outputs
    st.write(f"Predicted Index: {predicted_index}")

    if predicted_index in CLASS_NAMES:
        predicted_class = CLASS_NAMES[predicted_index]
        predicted_description = class_descriptions.get(predicted_class, "No description available")
    else:
        predicted_class = "Unknown"
        predicted_description = "No description available"

     # Extract real class from filename (assuming dataset filenames contain class names)
    real_class = "Unknown"
    filename = uploaded_file.name.lower()
    for class_index, class_name in CLASS_NAMES.items():
        if str(class_name).lower() in filename:
            real_class = class_name
            break

    # Compare prediction vs real class
    if real_class == predicted_class:
        st.success(f"✅ **Correct Prediction!**")
    else:
        st.warning(f"❌ **Incorrect Prediction**")

    st.write(f"**Actual Class:** {real_class}")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Description:** {predicted_description}")

    os.remove(temp_path)

# Footer
st.markdown("---")
st.markdown("📌 *Note: This is a basic model and may not always be accurate. More classes will be added soon!* 🚀")
