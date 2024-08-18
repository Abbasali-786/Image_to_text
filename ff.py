import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Title and Description
st.title("Image Caption Generator")
st.write("This app generates captions for an image using the BLIP model by Salesforce.")

# Hugging Face Token Input
hf_token = "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"

# Load the Processor and Model
try:
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_auth_token=hf_token)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_auth_token=hf_token)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Image Upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Generate Caption Button
if st.button("Generate Caption"):
    if uploaded_image:
        try:
            # Load and display the image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate caption
            text = "A picture of"
            inputs = processor(images=image, text=text, return_tensors="pt")

            # Ensure CUDA is available, otherwise use CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model.generate(**inputs)

            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Display the caption
            st.write(f"**Generated Caption:** {caption}")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.warning("Please upload an image.")
st.write["Devloped by Ghulam Abbas"]
