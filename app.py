import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import torch.nn as nn

# Constants
WIDTH = 224
HEIGHT = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load the model
@st.cache_data()
def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the saved model
model_path = './resnet50_finetuned.pth'
class_names = ['Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 'Common Kingfisher', 'Common Myna', 'Common Rosefinch']
model = load_model(model_path, len(class_names))

# Function to preprocess and predict on a single image
def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    
    return predicted_class

# Streamlit App Interface
st.title("Bird Species Classifier")
st.write("Upload an image of a bird, and the model will predict its species. (Only 6 classes are supported, see below)")
st.write("1. Asian Green Bee-Eater\n2. Brown-Headed Barbet\n3. Cattle Egret\n4. Common Kingfisher\n5. Common Myna\n6. Common Rosefinch")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Predict the class
    predicted_class = predict_image(image, model, class_names)

    # Hide the classifier message
    st.write("")  # Clears the "Classifying..." placeholder after prediction

    # Display the result with improved formatting
    st.markdown(f"<h2 style='text-align: center; color: green;'>Predicted Class: {predicted_class}</h2>", unsafe_allow_html=True)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
