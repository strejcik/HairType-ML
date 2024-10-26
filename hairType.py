from PIL import Image
import numpy as np
import joblib  # for loading the .pkl model file
import torch
from fastai.vision.all import *

# Load the image and preprocess it
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image from the file path and preprocess it to match the model's input size.
    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target size for resizing the image.
    Returns:
        np.array: The preprocessed image as a numpy array.
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert image to numpy array and normalize if needed
    image_array = np.array(image) / 255.0  # Scale pixel values to [0, 1]
    
    # Flatten the array if the model requires a 1D input, or adjust accordingly
    image_array = image_array.flatten().reshape(1, -1)  # Reshape to match model input
    
    return image_array

# Load the pre-trained model
def load_model(model_path):
    """
    Load the machine learning model from a .pkl file.
    Args:
        model_path (str): The path to the .pkl model file.
    Returns:
        model: The loaded ML model.
    """
    return load_learner(model_path)

# Apply the model to the preprocessed image
def predict_image(image_path, model_path):
    """
    Make a prediction on the image using the loaded model.
    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the .pkl model file.
    Returns:
        The model's prediction and class probabilities.
    """
    # Load the model
    model = load_model(model_path)
    
    # Predict with the model, using the file path directly since `fastai` models handle the image loading
    prediction, _, probabilities = model.predict(image_path)
    
    return prediction, probabilities

# Example usage:
image_path = "./image.jpg"  # Update with the actual image path
model_path = "./AI_Hair_Type_Detection.pkl"  # Update with the actual model path

# Get the prediction
prediction, probabilities = predict_image(image_path, model_path)
print("Prediction:", prediction)
print("Class Probabilities:", probabilities)