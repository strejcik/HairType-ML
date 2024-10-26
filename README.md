# HairType with Pre-trained Model

![alt text](https://i.imgur.com/MDJfzDT.gif)
This code script is a Python program designed for loading an image, preprocessing it, and making predictions using a pre-trained machine learning model. Model has been trained on 1994 images using fast.ai (ResNet-50). Image 'image.jpg' has been downloaded as an input example from https://thispersondoesnotexist.com/

## Libraries and Imports

- **PIL (Python Imaging Library)**: Used for image handling. Here, it's imported specifically to open and process images.
- **NumPy**: Primarily used for handling and manipulating numerical data. In this script, it helps to transform image data into a format the model can work with.
- **Joblib**: Typically used for loading `.pkl` (pickle) model files. However, in this code, it is unused since `fastai` handles model loading directly.
- **torch**: A deep learning library (PyTorch). It's often used with `fastai`, although not directly utilized in this script.
- **fastai**: This high-level library simplifies deep learning model implementation with PyTorch. Here, it's used for loading the model and predicting from the image.

## Code Structure and Function Descriptions

### 1. `load_and_preprocess_image(image_path, target_size=(224, 224))`

This function loads and preprocesses an image to match the model's input requirements.

- **Parameters**:
  - `image_path`: Path to the image file.
  - `target_size`: Desired size for resizing the image. Defaults to `(224, 224)`, a common input size for deep learning models.
- **Process**:
  - Opens the image, converts it to RGB (to ensure color consistency), resizes it, and then converts it into a NumPy array.
  - The pixel values are scaled to be between `[0, 1]` by dividing by `255.0`, which is typical for deep learning models as it standardizes the input.
  - Finally, the image array is reshaped to match the expected input size for certain ML models.
- **Returns**: A 2D NumPy array of the image, ready for input into the model.

### 2. `load_model(model_path)`

This function loads a pre-trained machine learning model using `fastai`'s `load_learner`.

- **Parameter**:
  - `model_path`: Path to the `.pkl` file containing the pre-trained model.
- **Returns**: The loaded model.

### 3. `predict_image(image_path, model_path)`

This function manages the complete prediction pipeline: loading the model, predicting on the image, and returning the results.

- **Parameters**:
  - `image_path`: Path to the image to be classified.
  - `model_path`: Path to the `.pkl` model file.
- **Process**:
  - Calls `load_model` to load the pre-trained model.
  - Uses `model.predict` on `image_path` directly, which `fastai` models handle natively. They load, preprocess, and predict on images directly from file paths.
- **Returns**: The model’s prediction (class label) and the associated class probabilities.

## Example Usage

1. Clone repository
2. Download ML model from: [ML Model](https://drive.google.com/file/d/1NuvzhUDu_7Isab63mirBQmHVFuY2S9mC) and put inside root directory
3. Make sure that you have installed Python 3.12.7 and checked "Add python.exe to PATH"
4. Install pip: py -m ensurepip --upgrade, open cmd and type: pip install --upgrade pip==24.2
5. Open cmd and install all dependencies from "requirements.txt" in order ie. pip install pillow, pip install joblib, and so on
6. Run: py ./hairType.py

The example at the end initializes paths for an image (`image_path`) and a model (`model_path`).
`predict_image` is called with these paths to get a prediction and probability values, which are then printed out.

## Summary

This script effectively combines image handling, preprocessing, and prediction using a deep learning model loaded with `fastai`. It prepares images for model input, loads a pre-trained model, and provides the model's prediction along with class probabilities for given images, in that case it predict's hair type.
