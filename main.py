from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

app = FastAPI()

# Load the model
model_path = 'cnn_model.h5'
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at path: {model_path}")

try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Define optimal thresholds for each class
optimal_thresholds = [
    (0, 0.7859, 0.8284),  # Class 0: covid
    (1, 0.3048, 0.4046),  # Class 1: normal
    (2, 0.3684, 0.4341)   # Class 2: pneumonia
]

# Define class names
class_names = {
    0: "covid",
    1: "normal",
    2: "pneumonia"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        img = image.load_img(io.BytesIO(contents), target_size=(256, 256))  # Adjust target_size as needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Get model predictions
        predictions = model.predict(img_array)

        # Convert predictions to standard Python lists
        predictions_list = predictions[0].tolist()

        # Apply optimal thresholds
        prediction_labels = []
        for i, (class_idx, lower_threshold, upper_threshold) in enumerate(optimal_thresholds):
            class_prediction = predictions_list[class_idx]
            if lower_threshold <= class_prediction <= upper_threshold:
                prediction_labels.append(class_names[class_idx])

        # If no predictions fall within the optimal thresholds, choose the class with the highest prediction value
        if not prediction_labels:
            prediction_labels.append(class_names[int(np.argmax(predictions_list))])

        return JSONResponse(content={
            "predictions": prediction_labels,
            "raw_predictions": predictions_list
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server. Use /predict to upload an image and get predictions."}
