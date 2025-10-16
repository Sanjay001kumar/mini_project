from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load model
model = load_model("trained_data_model.h5")

# Classes your model predicts
classes = ["10", "20", "50", "100", "200", "500"]

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction_text="⚠️ No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction_text="⚠️ No file selected")

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((192, 192))  # Match training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 192, 192, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Note: {predicted_class}",
        uploaded_image=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)
