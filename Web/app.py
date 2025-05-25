from flask import Flask, render_template, request, redirect, url_for ,Response, jsonify
import threading
import os
import uuid
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2



app = Flask(__name__, static_folder="static")


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_PREDICT = os.path.join(BASE_DIR, "static", "uploads", "predictions")
UPLOAD_FOLDER_CONTRIB = os.path.join(BASE_DIR, "static", "uploads", "contributions")
MODEL_PATH = os.path.join(BASE_DIR, "efficientnet.h5")
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")


class_names = ['glass', 'metal', 'other', 'paper', 'plastic']

model = load_model(MODEL_PATH)

os.makedirs(UPLOAD_FOLDER_PREDICT, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_CONTRIB, exist_ok=True)

@app.route('/')
def home():
    return render_template("layout.html", page="home")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    label = confidence = filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER_PREDICT, filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (256, 256))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            prediction = model.predict(img)
            index = np.argmax(prediction)
            label = class_names[index]
            confidence = float(prediction[0][index]) * 100

    return render_template("layout.html", page="predict", filename=filename, label=label, confidence=confidence)

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    if request.method == 'POST':
        file = request.files['image']
        label = request.form['label']
        if file and label:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER_CONTRIB, filename)
            file.save(filepath)

            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, datetime.now().isoformat(), label, request.remote_addr])

            return redirect(url_for('contribute'))
    return render_template("layout.html", page="contribute", class_names=class_names)


if __name__ == '__main__':
    app.run(debug=True)
