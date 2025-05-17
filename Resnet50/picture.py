import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


model = load_model("resnet50_recycling_model.h5")


class_names = ['glass', 'metal', 'other', 'paper', 'plastic']


image_folder = "testData"  


image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for file_name in image_files:
    image_path = os.path.join(image_folder, file_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Hatalı dosya: {file_name}")
        continue


    resized = cv2.resize(img, (224, 224))
    normalized = resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

  
    prediction = model.predict(input_tensor)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"{file_name} → Tahmin: {predicted_class} ({confidence:.2f})")
    cv2.putText(img, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
