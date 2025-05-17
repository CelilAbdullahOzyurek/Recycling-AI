import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('mobileNetV2/mobilenetv2.h5') 

class_names = ['glass', 'metal', 'other', 'paper', 'plastic']

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Camera capture failed.")
        break

    display_frame = frame.copy()

    img_resized = cv.resize(frame, (256, 256))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    label = f"{predicted_label} ({confidence*100:.2f}%)"
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(display_frame, label, (10, 40), font, 1, (0, 255, 0), 2)

    cv.imshow('Predict (press D to exit)', display_frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
print("Camera closed")
