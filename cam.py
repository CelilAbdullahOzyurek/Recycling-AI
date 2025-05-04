import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
  


model = load_model('model2.h5')

class_names = ['glass', 'metal', 'paper', 'plastic', 'other']

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break


    resized_frame = cv.resize(frame, (256, 256))
    img_array = img_to_array(resized_frame)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    
    label = f"{predicted_label.capitalize()} ({confidence*100:.2f}%)"
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(resized_frame, label, (10, 40), font, 1, (255, 0, 0), 3)

 
    cv.imshow('predict', resized_frame)

  
    if cv.waitKey(1) & 0xFF == ord('d'):
        break



cv.destroyAllWindows()
capture.release()
print("closed")