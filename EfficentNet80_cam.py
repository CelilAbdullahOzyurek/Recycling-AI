import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input


model = load_model('EfficentNetB0.h5')
class_names = ['glass', 'metal', 'other', 'paper', 'plastic'] 


capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    
    resized_frame = cv.resize(frame, (256, 256))
    img_array = img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  
    
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

   
    label = f"{predicted_label.capitalize()} ({confidence*100:.2f}%)"
    cv.putText(resized_frame, label, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

   
    cv.imshow('Atık Tahmini', resized_frame)

  
    if cv.waitKey(1) & 0xFF == ord('d'):
        break


capture.release()
cv.destroyAllWindows()
print("closed")
