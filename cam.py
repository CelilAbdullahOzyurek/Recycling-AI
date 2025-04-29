import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
  

# Modeli yükle
model = load_model('modelFinal_4.h5')
capture= cv.VideoCapture(0)
while True:
    # Görüntüyü oku
    ret, frame = capture.read()
    frame=cv.resize(frame, (256, 256))
    test_img = img_to_array(frame)
    test_img = test_img / 255.0
    test_img = np.expand_dims(test_img, axis=0)
    result= model.predict(test_img)
    result[0][0]
    if result[0][0] > 0.5:
        label = f"Paper ({result[0][0]*100:.2f}%)"
    else:
        label = f"Plastic ({(1-result[0][0])*100:.2f}%)"
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, label, (10, 40), font, 1, (255, 0, 0), 3)
    cv.imshow('predict', frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break
cv.destroyAllWindows()
capture.release()
print("closed")