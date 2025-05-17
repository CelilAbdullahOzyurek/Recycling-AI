import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mobileNetV2/mobilenetv2.h5')

class_names = ['glass', 'metal', 'other', 'paper', 'plastic']

img = cv2.imread('testData/cola.jpg') 
img_resized = cv2.resize(img, (256, 256))
img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0) 

prediction = model.predict(img_input) 
predicted_index = np.argmax(prediction) 
predicted_label = class_names[predicted_index]
confidence = prediction[0][predicted_index]

label = f"{predicted_label} ({confidence*100:.2f}%)"
font = cv2.FONT_HERSHEY_SIMPLEX
output_img = cv2.putText(img_resized.copy(), label, (20, 40), font, 1, (255, 0, 0), 3)

cv2.imshow('Prediction', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
