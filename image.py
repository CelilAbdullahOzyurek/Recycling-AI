import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')

class_names = ['glass', 'metal', 'other', 'paper', 'plastic']

img_input = cv2.imread('test/gazete.jpg')



img_resized = cv2.resize(img_input, (256, 256))

img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)

prediction = model.predict(img_input)


predicted_class_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_class_index]
confidence = prediction[0][predicted_class_index]
label = f"{predicted_class.capitalize()} ({confidence*100:.2f}%)"

font = cv2.FONT_HERSHEY_SIMPLEX
output_img = cv2.putText(img_resized, label, (20, 40), font, 1, (255, 0, 0), 3)

print("Prediction:", prediction[0])

cv2.imshow('predict', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()