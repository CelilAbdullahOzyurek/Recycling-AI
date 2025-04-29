import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('modelFinal_41.h5')

img = cv2.imread('burPlastic.jpg')
img_resized = cv2.resize(img, (256, 256))

img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)

prediction = model.predict(img_input)

probability = prediction[0][0]

if probability > 0.5:
    label = f"Paper ({probability*100:.2f}%)"
else:
    label = f"Plastic ({(1-probability)*100:.2f}%)"

font = cv2.FONT_HERSHEY_SIMPLEX


output_img = cv2.putText(img_resized, label, (20, 40), font, 1, (255, 0, 0), 3)

cv2.imshow('predict', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()