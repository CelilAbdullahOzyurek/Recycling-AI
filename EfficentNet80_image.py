import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


class_names = ['glass', 'metal', 'other', 'paper', 'plastic']


model = load_model('EfficentNetB0.h5')


img_input = cv2.imread('testData/cola.jpg')  

if img_input is None:
    raise FileNotFoundError("Görsel okunamadı, yolu kontrol et!")


img_resized = cv2.resize(img_input, (256, 256))
img_array = np.expand_dims(img_resized, axis=0)
img_array = preprocess_input(img_array)


prediction = model.predict(img_array)  
predicted_class_index = np.argmax(prediction[0])  
predicted_class = class_names[predicted_class_index]
confidence = prediction[0][predicted_class_index]


label = f"{predicted_class.capitalize()} ({confidence*100:.2f}%)"
output_img = cv2.putText(img_resized.copy(), label, (20, 40),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


print("Softmax Çıktısı:", prediction[0])
cv2.imshow("Tahmin", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
