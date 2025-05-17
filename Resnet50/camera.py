import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("resnet50_recycling_model.h5")


class_names = ['glass', 'metal', 'other', 'paper', 'plastic']


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

  
    img = cv2.resize(frame, (224, 224))              
    img = img.astype("float32") / 255.0              
    img = np.expand_dims(img, axis=0)                

   
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

  
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
