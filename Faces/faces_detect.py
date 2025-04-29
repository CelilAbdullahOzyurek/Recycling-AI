import cv2 as cv

haar_cascade = cv.CascadeClassifier('Faces/haarcascade_frontalface_default.xml')

img = cv.imread('faces/test2.webp')
if img is None:
    print("Image not found!")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

if haar_cascade.empty():
    print("Cascade xml file not loaded!")
    exit()

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)
cv.waitKey(0)
cv.destroyAllWindows()
