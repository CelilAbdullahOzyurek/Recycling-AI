import cv2 as cv

def rescaleFrame(frame, scale=0.50):
    width =int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def changeRes(width, height): # değiştirilecek boyutlar  
    # live video akışını yeniden boyutlandırmak için kullanılır.
    captured.set(3, width)  # 3 = width #3 ve 4 referansları, video boyutunu değiştirmek için kullanılır.
    captured.set(4, height)  # 4 = height


img = cv.imread('test.webp')  

resized_image = rescaleFrame(img, scale=0.5)  # 'image' penceresinde görüntü boyutunu %50 oranında küçültür.
cv.imshow('image', img)
cv.imshow('imageR', resized_image)  # 'image' penceresinde görüntüsünü sağlar
cv.waitKey(0)  # 'e' tuşuna basıldığında döngüden çıkılır.



captured = cv.VideoCapture(0)

# while True:
#     isTrue, frame = captured.read()
#     frame_resized = rescaleFrame(frame, scale=0.75)  # 'Video' penceresinde görüntü boyutunu %75 oranında küçültür.
#     cv.imshow('Video Resized', frame_resized)  # 'Video' penceresinde görüntüsünü sağlar
#     cv.imshow('Video ', frame)
#     if cv.waitKey(20) & 0xFF == ord('a'):   # 'e' tuşuna basıldığında döngüden çıkılır.
#         break
# captured.release()  # başka videoları açabilmek için.
# cv.destroyAllWindows()    # pencereleri kapatır. eAAA