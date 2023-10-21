import cv2
import tensorflow as tf

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    # mirror the image
    
    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()