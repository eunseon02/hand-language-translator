import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def click(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEHWHEEL:
        print("dfdsfsvgdgdgdgd")
       
cv2.namedWindow('Translator')

cv2.setMouseCallback('Translator', click)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    
    cv2.imshow('Translator', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()