import cv2 as cv    # OpenCV import
import numpy as np  # 행렬(# 마우스 이벤트 콜백함수 정의

def mouse_callback(event, x, y, flags, param): 
    print("마우스 이벤트 발생, x:", x ," y:", y) # 이벤트 발생한 마우스 위치 출력

img = np.zeros((256, 256, 3), np.uint8)  # 행렬 생성, (가로, 세로, 채널(rgb)),bit)

cv.namedWindow('image')  #마우스 이벤트 영역 윈도우 생성

cv.setMouseCallback('image', mouse_callback)

while(True):

    cv.imshow('image', img)

    k = cv.waitKey(1) & 0xFF
    if k == 27:    # ESC 키 눌러졌을 경우 종료
        print("ESC 키 눌러짐")
        break
cv.destroyAllWindows()