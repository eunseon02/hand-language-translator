# from operator import index
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image # 한글 출력
import keyboard
import time
# import hangul_utils

gesture = {
    0:'ga', 1:'na', 2:'da', 3:'ra', 4:'ma', 5:'ba', 6:'sa', 7:'a', 8:'ja', 9:'ha',
    10:'aa', 11:'ou', 12:'yoe', 13:'o', 14:'u', 15:'eu', 16:'lee', 17:'ae', 18:'e', 19:'space', 20:'clear', 21:'next'
}

print_gesture = {
    0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ', 7:'ㅇ', 8:'ㅈ', 9:'ㅎ',
    10:'ㅏ', 11:'ㅓ', 12:'ㅕ', 13:'ㅗ', 14:'ㅜ', 15:'ㅡ', 16:'ㅣ', 17:'ㅐ', 18:'ㅔ', 19:'space', 20:'clear', 21:'next'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# Gesture recognition data
file = np.genfromtxt("data/gesture_train.csv", encoding='UTF-8', delimiter=',')
anglefile = file[:,:-1]
labelfile = file[:,-1]
angle = anglefile.astype(np.float32)
label = labelfile.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

###
startTime = time.time() # 시작 시간 지정
prev_index = 0
sentence = ''
recognizeDelay = 1
###
#

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3) 
            idx = int(results[0][0])

            # Draw gesture result
            if idx in gesture.keys(): # gesture중 하나라면 저장된 value출력
                if idx != prev_index:
                    startTime = time.time() # 시작 시간 저장
                    # idx = prev_index
                else:
                    if time.time() - startTime > recognizeDelay: # 1초 이상 
                        if idx == 18: # space 출력
                            sentence += ' '
                        elif idx == 19: # clear 출력
                            sentence = ''
                        # elif idx ==20: # 다음 글자로 넘어감
                            # sentence += gesture[idx]
                        else:
                            # hand_translate += print_gesture[idx]
                            # sentence += hgtk.letter.compose(hand_translate) # 한글자씩 출력됨
                            sentence += print_gesture[idx]
                        startTime = time.time() # 시작시간 초기화
                        print('가나다')

                        # gesture가 출력값_
                cv2.putText(img, text='가나', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Translator', img)
    if cv2.waitKey(1) == ord('q'):
        break