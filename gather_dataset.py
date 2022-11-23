import cv2
import mediapipe as mp
import numpy as np

# import keyboard

gesture = {
    0:'ga', 1:'na', 2:'da', 3:'ra', 4:'ma', 5:'ba', 6:'sa', 7:'a', 8:'ja', 9:'ha',
    10:'aa', 11:'ou', 12:'yoe', 13:'o', 14:'u', 15:'eu', 16:'lee', 17:'ae', 18:'e', 19:'space', 20:'clear', 21:'next'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# Dataset
file = np.genfromtxt("data/gesture_train.csv", delimiter=',')
# file = file.transpose()
# print(file)

cap = cv2.VideoCapture(0)

# Click event
def click(event, x, y, flags, param):
    global data, file
    
    if event == cv2.EVENT_MOUSEWHEEL:
        file = np.vstack((file,data))
        print(file)

cv2.namedWindow('Translator')
cv2.setMouseCallback('Translator', click)

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

            # Get angle
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] 
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] 
            v = v2 - v1 # [20,3]
            
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # Normalize
            
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # if keyboard.is_pressed('a'):q
                # for data in angle:
            data = np.array([angle], dtype=np.float32)
            data = np.append(data, 21)

            # print([file])
            # print(data)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Translator', img)
        if cv2.waitKey(1) == ord('q'):
            break

np.savetxt('data/gesture_train_fy.csv', file, fmt='%.5f', delimiter=',')