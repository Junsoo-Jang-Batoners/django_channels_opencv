import mediapipe as mp
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import pandas as pd
import json
import pickle
import gzip
import torch

# from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hand = mp.solutions.hands
mp_pose = mp.solutions.pose
prev_time = 0
FPS = 8

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(32*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose[22:34], rh, lh]).tolist()

def extract_keypoints_pose(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(32*2)
    return pose[22:34].tolist()

def extract_keypoints_hand(results):
    if results.multi_handedness:
        if len(results.multi_handedness) == 1:
            hand = np.array([[res.x, res.y] for res in results.multi_hand_landmarks[0].landmark]).flatten()
            label = results.multi_handedness[0].classification[0].label
            if label == 'Left':
                hand = np.zeros(21*2) + hand
            else:
                hand = hand + np.zeros(21*2)
        else:
            rh = np.array([[res.x, res.y] for res in results.multi_hand_landmarks[0].landmark]).flatten()
            lh = np.array([[res.x, res.y] for res in results.multi_hand_landmarks[1].landmark]).flatten()
            hand = rh + lh
        return hand.tolist()

    else:
        return np.zeros(42*2).tolist()
    # print(results.multi_hand_landmarks)
    # print(len(results.multi_hand_landmarks))
    # print(type(results.multi_hand_landmarks))

    # hand = np.array([[res.x, res.y] for res in results.multi_hand_landmarks.landmark]).flatten() if results.multi_hand_landmarks else np.zeros(42*2)
    # # hand = hand.tolist()
    # if results.multi_handedness:
    #     if len(results.multi_handedness) == 1:
    #         label = results.multi_handedness[0].classification[0].label
    #         if label == 'Left':
    #             hand = np.zeros(21*2) + hand
    #         else:
    #             hand = hand + np.zeros(21*2)
    # return hand.tolist()




# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose :
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose :
    # filename = './[mix]P73_U19_N1.mp4'
    filecnt = 1
    filename = 'rt_inf_' + str(filecnt) + '.inference'
    cap = cv2.VideoCapture(0)
    cnt = 0 
    data = {}
    data['name'] = 'realtime'
    data['signer'] = 'realtime'
    data['gloss'] = ''
    data['text'] = ''
    sign = []
    quit_cnt = 0
    flag = True
    former_keypoints = [0 for _ in range(96)]

    # ????????? ???????????? ???????????? ??????
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('testvid'+str(filecnt)+'.avi', fourcc, 30.0, (int(width), int(height)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame ??????
        out.write(frame)

        # ????????? ?????????
        cnt += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print('lh')
        # print(results.left_hand_landmarks)
        # print(type(results.left_hand_landmarks))
        result_hand = hands.process(image)
        result_pose = pose.process(image)
        keypoints = extract_keypoints(results)
        # print('keypoints')
        print(keypoints)
        # print('hand result')
        # print(result_hand)
        # print(result_hand.multi_hand_landmarks)
        print(extract_keypoints_hand(result_hand))

        # print(result_hand.multi_handedness)
        # print(result_hand.multi_handedness[0].classification[0].label)

        # print(type(result_hand.multi_handedness[0].classification.label))
        # print(extract_keypoints_pose(result_pose))
        
        # print(type(keypoints))
        # print(len(keypoints))
        # print(f'{cnt} length : {len(keypoints)}')


        # landmark ????????? ??????
        if not results.pose_landmarks:
            print(f'{cnt} problem of pose')
            keypoints[:12] = former_keypoints[:12]
        if not results.left_hand_landmarks:
            print(f'{cnt} problem of left')
            keypoints[12:54] = former_keypoints[12:54]
        if not results.right_hand_landmarks:
            print(f'{cnt} problem of right')
            keypoints[54:96] = former_keypoints[54:96]

        # ?????? ????????? sign??? keypoints ??????
        if flag:
            sign.append(keypoints)

        # ?????? ???????????? keypoints ?????? ??????
        former_keypoints = keypoints


        # ?????? ??? ???????????? ?????? ????????? ?????? ????????? ?????? ??????????????? ??????
        if not results.left_hand_landmarks and not results.right_hand_landmarks and flag:
            quit_cnt += 1
            # print(quit_cnt)

        # ????????? ????????? ??? ????????? ??? ???????????? quit_cnt??? ?????????
        if quit_cnt and results.left_hand_landmarks and results.right_hand_landmarks and flag:
            quit_cnt = 0

        # ?????? ?????? ???????????? ??? ?????? ???????????? ?????? ????????? ??????????????? ?????? ?????? ON
        if results.left_hand_landmarks and results.right_hand_landmarks and not flag:
            quit_cnt = 0
            flag = True
            print('restart')
        
        # 30 ????????? (1???) ?????? ????????? ????????? ???????????? ?????? ???????????????
        if quit_cnt >= 30 and flag:
            # ????????? ????????? ?????????
            quit_cnt = 0
            # ?????? ?????? off
            flag = False
            # ???????????? ????????? sign??? tensor??? ???????????? ??????
            sign = torch.Tensor(sign)
            data['sign'] = sign
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f'{filename} is saved')

            
            # filename ??? server??? ?????? inference ????????? ????????????.

            inference_result = ''

            print(f'?????? ??????: {inference_result}')


            # ?????? sign ???????????? ?????? sign??? ?????????
            sign = []
            # ????????? ?????? ???????????? ?????? ?????? +1 ?????? ????????? ?????? ??????
            filecnt += 1
            filename = 'test' + str(filecnt) + '.test'

            # ??????????????? frame out release ?????? ??????
            out.release()
            out = cv2.VideoWriter('testvid'+str(filecnt)+'.avi', fourcc, 30.0, (int(width), int(height)))

       
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        
        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)




        cv2.imshow('Mediapipe', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print('END')
            break

    print(f'total frames: {cnt}')
    print(f'this vid\'s playtime: {round(cnt/30, 2)} s')