# send_video.py
import asyncio
import websockets
import numpy as np
import json
import cv2
import base64
import time
import mediapipe as mp
import torch
import pickle
import gzip
import os
import pandas as pd

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(17*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose[:34], lh, rh]).tolist()

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('quit')
    quit()
ret, frame = capture.read()
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),95]

# Send video screenshots to the server in real time
async def send_video(websocket):
    global ret, frame
    # setting for saving the file

    # global cam
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        filecnt = 1
        filename = 'inference/data/DCBA/rt_inf_' + str(filecnt) + '.inference'
        cnt = 0 
        data = {}
        data['name'] = 'realtime'
        data['signer'] = 'realtime'
        data['gloss'] = ''
        data['text'] = ''
        sign = []
        quit_cnt = 0
        flag = False
        former_keypoints = [0 for _ in range(96)]

        # 비디오 녹화해서 넘겨주기 위해
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('inference/data/DCBA/rt_inf_' + str(filecnt) + '.avi', fourcc, 30.0, (int(width), int(height)))

        while True:
            # time.sleep(0.1)   
            result, imgencode = cv2.imencode('.jpg', frame, encode_param)
            ficture_data = np.array(imgencode)
            img = ficture_data.tobytes()
            # base64 encoded transmission
            img = base64.b64encode(img).decode()
            
            await websocket.send("data:image/jpg;base64,"+ img)
			

            # mediapipe
            cnt += 1
            frame_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_mp)
            keypoints = extract_keypoints(results)

            if not results.pose_landmarks:
                # print(f'{cnt} problem of pose')
                keypoints[:34] = former_keypoints[:34]
            if not results.left_hand_landmarks:
                # print(f'{cnt} problem of left')
                keypoints[34:76] = former_keypoints[34:76]
            if not results.right_hand_landmarks:
                # print(f'{cnt} problem of right')
                keypoints[76:] = former_keypoints[76:]

            if flag:
                sign.append(keypoints)
                out.write(frame)

            # 직전 프레임의 keypoints 따로 저장
            former_keypoints = keypoints

            # 양손 안 보이는데 촬영 상태면 촬영 취소를 위해 종료카운터 세기
            if not results.left_hand_landmarks and not results.right_hand_landmarks and flag:
                quit_cnt += 1
                # print(quit_cnt)

            # 잠깐만 양손이 안 보였던 것 뿐이라면 quit_cnt를 초기화
            if quit_cnt and results.left_hand_landmarks and results.right_hand_landmarks and flag:
                quit_cnt = 0

            # 촬영 중단 상태인데 양 손이 보인다면 종료 카운터 초기화하고 촬영 상태 ON
            if results.left_hand_landmarks and results.right_hand_landmarks and not flag:
                quit_cnt = 0
                flag = True
                out = cv2.VideoWriter('inference/data/DCBA/rt_inf_' + str(filecnt) + '.avi', fourcc, 30.0, (int(width), int(height)))
                print('restart')
            
            # 30 프레임 (1초) 동안 양손이 보이지 않았는데 촬영 상태였다면
            if quit_cnt >= 30 and flag:
                # 프레임 카운터 초기화
                quit_cnt = 0
                # 촬영 상태 off
                flag = False
                # 지금까지 저장된 sign들 tensor로 변경하여 저장
                sign = torch.Tensor(sign)
                print(sign)
                print(data)
                # print(type(sign))
                # print(len(sign))
                # print(len(sign[0]))
                data['sign'] = sign
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                print(f'{filename} is saved')

                
                # filename 을 server로 보내 inference 결과를 얻어온다.

                # inference_result = ''

                # print(f'추론 결과: {inference_result}')


                # 새로 sign 저장하기 위해 sign은 초기화
                sign = []
                # 파일명 새로 지정하기 위해 번호 +1 하여 파일명 새로 부여
                filecnt += 1
                filename = 'inference/data/DCBA/rt_inf_' + str(filecnt) + '.inference'

                # 지금까지의 frame out release 하고 새로
                out.release()

                # await websocket.send()
            
            # printresult = ''

            # logic for saving the file

            ret, frame = capture.read()


async def main_logic():
    async with websockets.connect('ws://127.0.0.1:8000/ws/video/wms/') as websocket:
        await send_video(websocket)

asyncio.get_event_loop().run_until_complete(main_logic())