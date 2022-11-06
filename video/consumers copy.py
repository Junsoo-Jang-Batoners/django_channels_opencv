# video/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import cv2
import base64
import numpy as np
import torch
import pickle
import gzip
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(17*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose[:34], lh, rh]).tolist()

# global variables for mediapipe process
infdata = {}
infdata['signer'] = 'patient_tester'
infdata['name'] = ''
infdata['gloss'] = ''
infdata['text'] = ''
sign = []
former_keypoints =  [0 for _ in range(118)]
quit_cnt = 0
flag = False

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['v_name']
        self.room_group_name = 'video_%s' % self.room_name
        # print(self.room_name)
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'video_message',
                'message': text_data,
            }
        )

    # Receive message from room group
    async def video_message(self, event):
        global infdata
        global sign
        global former_keypoints
        global flag
        global quit_cnt
        global filenamenumber
        # print(self.scope['user'])
        # print(self.scope['user'].id)
        # print(self.scope['user'].username)
        message = event['message']

        # base64 data
        print(message)
        if message[0] == '"' and message[-1] == '"':
            message = message[1:-1]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))

        # mediapipe process
        # base64data from client
        base64data = message[22:]
        img_raw = np.frombuffer(base64.b64decode(base64data), np.uint8)
        img = cv2.imdecode(img_raw, cv2.IMREAD_UNCHANGED)
        # mediapipe process
        frame_mp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_mp)
        keypoints = extract_keypoints(results)

        # for none zero-keypoints
        if not results.pose_landmarks:
            keypoints[:34] = former_keypoints[:34]
        if not results.left_hand_landmarks:
            keypoints[34:76] = former_keypoints[34:76]
        if not results.right_hand_landmarks:
            keypoints[76:] = former_keypoints[76:]
        
        if flag:
            sign.append(keypoints)

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
            # out = cv2.VideoWriter('inference/data/DCBA/rt_inf_' + str(filecnt) + '.avi', fourcc, 30.0, (int(width), int(height)))
            print('restart')

        if quit_cnt >= 30 and flag:
            # 프레임 카운터 초기화
            quit_cnt = 0
            # 촬영 상태 off
            flag = False
            # 지금까지 저장된 sign들 tensor로 변경하여 저장
            sign = torch.Tensor(sign)
            print(sign)
            print(infdata)
            infdata['sign'] = sign
            filename = self.scope['user']
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f'{filename} is saved')

            
            # filename 을 server로 보내 inference 결과를 얻어온다.

            # inference_result = ''

            # print(f'추론 결과: {inference_result}')


            # 새로 sign 저장하기 위해 sign은 초기화
            sign = []
            # 파일명 새로 지정하기 위해 번호 +1 하여 파일명 새로 부여
            filenamenumber += 1
            filename = 'inference/data/DCBA/rt_inf_' + str(filenamenumber) + '.inference'

        # decoded_data = base64.b64decode(message)
        # print(decoded_data)
        # np_data = np.fromstring(decoded_data, np.uint8)
        # img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
        # cv2.imshow("test", img)


