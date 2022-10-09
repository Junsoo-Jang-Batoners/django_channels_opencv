# send_video.py
import asyncio
import websockets
import numpy as np
import json
import cv2
import base64
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(12*2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose[22:34], lh, rh]).tolist()


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('quit')
    quit()
ret, frame = capture.read()
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),95]

# Send video screenshots to the server in real time
async def send_video(websocket):
    global ret, frame
    # global cam
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            # time.sleep(0.1)   
            result, imgencode = cv2.imencode('.jpg', frame, encode_param)
            data = np.array(imgencode)
            img = data.tobytes()
            # base64 encoded transmission
            img = base64.b64encode(img).decode()
            
            await websocket.send("data:image/jpg;base64,"+ img)
			
            # mediapipe
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            keypoints = extract_keypoints(results)

            # logic for saving the file

            ret, frame = capture.read()


async def main_logic():
    async with websockets.connect('ws://127.0.0.1:8000/ws/video/wms/') as websocket:
        await send_video(websocket)

asyncio.get_event_loop().run_until_complete(main_logic())