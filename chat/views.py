# chat/views.py
from django.shortcuts import render, get_object_or_404
from django.contrib.auth import get_user_model
from chat.models import ChatModel
from django.http import JsonResponse
import json

# from inference.signjoey.inferencecalltest import inferencetestcall
from inference.signjoey.inference import inference

cfg_file = 'inference/configs/sign_dcba_inf.yaml'

ckpt = None

filecnt = 1
# Create your views here.

def match_avatar(message):
    if message == '어디가 아프세요?' or message == '어디가 아프세요':
        return 'slavatar/72.mp4'
    elif message == '어떻게 아프세요?' or message == '어떻게 아프세요':
        return 'slavatar/73.mp4'
    elif message == '어떻게 다치게 되셨나요?' or message == '어떻게 다치셨나요?':
        return 'slavatar/74.mp4'
    elif message == '통증 부위 좀 볼게요' or message == '통증 부위 좀 볼게요.':
        return 'slavatar/75.mp4'
    elif message == '힘 빼고 긴장 풀으세요' or message == '힘 빼고 긴장 풀으세요.':
        return 'slavatar/98.mp4'
    elif message == '엑스레이 검사 할게요' or message == '엑스레이 검사 할게요.':
        return 'slavatar/122.mp4'
    elif message == '뼈에는 이상 없습니다' or message == '뼈에는 이상 없습니다.':
        return 'slavatar/131.mp4'
    elif message == '물리치료 하러 오셔야 합니다' or message == '물리치료 하러 오셔야 합니다.' or message == '물리치료 받으러 오세요' or message == '물리치료 받으러 오세요.':
        return 'slavatar/135.mp4'


User = get_user_model()

def index(request):
    users = User.objects.exclude(username=request.user.username)
    return render(request, 'chat/index.html', context={'users': users})

def chatPage(request, username):
    # user_obj = User.objects.get(username=username)
    user_obj = get_object_or_404(User, username=username)
    # users = User.objects.exclude(username=request.user.username)

    if request.user.id > user_obj.id:
        thread_name = f'chat_{request.user.id}-{user_obj.id}'
    else:
        thread_name = f'chat_{user_obj.id}-{request.user.id}'
    message_objs = ChatModel.objects.filter(thread_name=thread_name)
    if request.user.id % 2:
        user_class = 'doctor'
    else:
        user_class = 'patient'
    return render(request, 'chat/main_chat.html', context={'user': user_obj, 'messages': message_objs, 'user_class': user_class})

def inferenceCall(request):
    global filecnt 

    # user_obj = get_object_or_404(User, username=username)
    if request.method == 'GET':
        return render(request, 'chat/inference_test.html', context={'inference_result': 'Not allowed connection'})
    elif request.method == 'POST':
        print(filecnt)

        inffilename = 'DCBA/rt_inf_' + str(filecnt) + '.inference'

        inference_result = inference(cfg_file=cfg_file, ckpt=ckpt, inf_dir=inffilename)

        filecnt += 1

        jsonObject = json.loads(request.body)
        jsonObject['message'] = inference_result
        print(jsonObject)
        # return render(request, 'chat/main_chat.html', context={'inference_result': inference_result})
        return JsonResponse(jsonObject)

def avatarCall(request):
    if request.method == 'GET':
        return render(request, 'chat/inference_test.html', context={'inference_result': 'Not allowed connection'})
    elif request.method == 'POST':
        jsonObject = json.loads(request.body)
        message = jsonObject['message'] 
        jsonObject['vid_dir'] = match_avatar(message)
        return JsonResponse(jsonObject)


def STTCall(request):
    if request.method == 'GET':
        return render(request, 'chat/inference_test.html', context={'inference_result': 'Not allowed connection'})
    elif request.method == 'POST':
        pass
        

# def room(request, room_name):
#     return render(request, 'chat/room.html', {
#         'room_name': room_name
#     })

# def index(request):
#     return render(request, 'chat/index.html', {})
