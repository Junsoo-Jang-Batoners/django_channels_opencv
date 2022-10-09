# chat/routing.py
from django.urls import re_path
from django.urls import path


import chat.consumers
import video.consumers

websocket_urlpatterns = [
    path('ws/<int:id>/', chat.consumers.ChatConsumer.as_asgi()),
    # re_path(r'ws/chat/(?P<room_name>\w+)/$', chat.consumers.ChatConsumer.as_asgi()),
	re_path(r'ws/video/(?P<v_name>\w+)/$', video.consumers.VideoConsumer.as_asgi())
]
