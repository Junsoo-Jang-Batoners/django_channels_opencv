# chat/urls.py
from django.urls import path

from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.index, name='home'),
    path('chat/<str:username>/', views.chatPage, name='chat'),    
    path('inference/', views.inferenceCall, name='inference')
]

    # path('', views.index, name='index'),
    # path('<str:room_name>/', views.room, name='room'),
