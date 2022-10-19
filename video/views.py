# video/views.py
from django.shortcuts import render

# Create your views here.
def index(request):
	return render(request, 'video/index.html')

def v_name(request, v_name):
	print(request)
	print(request.user.username)
	return render(request, 'video/video.html', {
		'v_name': v_name
	})