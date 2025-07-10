from django.urls import path
from . import views

app_name = 'sign_app'

urlpatterns = [
    path('realtime/', views.realtime_view, name='realtime'),
    path('continuous/', views.continuous_view, name='continuous'),
    path('animation/', views.animation_view, name='animation'),
    path('technology/', views.technology_view, name='technology'),
    path('about/', views.about_view, name='about'),
    path('recognize/', views.recognize_gesture, name='recognize'),
    path('get_gesture_info/', views.get_gesture_info, name='gesture-info'),
    path('generate_animation/', views.generate_animation, name='generate-animation'),  # 新增
    path('get_animation_history/', views.get_animation_history, name='get-animation-history'),
    path('clear_history/', views.clear_history, name='clear-history'),
    path('image_recognition/', views.image_recognition_view, name='image_recognition'),
    path('video_recognition/', views.video_recognition_view, name='video_recognition'),
    path('process_image/', views.handle_image_recognition, name='process_image'),
    path('process_video/', views.handle_video_recognition, name='process_video'),

]