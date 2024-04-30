from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('generate_questionnaire/', views.generate_questionnaire, name='generate_questionnaire'),
    path('ask_question/', views.ask_question, name='ask_question'),
    path('youtube_notes/', views.convert, name='youtube_notes'),
]
