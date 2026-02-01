from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_files, name='upload_files'),
    path('download_excel/<str:filename>/', views.download_excel, name='download_excel'),
    path('cleanup_excel/<str:filename>/', views.cleanup_excel, name='cleanup_excel'),
    path('results/', views.show_results, name='show_results'),
    path('cleanup/', views.cleanup, name='cleanup'),
    path('health/', views.health_check, name='health_check'),
]

