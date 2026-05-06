from django.contrib import admin
from django.urls import path
from temporal_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home),
    path('analyze/', views.analyze),
]