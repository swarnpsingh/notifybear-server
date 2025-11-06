from django.urls import path
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes

from .views import get_user_notifications, upload_notification

urlpatterns = [
    path("get/", permission_classes([IsAuthenticated])(get_user_notifications), name="get_user_notifications"),
    path("upload/", permission_classes([IsAuthenticated])(upload_notification), name="upload_notification"),
]
