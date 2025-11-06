from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Notifications
from .serializers import NotificationsSerializer


@api_view(["GET"])
def get_user_notifications(request):
    notifications = Notifications.objects.filter(user=request.user).order_by("-post_time")
    serializer = NotificationsSerializer(notifications, many=True)
    return Response(serializer.data)

@api_view(["POST"])
def upload_notification(request):
    serializer = NotificationsSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(user=request.user)
        return Response({"status": "success"})
    return Response(serializer.errors, status=400)
