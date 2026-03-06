from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.utils import timezone
from .models import MobileLog

@api_view(["POST"])
@permission_classes([AllowAny])
def mobile_log(request):
    msg = request.data.get("message")
    tag = request.data.get("tag", "APP")

    MobileLog.objects.create(
        user=request.user if request.user.is_authenticated else None,
        message=msg,
        tag=tag,
        timestamp=timezone.now()
    )

    return Response({"ok": True})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_mobile_logs(request):
    logs = MobileLog.objects.order_by("-timestamp")[:100]
    data = [{"tag": l.tag, "msg": l.message, "time": l.timestamp} for l in logs]
    return Response(data)