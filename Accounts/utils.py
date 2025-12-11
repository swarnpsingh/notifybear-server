from .models import AuthAuditLog

def log_auth_event(event, request=None, user=None, username=None, detail=None):
    ip = None
    ua = None
    if request is not None:
        # X-Forwarded-For support depends on proxy config
        xff = request.META.get("HTTP_X_FORWARDED_FOR")
        ip = xff.split(",")[0].strip() if xff else request.META.get("REMOTE_ADDR")
        ua = request.META.get("HTTP_USER_AGENT", "")

    AuthAuditLog.objects.create(
        event=event,
        user=user,
        username=username,
        ip=ip,
        user_agent=ua,
        detail=detail or {}
    )
