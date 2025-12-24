from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import timedelta
from django.db.models import Sum

from .models import (
    Notifications,
    App,
    InteractionEvent,
    DailyAggregate,
)

from .serializers import (
    NotificationsSerializer,
    IngestNotificationSerializer,
    IngestInteractionSerializer,
    AppSerializer,
    DailyAggregateSerializer,
)



@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_user_notifications(request):
    notifications = Notifications.objects.filter(user=request.user).order_by("-post_time")
    serializer = NotificationsSerializer(notifications, many=True)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_notification(request):
    serializer = NotificationsSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(user=request.user)
        return Response({"status": "success"})
    return Response(serializer.errors, status=400)


# -------------------------
# NEW: Helper (internal)
# -------------------------

def _get_or_create_app(user, package_name, app_label=""):
    """Ensures an App entry exists for the user + package."""
    app, created = App.objects.get_or_create(
        user=user,
        package_name=package_name,
        defaults={"app_label": app_label or package_name}
    )

    # Update readable label if provided
    if app_label and app.app_label != app_label:
        app.app_label = app_label
        app.save(update_fields=["app_label"])

    return app


# -------------------------
# NEW: Ingest posted notification
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ingest_notification(request):
    """
    Android → Server
    {
        "package_name": "com.whatsapp",
        "app_label": "WhatsApp",
        "notif_key": "0|com.whatsapp|12345",
        "posted_at": "2025-11-12T12:34:56Z",
        "title": "New message",
        "text": "Hey!"
    }
    """
    s = IngestNotificationSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    v = s.validated_data

    app = _get_or_create_app(
        request.user,
        v["package_name"],
        v.get("app_label", "")
    )

    # dedupe using notif_key
    notif_key = v.get("notif_key")

    if notif_key:
        notif, created = Notifications.objects.get_or_create(
            user=request.user,
            notif_key=notif_key,
            defaults=dict(
                package_name=v["package_name"],
                post_time=v.get("posted_at", timezone.now()),
                title=v.get("title", ""),
                text=v.get("text", ""),
            ),
        )
        # If already exists, we WON’T overwrite it — but you can if you want
    else:
        notif = Notifications.objects.create(
            user=request.user,
            package_name=v["package_name"],
            notif_key=None,
            post_time=v.get("posted_at", timezone.now()),
            title=v.get("title", ""),
            text=v.get("text", "")
        )

    return Response({"ok": True}, status=status.HTTP_201_CREATED)


# -------------------------
# NEW: Ingest interaction (click/swipe)
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ingest_interaction(request):
    """
    Android → Server
    {
        "package_name": "com.whatsapp",
        "app_label": "WhatsApp",
        "notif_key": "0|com.whatsapp|12345",
        "removed_at": "2025-11-12T12:35:10Z",
        "raw_reason": 2,
        "interaction_type": "CLICK"   or  "SWIPE"
    }
    """
    s = IngestInteractionSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    v = s.validated_data

    app = _get_or_create_app(
        request.user,
        v["package_name"],
        v.get("app_label", "")
    )

    InteractionEvent.objects.create(
        user=request.user,
        app=app,
        notif_key=v.get("notif_key"),
        removed_at=v["removed_at"],
        raw_reason=v.get("raw_reason"),
        interaction_type=v["interaction_type"],
    )

    # Optionally update original notification timestamps
    notif_key = v.get("notif_key")
    if notif_key:
        nr = Notifications.objects.filter(
            user=request.user, notif_key=notif_key
        ).first()

        if nr:
            if v["interaction_type"] == InteractionEvent.CLICK:
                nr.timestamp_opened = v["removed_at"]
                nr.save(update_fields=["timestamp_opened"])

            elif v["interaction_type"] == InteractionEvent.SWIPE:
                nr.timestamp_dismissed = v["removed_at"]
                nr.save(update_fields=["timestamp_dismissed"])

    return Response({"ok": True}, status=status.HTTP_201_CREATED)


# -------------------------
# NEW: List user's apps
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def apps_list(request):
    apps = App.objects.filter(user=request.user).order_by("app_label")
    serializer = AppSerializer(apps, many=True)
    return Response(serializer.data)


# -------------------------
# NEW: Stats for TODAY
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stats_today(request):
    today = timezone.now().date()

    qs = DailyAggregate.objects.filter(
        user=request.user,
        day=today
    ).select_related("app").order_by("-posts")

    serializer = DailyAggregateSerializer(qs, many=True)
    return Response(serializer.data)


# -------------------------
# NEW: Stats for range (default 7 days)
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stats_range(request):
    days = int(request.GET.get("days", 7))
    start = timezone.now().date() - timedelta(days=days - 1)

    qs = (
        DailyAggregate.objects.filter(
            user=request.user,
            day__gte=start
        )
        .select_related("app")
        .values("app__package_name", "app__app_label")
        .annotate(
            posts=Sum("posts"),
            clicks=Sum("clicks"),
            swipes=Sum("swipes"),
        )
        .order_by("-posts")
    )

    return Response(list(qs))

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def delete_notification(request):
    nid = request.data.get("notification_id")
    Notifications.objects.filter(id=nid, user=request.user).delete()
    return Response({"ok": True})