from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import timedelta
from django.db.models import Sum, Prefetch

from .models import (
    NotificationEvent,
    UserNotificationState,
    App,
    InteractionEvent,
    DailyAggregate,
)

from .serializers import (
    UserNotificationStateSerializer,
    NotificationEventSerializer,
    IngestNotificationSerializer,
    IngestInteractionSerializer,
    AppSerializer,
    DailyAggregateSerializer,
)


# -------------------------
# Helper function (internal)
# -------------------------
import hashlib

def compute_hash(v):
    parts = [
        v.get("title", ""),
        v.get("text", ""),
        v.get("big_text", ""),
        v.get("sub_text", ""),
        v.get("summary_text", ""),
    ]
    s = "|".join(str(p) for p in parts)
    return hashlib.sha256(s.encode()).hexdigest()

def _get_or_create_app(user, package_name, app_label=""):
    """Ensures an App entry exists for the user + package."""
    app, created = App.objects.get_or_create(
        user=user,
        package_name=package_name,
        defaults={"app_label": app_label or package_name}
    )

    # Update readable label if provided and different
    if app_label and app.app_label != app_label:
        app.app_label = app_label
        app.save(update_fields=["app_label"])

    return app


# -------------------------
# Get user's notifications (with state)
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_user_notifications(request):
    """
    Returns all notifications for the user with their interaction state.
    Ordered by most recent first.
    """
    # Get user notification states with related data
    states = UserNotificationState.objects.filter(
        user=request.user
    ).select_related(
        'notification_event',
        'notification_event__app'
    ).order_by('-notification_event__post_time')
    
    serializer = UserNotificationStateSerializer(states, many=True)
    return Response(serializer.data)


# @api_view(["POST"])
# @permission_classes([IsAuthenticated])
# def upload_notification(request):
#     """
#     Legacy endpoint - creates notification event and user state.
#     Consider migrating clients to use ingest_notification instead.
#     """
#     serializer = IngestNotificationSerializer(data=request.data)
#     if serializer.is_valid():
#         v = serializer.validated_data
        
#         # Get or create app
#         app = _get_or_create_app(
#             request.user,
#             v["package_name"],
#             v.get("app_label", "")
#         )
        
#         # Create notification event
#         notif_event = NotificationEvent.objects.create(
#             app=app,
#             notif_key=v.get("notif_key", f"legacy_{timezone.now().timestamp()}"),
#             title=v.get("title", ""),
#             text=v.get("text", ""),
#             post_time=v.get("posted_at", timezone.now())
#         )
        
#         # Create user state
#         UserNotificationState.objects.create(
#             user=request.user,
#             notification_event=notif_event
#         )
        
#         return Response({"status": "success"}, status=status.HTTP_201_CREATED)
    
#     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# -------------------------
# Ingest posted notification
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ingest_notification(request):
    s = IngestNotificationSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    v = s.validated_data

    # Get or create app
    app = _get_or_create_app(
        request.user,
        v["package_name"],
        v.get("app_label", "")
    )

    # Dedupe using (app, notif_key)
    notif_key = v.get("notif_key")
    
    if not notif_key:
        # Generate unique key if not provided
        notif_key = f"auto_{app.id}_{timezone.now().timestamp()}"

    # Try to get existing notification event
    notif_event, created = NotificationEvent.objects.get_or_create(
        app=app,
        notif_key=notif_key,
        defaults={
            "post_time": v.get("posted_at", timezone.now()),
            "title": v.get("title", ""),
            "text": v.get("text", ""),
            "big_text": v.get("big_text", ""),
            "sub_text": v.get("sub_text", ""),
            "summary_text": v.get("summary_text", ""),
            "info_text": v.get("info_text", ""),
            "text_lines": v.get("text_lines", ""),
            "channel_id": v.get("channel_id", ""),
            "conversation_title": v.get("conversation_title", ""),
            "people": v.get("people"),
            "large_icon_base64": v.get("large_icon_base64"),
            "picture_base64": v.get("picture_base64"),
            "content_hash": compute_hash(v),
        }
    )

    state, _ = UserNotificationState.objects.get_or_create(
        user=request.user,
        notification_event=notif_event
    )

    return Response(
        {
            "ok": True,
            "created": created,
            "state_id": state.id
        },
        status=status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )


# -------------------------
# Ingest interaction (click/swipe)
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ingest_interaction(request):
    """
    Android → Server
    Logs interaction event and updates UserNotificationState.
    
    Payload:
    {
        "package_name": "com.whatsapp",
        "app_label": "WhatsApp",
        "notif_key": "0|com.whatsapp|12345",
        "removed_at": "2025-11-12T12:35:10Z",
        "raw_reason": 2,
        "interaction_type": "CLICK"  # or "SWIPE"
    }
    """
    s = IngestInteractionSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    v = s.validated_data

    # Get or create app
    app = _get_or_create_app(
        request.user,
        v["package_name"],
        v.get("app_label", "")
    )

    # Find the notification event
    notif_key = v.get("notif_key")
    if not notif_key:
        return Response(
            {"error": "notif_key is required for interaction tracking"},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        notif_event = NotificationEvent.objects.get(
            app=app,
            notif_key=notif_key
        )
    except NotificationEvent.DoesNotExist:
        return Response(
            {"error": "Notification not found. Ensure notification was ingested first."},
            status=status.HTTP_404_NOT_FOUND
        )

    # Create interaction event (append-only)
    InteractionEvent.objects.create(
        user=request.user,
        notification_event=notif_event,
        interaction_type=v["interaction_type"],
        timestamp=v["removed_at"],
        raw_reason=v.get("raw_reason"),
    )

    # Update user notification state
    state, created = UserNotificationState.objects.get_or_create(
        user=request.user,
        notification_event=notif_event
    )

    if v["interaction_type"] == InteractionEvent.CLICK:
        if not state.opened_at:  # Only set if not already opened
            state.mark_opened(timestamp=v["removed_at"])
    
    elif v["interaction_type"] == InteractionEvent.SWIPE:
        if not state.dismissed_at:  # Only set if not already dismissed
            state.mark_dismissed(timestamp=v["removed_at"])

    return Response({"ok": True}, status=status.HTTP_201_CREATED)


# -------------------------
# Mark notification as opened
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_notification_opened(request, notification_id):
    """
    Mark a specific notification as opened.
    URL: POST /notifications/{notification_id}/mark_opened/
    """
    try:
        state = UserNotificationState.objects.get(
            user=request.user,
            notification_event_id=notification_id
        )
        state.mark_opened()
        return Response({"status": "opened", "opened_at": state.opened_at})
    except UserNotificationState.DoesNotExist:
        return Response(
            {"error": "Notification not found"},
            status=status.HTTP_404_NOT_FOUND
        )


# -------------------------
# Mark notification as dismissed
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def mark_notification_dismissed(request, notification_id):
    """
    Mark a specific notification as dismissed.
    URL: POST /notifications/{notification_id}/mark_dismissed/
    """
    try:
        state = UserNotificationState.objects.get(
            user=request.user,
            notification_event_id=notification_id
        )
        state.mark_dismissed()
        return Response({"status": "dismissed", "dismissed_at": state.dismissed_at})
    except UserNotificationState.DoesNotExist:
        return Response(
            {"error": "Notification not found"},
            status=status.HTTP_404_NOT_FOUND
        )


# -------------------------
# List user's apps
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def apps_list(request):
    """
    Returns all apps that have sent notifications to this user.
    """
    apps = App.objects.filter(user=request.user).order_by("app_label")
    serializer = AppSerializer(apps, many=True)
    return Response(serializer.data)


# -------------------------
# Stats for TODAY
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stats_today(request):
    """
    Returns aggregated stats for today, per app.
    """
    today = timezone.now().date()

    qs = DailyAggregate.objects.filter(
        user=request.user,
        day=today
    ).select_related("app").order_by("-posts")

    serializer = DailyAggregateSerializer(qs, many=True)
    return Response(serializer.data)


# -------------------------
# Stats for date range
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stats_range(request):
    """
    Returns aggregated stats for the past N days (default 7).
    Query param: ?days=7
    """
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


# -------------------------
# Delete notification (soft delete approach)
# -------------------------

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def delete_notification(request):
    """
    Deletes a notification event and its associated state.
    Note: This is a hard delete. Consider soft delete in production.
    
    Payload: {"notification_id": 123}
    """
    notification_id = request.data.get("notification_id")
    
    if not notification_id:
        return Response(
            {"error": "notification_id is required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Delete user state (will cascade delete if needed)
    deleted_count, _ = UserNotificationState.objects.filter(
        user=request.user,
        notification_event_id=notification_id
    ).delete()
    
    if deleted_count > 0:
        return Response({"ok": True, "deleted": True})
    else:
        return Response(
            {"error": "Notification not found"},
            status=status.HTTP_404_NOT_FOUND
        )


# -------------------------
# Get unread notifications count
# -------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def unread_count(request):
    """
    Returns count of unread notifications.
    """
    count = UserNotificationState.objects.filter(
        user=request.user,
        is_read=False
    ).count()
    
    return Response({"unread_count": count})

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def update_notification_state(request):
    state_id = request.data.get("state_id")
    try:
        ml_score = int(request.data.get("ml_score"))
    except (TypeError, ValueError):
        return Response({"error": "ml_score must be integer"}, status=400)


    if state_id is None or ml_score is None:
        return Response({"error": "state_id and ml_score required"}, status=400)

    try:
        state = UserNotificationState.objects.get(id=state_id, user=request.user)
    except UserNotificationState.DoesNotExist:
        return Response({"error": "Invalid state"}, status=404)
    if state.ml_score != ml_score:
        state.ml_score = ml_score
        state.save(update_fields=["ml_score"])

    return Response({"ok": True})