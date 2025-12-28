from django.urls import path
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes

from .views import (
    get_user_notifications,
    ingest_notification,
    ingest_interaction,
    apps_list,
    stats_today,
    stats_range,
    delete_notification,
    mark_notification_opened,
    mark_notification_dismissed,
    unread_count,
)

urlpatterns = [
    # -------------------------
    # Existing routes
    # -------------------------
    path(
        "get/",
        permission_classes([IsAuthenticated])(get_user_notifications),
        name="get_user_notifications"
    ),

    # -------------------------
    # Ingest endpoints (from Android client)
    # -------------------------
    path(
        "ingest/notification/",
        permission_classes([IsAuthenticated])(ingest_notification),
        name="ingest_notification"
    ),
    path(
        "ingest/interaction/",
        permission_classes([IsAuthenticated])(ingest_interaction),
        name="ingest_interaction"
    ),

    # -------------------------
    # Notification state management
    # -------------------------
    path(
        "<int:notification_id>/mark_opened/",
        permission_classes([IsAuthenticated])(mark_notification_opened),
        name="mark_notification_opened"
    ),
    path(
        "<int:notification_id>/mark_dismissed/",
        permission_classes([IsAuthenticated])(mark_notification_dismissed),
        name="mark_notification_dismissed"
    ),
    path(
        "delete/",
        permission_classes([IsAuthenticated])(delete_notification),
        name="delete_notification"
    ),

    # -------------------------
    # App & Statistics endpoints
    # -------------------------
    path(
        "apps/",
        permission_classes([IsAuthenticated])(apps_list),
        name="apps_list"
    ),
    path(
        "stats/today/",
        permission_classes([IsAuthenticated])(stats_today),
        name="stats_today"
    ),
    path(
        "stats/range/",
        permission_classes([IsAuthenticated])(stats_range),
        name="stats_range"
    ),
    path(
        "unread/count/",
        permission_classes([IsAuthenticated])(unread_count),
        name="unread_count"
    ),
]