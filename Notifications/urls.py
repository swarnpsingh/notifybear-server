from django.urls import path
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes

from .views import (
    get_user_notifications,
    upload_notification,
    ingest_notification,
    ingest_interaction,
    apps_list,
    stats_today,
    stats_range,
)

urlpatterns = [
    # Existing routes
    path("get/",permission_classes([IsAuthenticated])(get_user_notifications),name="get_user_notifications"),
    path("upload/",permission_classes([IsAuthenticated])(upload_notification),name="upload_notification"),

    # New: ingest posted notifications from Android
    path("ingest/notification/",permission_classes([IsAuthenticated])(ingest_notification),name="ingest_notification"),

    # New: ingest interactions (CLICK / SWIPE)
    path("ingest/interaction/",permission_classes([IsAuthenticated])(ingest_interaction),name="ingest_interaction"),

    # New: List apps a user has notifications from
    path("apps/",permission_classes([IsAuthenticated])(apps_list),name="apps_list"),

    # New: Stats for today
    path("stats/today/",permission_classes([IsAuthenticated])(stats_today),name="stats_today"),

    # New: Stats for last X days (default 7)
    path("stats/range/",permission_classes([IsAuthenticated])(stats_range),name="stats_range"),
]
