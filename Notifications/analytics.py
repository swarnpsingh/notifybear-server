from django.utils import timezone
from datetime import timedelta
from collections import defaultdict

from .models import UserNotificationState


def calculate_analytics(user, notif_type):

    now = timezone.now()
    start_of_week = now - timedelta(days=now.weekday())

    qs = UserNotificationState.objects.filter(
        user=user,
        notification_event__type=notif_type,
        notification_event__post_time__gte=start_of_week
    ).select_related("notification_event")

    # ========================
    # This week count
    # ========================
    this_week_count = qs.count()

    # ========================
    # Ignore rate
    # ========================
    total = qs.count()
    ignored = qs.filter(is_read=False).count()

    ignore_rate = (ignored / total) if total > 0 else 0.0

    # ========================
    # Weekly activity (Mon-Sun)
    # ========================
    weekday_counts = defaultdict(int)

    for obj in qs:
        day = obj.notification_event.post_time.weekday()  # 0=Mon
        weekday_counts[day] += 1

    weekly_activity = [weekday_counts[i] for i in range(7)]

    # ========================
    # Avg response (dummy example)
    # ========================
    avg_response = "1h"

    return {
        "thisWeekCount": this_week_count,
        "ignoreRate": round(ignore_rate, 2),
        "avgResponse": avg_response,
        "weeklyActivity": weekly_activity,
    }