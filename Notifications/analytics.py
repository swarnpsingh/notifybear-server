from django.utils import timezone
from datetime import timedelta
from collections import defaultdict
from django.db.models import F, ExpressionWrapper, DurationField, Avg

from .models import UserNotificationState


def calculate_analytics(user, notif_type):

    now = timezone.now()
    start_of_week = now - timedelta(days=7)

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
    ignored = qs.filter(is_read=False).count()
    ignore_rate = (ignored / this_week_count) if this_week_count > 0 else 0.0

    # ========================
    # Weekly activity (Mon-Sun)
    # ========================
    weekday_counts = defaultdict(int)

    # ========================
    # Hourly distribution (NEW)
    # 0-3,3-6,...21-24
    # ========================
    hourly_bins = [0] * 8

    for obj in qs:
        post_time = obj.notification_event.post_time

        weekday_counts[post_time.weekday()] += 1

        hour = post_time.hour
        bucket = hour // 3
        hourly_bins[bucket] += 1

    weekly_activity = [weekday_counts[i] for i in range(7)]

    # ========================
    # Avg response time (REAL)
    # ========================
    response_qs = qs.filter(opened_at__isnull=False).annotate(
        response_time=ExpressionWrapper(
            F("opened_at") - F("notification_event__post_time"),
            output_field=DurationField()
        )
    )

    avg_duration = response_qs.aggregate(avg=Avg("response_time"))["avg"]

    if avg_duration:
        total_seconds = int(avg_duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if hours > 0:
            avg_response = f"{hours}h"
        else:
            avg_response = f"{minutes}m"
    else:
        avg_response = "—"

    # ========================
    # Insights block (NEW)
    # ========================
    peak_day_index = weekly_activity.index(max(weekly_activity)) if this_week_count > 0 else 0
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    insights = {
        "ignorePercent": round(ignore_rate, 2),
        "peakDay": days[peak_day_index],
        "totalNotifications": this_week_count
    }

    return {
        "thisWeekCount": this_week_count,
        "ignoreRate": round(ignore_rate, 2),
        "avgResponse": avg_response,
        "weeklyActivity": weekly_activity,
        "timeDistribution": hourly_bins,
        "insights": insights,
    }