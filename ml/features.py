import numpy as np
from django.utils import timezone
from . import config

from Notifications.models import (
    NotificationEvent,
    UserNotificationState,
    InteractionEvent,
)


class FeatureEngineer:
    @staticmethod
    def extract(notification, user_stats, context):
        """
        Converts a notification + stats into the highly optimized 8-feature V5 vector.
        """
        dt = notification.post_time
        title = notification.title or ""
        text = notification.text or ""
        full_text = f"{title} {text}".lower().strip()
        
        # --- 1. Linear Time ---
        raw_hour = dt.hour + (dt.minute / 60.0)
        
        # --- 2. Channel Micro-Reputation (Smoothed) ---
        channel_id = getattr(notification, 'channel_id', 'default')
        channel_key = f"channel_{notification.app_id}_{channel_id}"
        
        ch_sent = user_stats.get(f"{channel_key}_sent", 0)
        ch_clicks = user_stats.get(f"{channel_key}_clicks", 0)
        
        # Fallback to overall app CTR if the channel is entirely new
        app_ctr = user_stats.get(f"app_{notification.app_id}_ctr", config.GLOBAL_OPEN_RATE_PRIOR)
        
        numerator = ch_clicks + (config.SMOOTHING_WEIGHT * app_ctr)
        denominator = ch_sent + config.SMOOTHING_WEIGHT
        channel_historical_ctr = numerator / denominator

        # --- 3. Stateful Context ---
        sec_since_action = context.get("sec_since_last_action", 86400.0)
        is_active_session = 1.0 if sec_since_action < 180.0 else 0.0
        
        time_since_last_notif_sec = context.get("time_since_last_notif_sec", 86400.0)

        # --- 4. Structural Meta-Features ---
        text_len = len(full_text)
        
        digit_count = sum(c.isdigit() for c in full_text)
        digit_density = digit_count / (text_len + 1e-5)
        
        title_body_ratio = len(title) / (len(text) + 1e-5)
        
        excl_count = full_text.count('!')
        exclamation_density = excl_count / (text_len + 1e-5)
        
        # --- 5. Rolling Volume ---
        notifs_past_24h = user_stats.get("notifs_past_24h", 1.0)

        # --- Return Vector ---
        # MUST exactly match config.FEATURE_NAMES order
        return np.array([
            raw_hour, 
            channel_historical_ctr, 
            is_active_session, 
            time_since_last_notif_sec, 
            digit_density, 
            title_body_ratio, 
            exclamation_density, 
            notifs_past_24h
        ], dtype=np.float32)

    @staticmethod
    def fetch_training_rows(user, apps=None, lookback_days=30):
        """
        Yield tuples (feature_vector, label) for the given user.

        Rules enforced:
        - Labels computed via ml.service.service.calculate_label
        - Rows with label None are dropped
        - Uses Django ORM to fetch NotificationEvent and UserNotificationState
        - `apps` may be a list of app ids or package_name strings
        """
        from .service import service as priority_service
        from datetime import timedelta
        from django.utils import timezone

        cutoff = timezone.now() - timedelta(days=lookback_days)

        qs = NotificationEvent.objects.select_related('app').filter(app__user=user, post_time__gte=cutoff)

        if apps:
            # try to detect if apps elements are ints (ids) or strings (package names)
            if all(isinstance(a, int) for a in apps):
                qs = qs.filter(app_id__in=apps)
            else:
                qs = qs.filter(app__package_name__in=apps)

        # Order chronologically so context calculations are simpler
        qs = qs.order_by('post_time')

        # For performance we prefetch related states
        from django.db.models import Prefetch
        user_states = UserNotificationState.objects.filter(user=user)
        qs = qs.prefetch_related(Prefetch('user_states', queryset=user_states, to_attr='__user_states'))

        for notif in qs.iterator():
            # Locate user state if present
            user_state = None
            if hasattr(notif, '__user_states') and notif.__user_states:
                # There should be at most one per (user, notification_event)
                user_state = notif.__user_states[0]

            sent_time = notif.post_time
            opened_time = getattr(user_state, 'opened_at', None) if user_state else None
            dismissed_time = getattr(user_state, 'dismissed_at', None) if user_state else None

            label = priority_service.calculate_label(sent_time, opened_time, dismissed_time)
            # Drop pending/unknown labels
            if label is None:
                continue

            # Build minimal user_stats used by FeatureEngineer.extract
            channel_id = getattr(notif, 'channel_id', 'default')
            channel_key = f"channel_{notif.app_id}_{channel_id}"

            recent_events = NotificationEvent.objects.filter(
                app=notif.app,
                channel_id=channel_id,
                post_time__gte=cutoff
            )
            ch_sent = recent_events.count()

            ch_clicks = UserNotificationState.objects.filter(
                notification_event__in=recent_events,
                opened_at__isnull=False
            ).count()

            app_events = NotificationEvent.objects.filter(app=notif.app, post_time__gte=cutoff)
            app_sent = app_events.count()
            app_clicks = UserNotificationState.objects.filter(
                notification_event__in=app_events,
                opened_at__isnull=False
            ).count()

            app_ctr = (app_clicks / app_sent) if app_sent > 0 else config.GLOBAL_OPEN_RATE_PRIOR

            notifs_past_24h = NotificationEvent.objects.filter(app__user=user, post_time__gte=sent_time - timedelta(days=1)).count()

            user_stats = {
                f"{channel_key}_sent": ch_sent,
                f"{channel_key}_clicks": ch_clicks,
                f"app_{notif.app_id}_ctr": app_ctr,
                "notifs_past_24h": notifs_past_24h,
            }

            # context
            last_interaction = InteractionEvent.objects.filter(user=user, timestamp__lt=sent_time).order_by('-timestamp').first()
            sec_since_last_action = (sent_time - last_interaction.timestamp).total_seconds() if last_interaction else 86400.0

            prev_notif = NotificationEvent.objects.filter(app__user=user, post_time__lt=sent_time).order_by('-post_time').first()
            time_since_last_notif_sec = (sent_time - prev_notif.post_time).total_seconds() if prev_notif else 86400.0

            context = {
                "sec_since_last_action": sec_since_last_action,
                "time_since_last_notif_sec": time_since_last_notif_sec,
            }

            vector = FeatureEngineer.extract(notif, user_stats, context)

            # Ensure label is strictly 0.0 or 1.0
            if label not in (0.0, 1.0, 0, 1):
                continue

            yield vector, float(label)