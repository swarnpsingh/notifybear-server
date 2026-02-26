import numpy as np
import pandas as pd
from django.utils import timezone
from django.db.models import Count, Q
from . import config

from Notifications.models import (
    NotificationEvent,
    UserNotificationState,
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
        # Lazy import to avoid circular imports at module import time
        from .service import service as priority_service
        from datetime import timedelta

        cutoff = timezone.now() - timedelta(days=lookback_days)

        # 1. Optimized Batch Query for CTRs (single query for all history)
        stats = (
            NotificationEvent.objects.filter(app__user=user)
            .values('app_id', 'channel_id')
            .annotate(
                sent=Count('id'),
                clicks=Count('user_states', filter=Q(user_states__opened_at__isnull=False)),
            )
        )

        ctr_map = {}
        for s in stats:
            key = f"{s['app_id']}_{s['channel_id']}"
            ctr_map[key] = (s['clicks'] / s['sent']) if s['sent'] > 0 else config.GLOBAL_OPEN_RATE_PRIOR

        # 2. Fetch all valid notification logs in one go (prefetch user_states)
        qs = (
            NotificationEvent.objects.filter(app__user=user, post_time__gte=cutoff)
            .select_related('app')
            .prefetch_related('user_states')
        )
        if apps:
            qs = qs.filter(app__package_name__in=apps)

        notifications = list(qs.order_by('post_time'))
        if not notifications:
            return

        # 3. Use Pandas for high-speed rolling calculations
        df = pd.DataFrame([{'id': n.id, 'time': n.post_time} for n in notifications])
        if df.empty:
            return

        df['time'] = pd.to_datetime(df['time'])
        df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(86400.0)
        df['rolling_24h'] = df.rolling('1D', on='time')['id'].count().fillna(0)

        # 4. Final loop with zero DB queries inside
        for i, notif in enumerate(notifications):
            state = notif.user_states.first() if hasattr(notif, 'user_states') else None
            label = priority_service.calculate_label(
                notif.post_time,
                getattr(state, 'opened_at', None),
                getattr(state, 'dismissed_at', None),
            )
            if label is None:
                continue

            # Precomputed stats
            channel_key = f"{notif.app_id}_{notif.channel_id}"
            app_ctr = ctr_map.get(channel_key, config.GLOBAL_OPEN_RATE_PRIOR)

            user_stats = {
                'notifs_past_24h': float(df.iloc[i]['rolling_24h']),
                f"app_{notif.app_id}_ctr": float(app_ctr),
            }

            context = {
                'time_since_last_notif_sec': float(df.iloc[i]['time_diff']),
            }

            vector = FeatureEngineer.extract(notif, user_stats, context)
            yield vector, float(label)