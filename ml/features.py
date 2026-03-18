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
        Converts a notification + stats into the highly optimized 16-feature vector.
        """
        dt = notification.post_time
        title = notification.title or ""
        text = notification.text or ""
        full_text = f"{title} {text}".lower().strip()

        # --- TIME ---
        raw_hour = dt.hour + (dt.minute / 60.0)
        hour_sin = np.sin(2 * np.pi * raw_hour / 24)
        hour_cos = np.cos(2 * np.pi * raw_hour / 24)

        # --- CTR ---
        channel_ctr = user_stats.get("channel_ctr", 0.1)
        app_ctr = user_stats.get(f"app_{notification.app_id}_ctr", 0.1)

        # --- CONTEXT ---
        is_active_session = 1.0 if context.get("sec_since_last_action", 86400) < 180 else 0.0
        time_since_last = context.get("time_since_last_notif_sec", 86400.0)

        # --- TEXT ---
        text_len = len(full_text)

        digit_density = sum(c.isdigit() for c in full_text) / (text_len + 1e-5)
        exclamation_density = full_text.count('!') / (text_len + 1e-5)
        title_body_ratio = len(title) / (len(text) + 1e-5)

        # --- SEMANTIC FLAGS ---
        def contains(words):
            return float(any(w in full_text for w in words))

        is_otp = contains(["otp", "code", "verification"])
        is_transaction = contains(["debited", "credited", "paid", "txn"])
        is_message = contains(["message", "chat", "call"])
        is_promo = contains(["sale", "offer", "discount"])
        is_urgent = contains(["urgent", "alert", "important"])

        # --- VOLUME ---
        notifs_24h = user_stats.get("notifs_past_24h", 1.0)

        return np.array([
            hour_sin,
            hour_cos,
            channel_ctr,
            app_ctr,
            is_active_session,
            time_since_last,
            digit_density,
            exclamation_density,
            title_body_ratio,
            notifs_24h,
            is_otp,
            is_transaction,
            is_message,
            is_promo,
            is_urgent,
            text_len
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
                'channel_ctr': float(app_ctr),
                f"app_{notif.app_id}_ctr": float(app_ctr),
            }

            context = {
                'time_since_last_notif_sec': float(df.iloc[i]['time_diff']),
            }

            vector = FeatureEngineer.extract(notif, user_stats, context)
            yield vector, float(label)