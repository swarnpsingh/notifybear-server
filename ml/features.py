import re
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
        SYNCED WITH V14 EDGE ONNX MODEL.
        """
        dt = notification.post_time
        
        # 1. Clean and combine text (Match Edge logic exactly)
        safe_title = str(notification.title or "").strip().lower()
        safe_body = str(notification.text or "").strip().lower()
        full_text = f"{safe_title} {safe_body}".strip()

        text_length = float(len(full_text)) if len(full_text) > 0 else 5.0
        if len(full_text) == 0:
            full_text = "empty"

        # --- TIME ---
        raw_hour = dt.hour + (dt.minute / 60.0)
        hour_sin = float(np.sin(2 * np.pi * raw_hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * raw_hour / 24.0))

        # --- CTR ---
        channel_ctr = float(user_stats.get("channel_ctr", 0.0))
        app_ctr = float(user_stats.get(f"app_{notification.app_id}_ctr", 0.0))

        # --- CONTEXT ---
        is_active_session = 1.0 if context.get("sec_since_last_action", 86400) < 180 else 0.0
        time_since_last = float(context.get("time_since_last_notif_sec", 0.0))

        # --- TEXT COMPLEXITY ---
        digit_density = sum(c.isdigit() for c in full_text) / text_length
        exclamation_density = full_text.count('!') / text_length
        
        body_len = float(len(safe_body)) if len(safe_body) > 0 else 1.0
        title_body_ratio = len(safe_title) / body_len

        # --- SEMANTIC FLAGS (REGEX) ---
        # Using exact patterns from the Android edge to prevent training-serving skew
        is_otp = 1.0 if re.search(r'\b\d{4,6}\b|otp|verification|code', full_text) else 0.0
        is_transaction = 1.0 if re.search(r'spent|debited|credited|transaction|a/c|bank|balance|rs\.|inr|payment', full_text) else 0.0
        is_urgent = 1.0 if re.search(r'urgent|immediate|action required|important|alert', full_text) else 0.0
        is_promo = 1.0 if re.search(r'off|sale|discount|deal|limited time|promo', full_text) else 0.0
        
        # Message checks the app_id/package_name, not the text
        safe_app_id = str(notification.app_id or "").lower()
        is_message = 1.0 if re.search(r'whatsapp|telegram|messenger|messaging', safe_app_id) else 0.0

        # --- VOLUME ---
        notifs_24h = float(user_stats.get("notifs_past_24h", 0.0))

        # --- THE V14 FEATURE VECTOR ---
        # Order is strictly enforced for ONNX compatibility
        return np.array([
            hour_sin,               # 1
            hour_cos,               # 2
            channel_ctr,            # 3
            app_ctr,                # 4
            is_active_session,      # 5
            time_since_last,        # 6
            digit_density,          # 7
            exclamation_density,    # 8
            title_body_ratio,       # 9
            notifs_24h,             # 10
            is_otp,                 # 11
            is_transaction,         # 12
            is_message,             # 13
            is_promo,               # 14
            is_urgent,              # 15
            text_length             # 16
        ], dtype=np.float32)

    @staticmethod
    def fetch_training_rows(user, apps=None, lookback_days=30, max_samples=None):
        """
        Yield tuples (feature_vector, label) for the given user.
        """
        from .service import service as priority_service
        from datetime import timedelta

        cutoff = timezone.now() - timedelta(days=lookback_days)

        # 1. Optimized Batch Query for CTRs
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

        # 2. Fetch all valid logs
        qs = (
            NotificationEvent.objects.filter(app__user=user, post_time__gte=cutoff)
            .select_related('app')
            .prefetch_related('user_states')
        )
        if apps:
            qs = qs.filter(app__package_name__in=apps)

        if max_samples:
            qs = qs.order_by('-post_time')[:max_samples]
        else:
            qs = qs.order_by('post_time')
        notifications = list(qs)
        
        if not notifications:
            return

        # 3. Pandas Rolling Calcs
        df = pd.DataFrame([{'id': n.id, 'time': n.post_time} for n in notifications])
        if df.empty:
            return

        df['time'] = pd.to_datetime(df['time'])
        df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(86400.0)
        
        # Subtracted 1 to exclude the current notification from the "past" count
        df['rolling_24h'] = df.rolling('1D', on='time')['id'].count().fillna(1.0) - 1.0 
        count = 0
        # 4. Final loop yielding feature vectors and labels
        for i, notif in enumerate(notifications):
            if max_samples is not None and count >= max_samples:
                break
            state = notif.user_states.order_by('-id').first() if hasattr(notif, 'user_states') else None
            manual_score = None
            if state and getattr(state, "manual_priority", None) is not None:
                if state.manual_priority == "HIGH":
                    manual_score = 1.0
                elif state.manual_priority == "LOW":
                    manual_score = 0.0
                else:
                    manual_score = 0.5

            base_label = priority_service.calculate_label(
                notif.post_time,
                getattr(state, 'opened_at', None) if state else None,
                getattr(state, 'dismissed_at', None) if state else None,
            )

            label = None
            
            if manual_score is not None:
                label = manual_score
            elif base_label is not None:
                label = base_label
                
            if label is None:
                continue

            channel_key = f"{notif.app_id}_{notif.channel_id}"
            app_ctr = ctr_map.get(channel_key, config.GLOBAL_OPEN_RATE_PRIOR)

            user_stats = {
                'notifs_past_24h': float(df.iloc[i]['rolling_24h']),
                'channel_ctr': float(app_ctr),
                f"app_{notif.app_id}_ctr": float(app_ctr),
            }

            context = {
                'time_since_last_notif_sec': float(df.iloc[i]['time_diff']),
                # In a real system, you might fetch actual session states, falling back to 86400
                'sec_since_last_action': 86400 
            }

            vector = FeatureEngineer.extract(notif, user_stats, context)
            
            if manual_score is not None:
                if vector[10] == 1.0 or vector[11] == 1.0:
                    label = 1.0
                
            yield vector, float(label)
            
            count += 1
