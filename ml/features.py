import logging
import re
from datetime import timedelta
from django.utils import timezone
from django.core.cache import cache
from django.db.models import F, Avg, Count, Q

logger = logging.getLogger(__name__)


class FeatureExtractor:
    # Keyword lists for text analysis
    URGENT_WORDS = [
        "otp", "urgent", "verify", "code", "important", 
        "asap", "now", "emergency", "alert", "action required",
        "expires", "deadline", "verify", "confirm"
    ]
    
    PROMO_WORDS = [
        "sale", "offer", "discount", "deal", "win", 
        "free", "limited time", "coupon", "promo", "save",
        "% off", "flash sale"
    ]
    
    PERSON_WORDS = [
        "mom", "dad", "boss", "wife", "husband", "manager",
        "sir", "madam", "team", "family"
    ]
    
    QUESTION_WORDS = ["?", "when", "where", "what", "why", "how", "can you"]
    
    # App priority mapping (weak prior, ML can override)
    APP_PRIORITY = {
        # Messaging & Calls (highest priority)
        'com.whatsapp': 0.9,
        'com.whatsapp.w4b': 0.9,
        'org.telegram.messenger': 0.85,
        'com.google.android.apps.messaging': 0.85,
        'com.android.mms': 0.85,
        'com.android.phone': 0.95,
        
        # Email (high priority)
        'com.google.android.gm': 0.8,
        'com.microsoft.office.outlook': 0.8,
        
        # Work apps (high priority)
        'com.slack': 0.85,
        'com.microsoft.teams': 0.85,
        'us.zoom.videomeetings': 0.8,
        
        # Calendar (high priority)
        'com.google.android.calendar': 0.85,
        
        # Social media (medium-low priority)
        'com.instagram.android': 0.3,
        'com.facebook.katana': 0.3,
        'com.facebook.orca': 0.4,  # Messenger slightly higher
        'com.snapchat.android': 0.3,
        'com.twitter.android': 0.35,
        
        # Entertainment (low priority)
        'com.google.android.youtube': 0.2,
        'com.netflix.mediaclient': 0.2,
        'com.spotify.music': 0.25,
        
        # Games (lowest priority)
        'com.king.candycrushsaga': 0.1,
        'com.supercell.clashofclans': 0.1,
    }
    
    @classmethod
    def extract(cls, notification, user, user_stats=None):
        # Get user stats (cached or computed)
        if user_stats is None:
            user_stats = cls.get_cached_user_stats(user, notification.app, cutoff=notification.post_time)
        
        # Combine all text fields
        text = cls._combine_text(notification)
        text_lower = text.lower()
        
        # === 1. Basic Temporal Features ===
        features = {
            "hour": notification.post_time.hour,
            "day_of_week": notification.post_time.weekday(),
            "is_weekend": int(notification.post_time.weekday() >= 5),
            "is_work_hours": int(9 <= notification.post_time.hour <= 17),
            "is_sleep_hours": int(notification.post_time.hour < 7 or notification.post_time.hour > 22),
            "is_morning": int(6 <= notification.post_time.hour < 12),
            "is_afternoon": int(12 <= notification.post_time.hour < 18),
            "is_evening": int(18 <= notification.post_time.hour < 23),
        }
        
        # === 2. Text Content Features ===
        features.update({
            "text_length": len(text),
            "title_length": len(notification.title or ""),
            "has_urgent": int(any(w in text_lower for w in cls.URGENT_WORDS)),
            "has_promo": int(any(w in text_lower for w in cls.PROMO_WORDS)),
            "has_person": int(any(w in text_lower for w in cls.PERSON_WORDS)),
            "has_question": int(any(w in text_lower for w in cls.QUESTION_WORDS)),
            "has_numbers": int(bool(re.search(r'\d', text))),
            "has_url": int(bool(re.search(r'http|www\.', text_lower))),
            "is_short": int(len(text) < 50),
            "is_long": int(len(text) > 200),
            "word_count": len(text.split()),
            "uppercase_ratio": cls._calculate_uppercase_ratio(text),
        })
        
        # === 3. App Features ===
        features.update({
            "app": notification.app.package_name,
            "app_priority": cls.APP_PRIORITY.get(notification.app.package_name, 0.5),
        })
        
        # === 4. User Behavioral Features (from cache) ===
        features.update({
            "app_open_rate": user_stats.get("app_open_rate", 0.5),
            "app_avg_reaction_time": user_stats.get("app_avg_reaction_time", 300.0),
            "user_global_open_rate": user_stats.get("user_global_open_rate", 0.5),
            "notifications_today": user_stats.get("notifications_today", 0),
            "notifications_this_hour": user_stats.get("notifications_this_hour", 0),
            "time_since_last_notif": user_stats.get("time_since_last_notif", 3600.0),
            "is_first_of_day": int(user_stats.get("notifications_today", 0) == 0),
        })
        
        # === 5. Channel Features ===
        features["channel_id"] = notification.channel_id or "unknown"
        
        # === 6. Derived Features ===
        features.update({
            # High-signal combinations
            "is_likely_otp": int(
                features["has_urgent"] and 
                features["has_numbers"] and 
                features["is_short"]
            ),
            "is_likely_promo": int(
                features["has_promo"] or 
                (features["is_long"] and features["has_url"])
            ),
            "is_high_priority_app": int(features["app_priority"] > 0.7),
            
            # Context features
            "is_notification_burst": int(features["notifications_this_hour"] > 5),
            "is_rare_notification": int(features["time_since_last_notif"] > 7200),  # 2 hours
        })
        
        return features
    
    @classmethod
    def get_cached_user_stats(cls, user, app=None, cutoff=None, cache_duration=3600):
        # Build cache key
        cache_key = f"user_stats_{user.id}"
        if app:
            cache_key += f"_app_{app.id}"
        
        # Try cache first
        stats = cache.get(cache_key)
        if stats is not None:
            logger.debug(f"Cache HIT for {cache_key}")
            return stats
        
        # Cache miss, compute stats
        logger.debug(f"Cache MISS for {cache_key}, computing...")
        stats = cls._compute_user_stats(user, app, cutoff)
        
        # Cache for 1 hour
        cache.set(cache_key, stats, cache_duration)
        
        return stats
    
    @classmethod
    def _compute_user_stats(cls, user, app=None, before_time=None):
        """
        Compute user behavioral statistics from database.
        
        This is expensive, so results are cached.
        """
        from Notifications.models import UserNotificationState, NotificationEvent
        
        now = timezone.now()
        today = now.date()
        
        # === App-specific stats ===
        if app:
            app_states = UserNotificationState.objects.filter(
                user=user,
                notification_event__app=app,
                notification_event__post_time__lt=before_time if before_time else timezone.now()
            )
            total_app_notifs = app_states.count()
            opened_app_notifs = app_states.filter(opened_at__isnull=False).count()
            app_open_rate = opened_app_notifs / total_app_notifs if total_app_notifs > 0 else 0.5
            
            # Average reaction time for this app
            app_reactions = app_states.filter(
                opened_at__isnull=False
            ).annotate(
                reaction=F('opened_at') - F('notification_event__post_time')
            ).values_list('reaction', flat=True)
            
            if app_reactions:
                avg_reaction = sum(rt.total_seconds() for rt in app_reactions) / len(app_reactions)
            else:
                avg_reaction = 300.0  # Default 5 minutes
        else:
            app_open_rate = 0.5
            avg_reaction = 300.0
        
        # === Global user stats ===
        all_states = UserNotificationState.objects.filter(
            user=user,
            notification_event__post_time__lt=before_time if before_time else timezone.now()
        )
        total_notifs = all_states.count()
        opened_notifs = all_states.filter(opened_at__isnull=False).count()
        global_open_rate = opened_notifs / total_notifs if total_notifs > 0 else 0.5
        
        # === Today's activity ===
        notifications_today = NotificationEvent.objects.filter(
            app__user=user,
            post_time__lt=before_time if before_time else timezone.now(),
            post_time__date=today
        ).count()
        
        # === This hour's activity ===
        notifications_this_hour = NotificationEvent.objects.filter(
            app__user=user,
            post_time__gte=now - timedelta(hours=1)
        ).count()
        
        # === Time since last notification ===
        last_notif = NotificationEvent.objects.filter(
            app__user=user
        ).order_by('-post_time').first()
        
        if last_notif:
            time_since_last = (now - last_notif.post_time).total_seconds()
        else:
            time_since_last = 3600.0  # Default 1 hour
        
        return {
            "app_open_rate": app_open_rate,
            "app_avg_reaction_time": avg_reaction,
            "user_global_open_rate": global_open_rate,
            "notifications_today": notifications_today,
            "notifications_this_hour": notifications_this_hour,
            "time_since_last_notif": time_since_last,
        }
    
    @classmethod
    def invalidate_user_cache(cls, user, app=None):
        cache_key = f"user_stats_{user.id}"
        cache.delete(cache_key)
        
        if app:
            cache_key_app = f"user_stats_{user.id}_app_{app.id}"
            cache.delete(cache_key_app)
        
        logger.debug(f"Invalidated cache for user {user.id}")
    
    @classmethod
    def precompute_all_user_stats(cls, users=None):
        from django.contrib.auth import get_user_model
        
        if users is None:
            User = get_user_model()
            users = User.objects.all()
        
        for user in users:
            try:
                # Compute and cache
                stats = cls._compute_user_stats(user)
                cache_key = f"user_stats_{user.id}"
                cache.set(cache_key, stats, 3600)  # 1 hour
                
                logger.info(f"Precomputed stats for user {user.id}")
            except Exception as e:
                logger.error(f"Failed to precompute stats for user {user.id}: {e}")
    
    @staticmethod
    def _combine_text(notification):
        parts = [
            notification.title or "",
            notification.text or "",
            notification.big_text or "",
            notification.sub_text or "",
        ]
        return " ".join(p for p in parts if p)
    
    @staticmethod
    def _calculate_uppercase_ratio(text):
        if not text:
            return 0.0
        
        uppercase_count = sum(1 for c in text if c.isupper())
        letter_count = sum(1 for c in text if c.isalpha())
        
        if letter_count == 0:
            return 0.0
        
        return uppercase_count / letter_count
    
    @classmethod
    def batch_extract(cls, notifications, user):
        # Compute user stats once for all notifications
        user_stats = cls.get_cached_user_stats(user)
        
        # Extract features for each notification
        features_list = []
        for notif in notifications:
            features = cls.extract(notif, user, user_stats=user_stats)
            features_list.append(features)
        
        logger.info(f"Batch extracted features for {len(notifications)} notifications")
        return features_list
    
    @staticmethod
    def to_vector(feature_dict):
        keys = sorted(feature_dict.keys())  # fixed order

        vector = []
        for k in keys:
            v = feature_dict[k]

            if isinstance(v, (int, float)):
                vector.append(float(v))
            else:
                vector.append(0.0)  # ignore non-numeric features

        return vector