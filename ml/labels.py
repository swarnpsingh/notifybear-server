"""
Observed-based labeling system.
Labels are derived ONLY from actual user behavior, not engineered rules.
"""

import logging
from django.utils import timezone

logger = logging.getLogger(__name__)


class ObservedLabeler:
    """
    Generate labels based ONLY on observed user behavior.
    
    This replaces the rule-based NotificationLabeler.
    Key principle: Let users teach the model what's important,
    don't impose our assumptions.
    """
    
    @staticmethod
    def label_from_behavior(notification, user):
        """
        Generate engagement score from actual user behavior.
        
        Args:
            notification: NotificationEvent instance
            user: User instance
        
        Returns:
            float or None: Engagement score 0.0-1.0, or None if no interaction yet
            
        Engagement score interpretation:
            1.0 = Opened immediately (< 30 sec) - VERY important to user
            0.8 = Opened quickly (< 5 min) - Important
            0.6 = Opened within 30 min - Moderately important
            0.4 = Opened within 2 hours - Somewhat important
            0.2 = Opened eventually - Low importance
            0.1 = Dismissed without opening - Not important
            0.0 = Ignored completely - Noise
        """
        from Notifications.models import UserNotificationState
        
        # Get user's interaction with this notification
        state = notification.user_states.filter(user=user).first()
        
        if not state:
            logger.debug(f"No state found for notification {notification.id}, user {user.id}")
            return None
        
        # === Case 1: No interaction at all ===
        if not state.opened_at and not state.dismissed_at:
            # Check if enough time has passed to consider it "ignored"
            time_elapsed = (timezone.now() - notification.post_time).total_seconds()
            
            if time_elapsed > 86400:  # 24 hours
                # Ignored for 24+ hours = definitely not important
                return 0.0
            else:
                # Too early to tell, no label yet
                return None
        
        # === Case 2: Dismissed without opening ===
        if state.dismissed_at and not state.opened_at:
            # User explicitly said "not interested"
            return 0.1
        
        # === Case 3: User opened the notification ===
        if state.opened_at:
            reaction_time = (state.opened_at - notification.post_time).total_seconds()
            
            # Map reaction time to engagement score
            if reaction_time < 30:
                # Opened within 30 seconds = VERY important
                engagement = 1.0
            elif reaction_time < 300:  # 5 minutes
                # Quick open = Important
                engagement = 0.8
            elif reaction_time < 1800:  # 30 minutes
                # Moderate delay = Moderately important
                engagement = 0.6
            elif reaction_time < 7200:  # 2 hours
                # Longer delay = Somewhat important
                engagement = 0.4
            else:
                # Opened eventually = Low importance
                engagement = 0.2
            
            # Bonus: If dismissed AFTER opening, still shows engagement
            # (already captured by opening time)
            
            return engagement
        
        # Shouldn't reach here, but default to None
        logger.warning(f"Unexpected state for notification {notification.id}")
        return None
    
    @staticmethod
    def can_be_labeled(notification, user):
        """
        Check if notification has enough interaction data to generate a label.
        
        Args:
            notification: NotificationEvent instance
            user: User instance
        
        Returns:
            bool: True if we can generate a label
        """
        from Notifications.models import UserNotificationState
        
        state = notification.user_states.filter(user=user).first()
        
        if not state:
            return False
        
        # Can label if user has interacted (opened or dismissed)
        if state.opened_at or state.dismissed_at:
            return True
        
        # Can also label if enough time has passed and user ignored it
        time_elapsed = (timezone.now() - notification.post_time).total_seconds()
        if time_elapsed > 86400:  # 24 hours
            return True
        
        return False
    
    @staticmethod
    def batch_label(notifications, user):
        """
        Efficiently label multiple notifications for a user.
        
        Args:
            notifications: QuerySet or list of NotificationEvent instances
            user: User instance
        
        Returns:
            dict: {notification_id: engagement_score}
        """
        from Notifications.models import UserNotificationState
        
        # Prefetch states for efficiency
        notif_ids = [n.id for n in notifications]
        states = UserNotificationState.objects.filter(
            user=user,
            notification_event_id__in=notif_ids
        ).select_related('notification_event')
        
        # Build state lookup
        state_map = {state.notification_event_id: state for state in states}
        
        # Label each notification
        labels = {}
        for notif in notifications:
            state = state_map.get(notif.id)
            
            if state:
                # Create temporary state assignment for labeling
                if not hasattr(notif, 'user_states'):
                    notif._cached_state = state
                
                label = ObservedLabeler.label_from_behavior(notif, user)
                if label is not None:
                    labels[notif.id] = label
        
        logger.info(f"Labeled {len(labels)} / {len(notifications)} notifications for user {user.id}")
        return labels
    
    @staticmethod
    def get_label_distribution(user):
        """
        Get distribution of labels for a user (for monitoring).
        
        Returns:
            dict: Statistics about user's engagement patterns
        """
        from Notifications.models import NotificationEvent
        
        notifications = NotificationEvent.objects.filter(
            app__user=user
        ).prefetch_related('user_states')
        
        labels = []
        for notif in notifications:
            label = ObservedLabeler.label_from_behavior(notif, user)
            if label is not None:
                labels.append(label)
        
        if not labels:
            return {
                "count": 0,
                "mean": None,
                "high_engagement_rate": None,
            }
        
        import numpy as np
        labels_array = np.array(labels)
        
        return {
            "count": len(labels),
            "mean": float(labels_array.mean()),
            "std": float(labels_array.std()),
            "high_engagement_rate": float((labels_array >= 0.7).mean()),
            "low_engagement_rate": float((labels_array <= 0.3).mean()),
        }


class FallbackLabeler:
    """
    Simple heuristic-based labeling for when we have no behavioral data.
    
    ONLY used for:
    - Cold start (new users)
    - Fallback predictions
    - UI explanations
    
    NOT used for training labels.
    """
    
    @staticmethod
    def estimate_importance(features):
        """
        Estimate importance from features (no behavioral data).
        
        Returns:
            float: 0.0-1.0 importance estimate
        """
        score = 0.5  # Start at medium
        
        # Strong signals only
        if features.get("has_urgent", 0) and features.get("has_numbers", 0):
            # Likely OTP or verification code
            score = 0.8
        elif features.get("has_promo", 0):
            # Likely promotional
            score = 0.3
        elif features.get("app_priority", 0.5) > 0.8:
            # High-priority app (messaging, calls)
            score = 0.7
        
        return score