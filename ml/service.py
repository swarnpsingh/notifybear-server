"""
ML Service layer - main interface for predictions and training.

Key changes:
- Uses global model (not per-user models)
- Efficient batch predictions
- Proper caching and error handling
- Clean API for views to use
"""

import logging
from django.core.cache import cache
from django.utils import timezone

from Notifications.models import NotificationEvent, UserNotificationState
from ml.shared_model import get_global_model, train_and_save_global_model, FallbackPredictor
from ml.features import FeatureExtractor
from ml.labels import ObservedLabeler, FallbackLabeler

logger = logging.getLogger(__name__)


class MLService:
    """
    Main service for ML predictions and model management.
    
    This is the primary interface that views should use.
    """
    
    @classmethod
    def predict_engagement(cls, notification, user):
        """
        Predict engagement score for a notification.
        
        Args:
            notification: NotificationEvent instance
            user: User instance
        
        Returns:
            float: 0.0-1.0 engagement score (higher = more likely to engage)
        """
        try:
            # Get global model
            model = get_global_model()
            
            if not model.is_trained:
                logger.info("Global model not trained, using fallback")
                return cls._fallback_predict(notification, user)
            
            # Extract features
            features = FeatureExtractor.extract(notification, user)
            
            # Predict
            engagement_score = model.predict(features)
            
            logger.debug(
                f"Predicted engagement {engagement_score:.3f} for "
                f"notification {notification.id} (user {user.id}, app {notification.app.package_name})"
            )
            
            return engagement_score
        
        except Exception as e:
            logger.error(f"Prediction error for notification {notification.id}: {e}", exc_info=True)
            return cls._fallback_predict(notification, user)
    
    @classmethod
    def predict_engagement_batch(cls, notifications, user):
        """
        Predict engagement for multiple notifications efficiently.
        
        Args:
            notifications: List/QuerySet of NotificationEvent instances
            user: User instance
        
        Returns:
            dict: {notification_id: engagement_score}
        """
        try:
            # Get global model
            model = get_global_model()
            
            if not model.is_trained:
                logger.info("Global model not trained, using fallback")
                return {notif.id: cls._fallback_predict(notif, user) for notif in notifications}
            
            # Batch extract features (efficient)
            features_list = FeatureExtractor.batch_extract(list(notifications), user)
            
            # Batch predict
            engagement_scores = model.predict_batch(features_list)
            
            # Map to notification IDs
            predictions = {
                notif.id: float(score)
                for notif, score in zip(notifications, engagement_scores)
            }
            
            logger.info(f"Batch predicted {len(predictions)} notifications for user {user.id}")
            
            return predictions
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            # Fallback to individual predictions
            return {notif.id: cls._fallback_predict(notif, user) for notif in notifications}
    
    @classmethod
    def rank_notifications(cls, notifications, user):
        """
        Rank notifications by predicted engagement.
        
        Args:
            notifications: List/QuerySet of NotificationEvent instances
            user: User instance
        
        Returns:
            list: Notifications sorted by engagement (high to low)
        """
        if not notifications:
            return []
        
        # Get predictions
        predictions = cls.predict_engagement_batch(notifications, user)
        
        # Sort by engagement score
        ranked = sorted(
            notifications,
            key=lambda n: predictions.get(n.id, 0.5),
            reverse=True
        )
        
        logger.info(f"Ranked {len(ranked)} notifications for user {user.id}")
        
        return ranked
    
    @classmethod
    def _fallback_predict(cls, notification, user):
        """
        Fallback prediction when model not available.
        
        Uses simple heuristics based on features.
        """
        try:
            features = FeatureExtractor.extract(notification, user)
            return FallbackPredictor.predict(features)
        except Exception as e:
            logger.error(f"Even fallback prediction failed: {e}")
            return 0.5  # Ultimate fallback
    
    @classmethod
    def get_priority_bucket(cls, engagement_score):
        """
        Convert engagement score to priority bucket.
        
        Args:
            engagement_score: float 0.0-1.0
        
        Returns:
            int: 0 (low), 1 (medium), 2 (high)
        """
        if engagement_score >= 0.7:
            return 2  # High priority
        elif engagement_score >= 0.4:
            return 1  # Medium priority
        else:
            return 0  # Low priority
    
    @classmethod
    def retrain_global_model(cls, min_users=5, samples_per_user=100):
        """
        Retrain the global model.
        
        This should be called periodically (e.g., daily via Celery).
        
        Args:
            min_users: Minimum users needed to train
            samples_per_user: Max samples per user
        
        Returns:
            bool: Success
        """
        logger.info("MLService: Initiating global model retraining")
        
        success = train_and_save_global_model(
            min_users=min_users,
            samples_per_user=samples_per_user
        )
        
        if success:
            logger.info("MLService: Global model retrained successfully")
        else:
            logger.error("MLService: Global model retraining failed")
        
        return success
    
    @classmethod
    def get_model_info(cls):
        """
        Get information about the current model.
        
        Returns:
            dict: Model metadata
        """
        model = get_global_model()
        return model.get_info()
    
    @classmethod
    def should_retrain(cls):
        """
        Check if global model should be retrained.
        
        Criteria:
        - Model doesn't exist
        - Model is old (>7 days)
        - Significant new data available
        
        Returns:
            tuple: (should_retrain: bool, reason: str)
        """
        model = get_global_model()
        
        if not model.is_trained:
            return True, "Model not trained yet"
        
        if model.last_trained is None:
            return True, "Model training date unknown"
        
        # Check age
        age_days = (timezone.now() - model.last_trained).days
        if age_days > 7:
            return True, f"Model is {age_days} days old (stale)"
        
        # Check if significant new data
        from django.contrib.auth import get_user_model
        from django.db.models import Count, Q
        
        User = get_user_model()
        
        # Count users with recent interactions
        recent_cutoff = timezone.now() - timezone.timedelta(days=1)
        active_users = User.objects.filter(
            notification_states__created_at__gte=recent_cutoff
        ).distinct().count()
        
        if active_users > 10:
            return True, f"{active_users} users with new data in last 24h"
        
        return False, "Model is fresh"
    
    @classmethod
    def update_notification_scores(cls, user, notification_ids=None):
        """
        Update ml_score field for user's notifications.
        
        This can be called after model retraining to update scores.
        
        Args:
            user: User instance
            notification_ids: Optional list of specific notification IDs to update
        """
        # Get notifications to update
        query = UserNotificationState.objects.filter(user=user)
        
        if notification_ids:
            query = query.filter(notification_event_id__in=notification_ids)
        
        states = query.select_related('notification_event', 'notification_event__app')
        
        if not states.exists():
            logger.info(f"No notifications to update for user {user.id}")
            return
        
        # Get predictions
        notifications = [state.notification_event for state in states]
        predictions = cls.predict_engagement_batch(notifications, user)
        
        # Update ml_score field
        updated_count = 0
        for state in states:
            score = predictions.get(state.notification_event_id)
            if score is not None:
                state.ml_score = score
                state.save(update_fields=['ml_score', 'last_updated'])
                updated_count += 1
        
        logger.info(f"Updated ml_score for {updated_count} notifications (user {user.id})")


class MLMonitor:
    """
    Monitor ML system health and performance.
    """
    
    @staticmethod
    def get_system_stats():
        """
        Get overall ML system statistics.
        
        Returns:
            dict: System stats
        """
        from django.contrib.auth import get_user_model
        from django.db.models import Count, Q, Avg
        
        User = get_user_model()
        
        # User stats
        total_users = User.objects.count()
        
        users_with_data = User.objects.annotate(
            notif_count=Count('apps__notification_events')
        ).filter(notif_count__gt=0).count()
        
        users_with_interactions = User.objects.annotate(
            interaction_count=Count(
                'notification_states',
                filter=Q(notification_states__opened_at__isnull=False) | 
                       Q(notification_states__dismissed_at__isnull=False)
            )
        ).filter(interaction_count__gt=0).count()
        
        # Notification stats
        total_notifications = NotificationEvent.objects.count()
        
        labeled_notifications = NotificationEvent.objects.filter(
            Q(user_states__opened_at__isnull=False) | 
            Q(user_states__dismissed_at__isnull=False)
        ).distinct().count()
        
        # Model info
        model_info = MLService.get_model_info()
        
        # Recent activity
        recent_cutoff = timezone.now() - timezone.timedelta(days=7)
        recent_notifications = NotificationEvent.objects.filter(
            post_time__gte=recent_cutoff
        ).count()
        
        return {
            "users": {
                "total": total_users,
                "with_notifications": users_with_data,
                "with_interactions": users_with_interactions,
            },
            "notifications": {
                "total": total_notifications,
                "labeled": labeled_notifications,
                "label_rate": labeled_notifications / total_notifications if total_notifications > 0 else 0,
                "recent_7d": recent_notifications,
            },
            "model": model_info,
            "timestamp": timezone.now().isoformat(),
        }
    
    @staticmethod
    def check_model_health():
        """
        Check if ML system is healthy.
        
        Returns:
            dict: Health check results
        """
        issues = []
        warnings = []
        
        # Check if model is trained
        model = get_global_model()
        if not model.is_trained:
            issues.append("Global model not trained")
        
        # Check model age
        if model.last_trained:
            age_days = (timezone.now() - model.last_trained).days
            if age_days > 14:
                issues.append(f"Model is stale ({age_days} days old)")
            elif age_days > 7:
                warnings.append(f"Model is aging ({age_days} days old)")
        
        # Check data availability
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        eligible_users = User.objects.annotate(
            labeled_count=Count(
                'notification_states',
                filter=Q(notification_states__opened_at__isnull=False) | 
                       Q(notification_states__dismissed_at__isnull=False)
            )
        ).filter(labeled_count__gte=20).count()
        
        if eligible_users < 5:
            warnings.append(f"Only {eligible_users} users with sufficient data")
        
        # Check prediction latency (sample)
        import time
        try:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            user = User.objects.first()
            notif = NotificationEvent.objects.first()
            
            if user and notif:
                start = time.time()
                MLService.predict_engagement(notif, user)
                latency = (time.time() - start) * 1000  # ms
                
                if latency > 500:
                    warnings.append(f"High prediction latency: {latency:.0f}ms")
        except Exception as e:
            warnings.append(f"Could not test prediction latency: {e}")
        
        # Determine overall health
        if issues:
            status = "unhealthy"
        elif warnings:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "timestamp": timezone.now().isoformat(),
        }