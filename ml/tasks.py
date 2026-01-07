"""
ML tasks without Celery - runs synchronously via Django signals and management commands.

For production with many users, consider migrating to Celery.
For MVP with <500 users, this approach works fine.
"""

import logging
from django.utils import timezone
from django.contrib.auth import get_user_model

from ml.service import MLService, MLMonitor
from ml.features import FeatureExtractor

logger = logging.getLogger(__name__)
User = get_user_model()


def retrain_global_model_sync(min_users=5, samples_per_user=100):
    """
    Retrain global model synchronously.
    
    Call this manually or via management command.
    
    Args:
        min_users: Minimum users needed
        samples_per_user: Max samples per user
    
    Returns:
        dict: Training results
    """
    logger.info("=" * 60)
    logger.info("SYNC TASK: Retrain Global Model")
    logger.info("=" * 60)
    
    try:
        should_retrain, reason = MLService.should_retrain()
        
        if not should_retrain:
            logger.info(f"Skipping retrain: {reason}")
            return {
                "success": False,
                "skipped": True,
                "reason": reason,
            }
        
        logger.info(f"Retraining: {reason}")
        
        start_time = timezone.now()
        success = MLService.retrain_global_model(
            min_users=min_users,
            samples_per_user=samples_per_user
        )
        duration = (timezone.now() - start_time).total_seconds()
        
        if success:
            logger.info(f"Global model retrained in {duration:.1f}s")
            model_info = MLService.get_model_info()
            
            return {
                "success": True,
                "duration_seconds": duration,
                "model_info": model_info,
            }
        else:
            return {
                "success": False,
                "error": "Training failed",
            }
    
    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


def update_user_notification_scores_sync(user, notification_ids=None):
    """
    Update ml_score for user's notifications synchronously.
    
    Args:
        user: User instance
        notification_ids: Optional list of notification IDs
    """
    try:
        MLService.update_notification_scores(user, notification_ids)
        logger.info(f"Updated scores for user {user.id}")
    except Exception as e:
        logger.error(f"Failed to update scores for user {user.id}: {e}")


def precompute_user_stats_sync(user):
    """
    Precompute and cache stats for a user.
    
    Args:
        user: User instance
    """
    try:
        FeatureExtractor.get_cached_user_stats(user)
        logger.debug(f"Precomputed stats for user {user.id}")
    except Exception as e:
        logger.error(f"Failed to precompute stats for user {user.id}: {e}")


def check_ml_health_sync():
    """
    Check ML system health synchronously.
    
    Returns:
        dict: Health check results
    """
    try:
        health = MLMonitor.check_model_health()
        
        if health["status"] == "unhealthy":
            logger.error(f"ML system unhealthy: {health['issues']}")
        elif health["status"] == "degraded":
            logger.warning(f"ML system degraded: {health['warnings']}")
        
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}