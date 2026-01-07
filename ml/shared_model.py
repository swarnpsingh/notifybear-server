"""
Shared global model for all users.

Key innovation:
- ONE model for all users (not per-user models)
- Learns from ALL users' data (shared learning)
- User-specific features provide personalization
- Much easier to maintain and scale
"""

import os
import logging
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Count, Q
from django.core.cache import cache

from Notifications.models import NotificationEvent
from ml.model import UserNotificationModel, ModelEvaluator
from ml.labels import ObservedLabeler
from ml.features import FeatureExtractor
from ml.synthetic import SyntheticDataGenerator

logger = logging.getLogger(__name__)
User = get_user_model()


class SharedNotificationModel:
    """
    Single global model shared across all users.
    
    Benefits:
    - Learns from all users (stronger patterns)
    - New users get instant predictions (no cold start)
    - One model to maintain (not 1000s)
    - Scales to millions of users
    
    Personalization:
    - User-specific features (open rates, behavior stats)
    - Model learns "users like this user tend to..."
    """
    
    def __init__(self):
        self.global_model = None
        self.is_trained = False
        self.last_trained = None
        self.training_stats = {}
    
    def train_global_model(self, min_users=5, samples_per_user=100, model_type='gbm'):
        """
        Train ONE model on data from ALL users.
        
        Args:
            min_users: Minimum users needed to train
            samples_per_user: Max samples to take from each user
            model_type: 'ridge' or 'gbm'
        
        Returns:
            bool: Success
        """
        logger.info("=" * 60)
        logger.info("TRAINING GLOBAL MODEL")
        logger.info("=" * 60)
        
        # Find users with sufficient labeled data
        eligible_users = User.objects.annotate(
            labeled_count=Count(
                'notification_states',
                filter=Q(notification_states__opened_at__isnull=False) | 
                       Q(notification_states__dismissed_at__isnull=False)
            )
        ).filter(labeled_count__gte=20)  # At least 20 labeled notifications
        
        user_count = eligible_users.count()
        
        if user_count < min_users:
            logger.warning(f"Not enough users to train global model: {user_count} < {min_users}")
            return False
        
        logger.info(f"Found {user_count} eligible users for training")
        
        # Collect data from all users
        all_data = []
        user_contributions = {}
        
        for user in eligible_users:
            user_data = self._collect_user_data(user, max_samples=samples_per_user)
            
            if user_data:
                all_data.extend(user_data)
                user_contributions[user.id] = len(user_data)
                logger.info(f"User {user.id}: contributed {len(user_data)} samples")
        
        if len(all_data) < 100:
            logger.warning(f"Insufficient total data: {len(all_data)} < 100")
            return False
        
        logger.info(f"Collected {len(all_data)} total samples from {len(user_contributions)} users")
        
        # Analyze data quality
        self._analyze_training_data(all_data)
        
        # Shuffle to mix users
        import random
        random.shuffle(all_data)
        
        # Decide model type based on data size
        if len(all_data) < 500:
            model_type = 'ridge'
            logger.info("Using Ridge (data size < 500)")
        else:
            logger.info(f"Using {model_type.upper()}")
        
        # Train model
        try:
            logger.info("Training model...")
            self.global_model = UserNotificationModel(model_type=model_type)
            metrics = self.global_model.train(all_data, validate=True)
            
            self.is_trained = True
            self.last_trained = timezone.now()
            
            # Store training stats
            self.training_stats = {
                "num_users": len(user_contributions),
                "total_samples": len(all_data),
                "samples_per_user": user_contributions,
                "metrics": metrics,
                "model_type": model_type,
                "trained_at": self.last_trained.isoformat(),
            }
            
            logger.info("=" * 60)
            logger.info("GLOBAL MODEL TRAINING COMPLETE")
            logger.info(f"  Users: {len(user_contributions)}")
            logger.info(f"  Samples: {len(all_data)}")
            logger.info(f"  Val RMSE: {metrics.get('val_rmse', 'N/A')}")
            logger.info(f"  Val MAE: {metrics.get('val_mae', 'N/A')}")
            logger.info(f"  Rank Corr: {metrics.get('rank_correlation', 'N/A')}")
            logger.info("=" * 60)
            
            return True
        
        except Exception as e:
            logger.error(f"Global model training failed: {e}", exc_info=True)
            return False
    
    def _collect_user_data(self, user, max_samples=100):
        """
        Collect labeled training data from a single user.
        
        Args:
            user: User instance
            max_samples: Maximum samples to collect
        
        Returns:
            list: [(features, engagement_score), ...]
        """
        user_data = []
        
        # Get user's notifications with interactions
        notifications = NotificationEvent.objects.filter(
            app__user=user
        ).select_related('app').prefetch_related('user_states').order_by('-post_time')[:max_samples * 2]
        
        for notif in notifications:
            if ObservedLabeler.can_be_labeled(notif, user):
                try:
                    features = FeatureExtractor.extract(notif, user)
                    engagement_score = ObservedLabeler.label_from_behavior(notif, user)
                    
                    if engagement_score is not None:
                        user_data.append((features, engagement_score))
                        
                        # Stop once we have enough
                        if len(user_data) >= max_samples:
                            break
                
                except Exception as e:
                    logger.error(f"Failed to extract data from notification {notif.id}: {e}")
        
        return user_data
    
    def _analyze_training_data(self, dataset):
        """
        Analyze training data quality (for logging).
        """
        import numpy as np
        
        labels = np.array([label for _, label in dataset])
        
        logger.info("Training Data Analysis:")
        logger.info(f"  Label mean: {labels.mean():.3f}")
        logger.info(f"  Label std: {labels.std():.3f}")
        logger.info(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}]")
        logger.info(f"  High engagement (>=0.7): {(labels >= 0.7).mean():.1%}")
        logger.info(f"  Low engagement (<0.4): {(labels < 0.4).mean():.1%}")
        
        # Check for label quality issues
        if labels.std() < 0.1:
            logger.warning("Very low label variance - all engagement scores are similar!")
        
        if (labels >= 0.7).mean() > 0.8:
            logger.warning("Most labels are high engagement - possible data bias")
        
        if (labels < 0.3).mean() > 0.8:
            logger.warning("Most labels are low engagement - users may be disengaged")
    
    def predict(self, features):
        """
        Predict engagement score using global model.
        
        Args:
            features: Feature dict
        
        Returns:
            float: 0.0-1.0 engagement score
        """
        if not self.is_trained or self.global_model is None:
            logger.warning("Global model not trained, returning default score")
            return 0.5
        
        try:
            return self.global_model.predict(features)
        except Exception as e:
            logger.error(f"Global model prediction failed: {e}")
            return 0.5
    
    def predict_batch(self, features_list):
        """
        Batch predict for multiple notifications.
        
        Args:
            features_list: List of feature dicts
        
        Returns:
            list: Engagement scores
        """
        if not self.is_trained or self.global_model is None:
            logger.warning("Global model not trained, returning default scores")
            return [0.5] * len(features_list)
        
        try:
            return self.global_model.predict_batch(features_list)
        except Exception as e:
            logger.error(f"Global model batch prediction failed: {e}")
            return [0.5] * len(features_list)
    
    def save(self, path):
        """
        Save global model to disk.
        """
        if self.global_model is None:
            raise ValueError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.global_model.save(path)
        
        # Save metadata
        import json
        metadata_path = path.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"Global model saved to {path}")
    
    def load(self, path):
        """
        Load global model from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.global_model = UserNotificationModel()
        self.global_model.load(path)
        self.is_trained = True
        
        # Load metadata if available
        metadata_path = path.replace('.joblib', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                self.training_stats = json.load(f)
            
            if 'trained_at' in self.training_stats:
                from django.utils.dateparse import parse_datetime
                self.last_trained = parse_datetime(self.training_stats['trained_at'])
        
        logger.info(f"Global model loaded from {path}")
    
    def get_info(self):
        """
        Get model information.
        
        Returns:
            dict: Model metadata
        """
        if not self.is_trained:
            return {"trained": False}
        
        info = {
            "trained": True,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "model_type": self.global_model.model_type if self.global_model else None,
        }
        
        if self.training_stats:
            info.update({
                "num_users": self.training_stats.get("num_users"),
                "total_samples": self.training_stats.get("total_samples"),
                "val_rmse": self.training_stats.get("metrics", {}).get("val_rmse"),
                "val_mae": self.training_stats.get("metrics", {}).get("val_mae"),
            })
        
        return info


# Singleton instance
_global_model_instance = None
_global_model_lock = False


def get_global_model():
    """
    Get or create the global model singleton.
    
    Returns:
        SharedNotificationModel
    """
    global _global_model_instance, _global_model_lock
    
    if _global_model_instance is None:
        # Check cache first
        cached_model = cache.get('global_notification_model')
        if cached_model:
            logger.info("Loaded global model from cache")
            _global_model_instance = cached_model
            return _global_model_instance
        
        # Create new instance
        _global_model_instance = SharedNotificationModel()
        
        # Try to load from disk
        model_path = os.path.join(settings.BASE_DIR, 'models', 'global_model.joblib')
        if os.path.exists(model_path):
            try:
                _global_model_instance.load(model_path)
                logger.info("Loaded global model from disk")
                
                # Cache for 1 hour
                cache.set('global_notification_model', _global_model_instance, 3600)
            except Exception as e:
                logger.error(f"Failed to load global model from disk: {e}")
    
    return _global_model_instance


def invalidate_global_model_cache():
    """
    Invalidate cached global model (call after retraining).
    """
    global _global_model_instance
    _global_model_instance = None
    cache.delete('global_notification_model')
    logger.info("Global model cache invalidated")


def train_and_save_global_model(min_users=5, samples_per_user=100):
    """
    Train global model and save to disk.
    
    This should be called periodically (e.g., daily via Celery task).
    
    Args:
        min_users: Minimum users needed
        samples_per_user: Max samples per user
    
    Returns:
        bool: Success
    """
    model = SharedNotificationModel()
    
    success = model.train_global_model(
        min_users=min_users,
        samples_per_user=samples_per_user
    )
    
    if success:
        # Save to disk
        model_path = os.path.join(settings.BASE_DIR, 'models', 'global_model.joblib')
        model.save(model_path)
        
        # Invalidate cache so next request loads new model
        invalidate_global_model_cache()
        
        logger.info("Global model trained and saved successfully")
        return True
    else:
        logger.error("Global model training failed")
        return False


class FallbackPredictor:
    """
    Simple rule-based predictor when global model not available.
    """
    
    @staticmethod
    def predict(features):
        """
        Simple heuristic prediction.
        
        Returns:
            float: 0.0-1.0 engagement estimate
        """
        from ml.labels import FallbackLabeler
        return FallbackLabeler.estimate_importance(features)