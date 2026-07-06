from datetime import timedelta
import logging
import os
from django.utils import timezone

from ml.models import TrainingFeature

from .model import NotificationClassifier
from .features import FeatureEngineer

logger = logging.getLogger(__name__)

INIT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "init_models",
    "init.onnx"
)

class ModelRetrainer:
    MIN_NEW_FEATURES = 50
    
    @staticmethod
    def should_retrain_from_features(user):
        unused_count = TrainingFeature.objects.filter(user=user, used_for_training=False).count()
        if unused_count < ModelRetrainer.MIN_NEW_FEATURES:
            return False, f"only {unused_count} new features"
        return True, "sufficient new features"

    @staticmethod
    def train_model(user, apps=None, lookback_days=30):
        """
        Strict retraining pipeline following the project's rules:
        - Fetch historical notifications for the user (filter by apps if provided)
        - For each notification compute label via service.calculate_label
        - Drop rows where label is None
        - Use mobile-extracted features stored in UserNotificationState
        - Train NotificationClassifier.train(X, y)

        Returns (metrics, file_path) where metrics contains sample counts.
        """
        X = []
        y = []

        rows = FeatureEngineer.fetch_training_rows(user, apps=apps, lookback_days=lookback_days, max_samples=1000)
        used_ids = []
        for row_id, vec, label in rows:
            used_ids.append(row_id)
            X.append(vec)
            y.append(label)

        metrics = {"samples": len(X)}

        clf = NotificationClassifier()
        if len(X) < 50:
            if os.path.exists(INIT_MODEL_PATH):
                logger.error("init.onnx returned due to insufficient training data")
                return metrics, None
            logger.error("init.onnx not found!")
            return metrics, None

        clf.train(X, y)
        TrainingFeature.objects.filter(
            id__in=used_ids
        ).update(
            used_for_training=True
        )

        # Update user's last retrain timestamp when possible
        try:
            # last_model_retrain lives on the user's profile object
            user.profile.last_model_retrain = timezone.now()
            user.profile.save()
        except Exception as e:
            logger.debug("Profile not found or could not persist last_model_retrain: "+str(e))

        return metrics, clf