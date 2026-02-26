from datetime import timedelta
import logging
import os
from django.utils import timezone

from .model import NotificationClassifier
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelRetrainer:
    MIN_RETRAIN_INTERVAL = timedelta(hours=6)

    @staticmethod
    def should_retrain(user):
        last = getattr(user, "last_model_retrain", None)
        if last and (timezone.now() - last) < ModelRetrainer.MIN_RETRAIN_INTERVAL:
            return False, "recently retrained"
        return True, "ok"

    @staticmethod
    def train_model(user, apps=None, lookback_days=30):
        """
        Strict retraining pipeline following the project's rules:
        - Fetch historical notifications for the user (filter by apps if provided)
        - For each notification compute label via service.calculate_label
        - Drop rows where label is None
        - Extract features with FeatureEngineer.extract
        - Train NotificationClassifier.train(X, y)

        Returns (metrics, file_path) where metrics contains sample counts.
        """
        X = []
        y = []

        rows = list(FeatureEngineer.fetch_training_rows(user, apps=apps, lookback_days=lookback_days))
        for vec, label in rows:
            X.append(vec)
            y.append(label)

        metrics = {"samples": len(X)}

        clf = NotificationClassifier()
        if len(X) < 0:
            logger.info("Not enough data to train: %d samples", len(X))
            return metrics, None

        clf.train(X, y)

        # Update user's last retrain timestamp when possible
        try:
            # last_model_retrain lives on the user's profile object
            user.profile.last_model_retrain = timezone.now()
            user.profile.save()
        except Exception:
            logger.debug("Profile not found or could not persist last_model_retrain")

        # Prefer ONNX if exported, else pickle
        if os.path.exists(clf.onnx_path):
            return metrics, clf.onnx_path
        if os.path.exists(clf.model_path):
            return metrics, clf.model_path

        return metrics, None
