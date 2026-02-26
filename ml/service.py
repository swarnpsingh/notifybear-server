from datetime import timedelta
from django.utils import timezone
from .features import FeatureEngineer
from .model import NotificationClassifier
import logging

logger = logging.getLogger(__name__)

class PriorityService:
    def __init__(self):
        self.model = NotificationClassifier()
        self.model.load()

    def predict(self, notification, user_stats, context):
        """
        Pure ML Inference Path. No heuristics. No fallbacks.
        """
        # 1. Extract the 8-feature V5 vector
        # If extraction fails due to bad data, we want this to throw an error 
        # so it gets caught by your monitoring, rather than failing silently into a fallback.
        vector = FeatureEngineer.extract(notification, user_stats, context)

        # 2. Pure ML Prediction
        score = self.model.predict(vector)

        # 3. Bucketize (Business Rule applied post-inference)
        bucket = "normal"
        if score > 0.8: bucket = "high"
        elif score < 0.3: bucket = "low"

        return {
            "score": score,
            "bucket": bucket,
            "vector": vector.tolist()
        }

    def calculate_label(self, sent_time, opened_time=None, dismissed_time=None):
        """
        Determines the strict binary ground truth for the V5 Classifier.
        """
        # Case A: User opened it (Positive Class)
        if opened_time:
            return 1.0
            
        # Case B: User explicitly dismissed it (Negative Class)
        if dismissed_time:
            return 0.0
            
        # Case C: Ignored indefinitely
        if (timezone.now() - sent_time).total_seconds() > 86400:
            return 0.0
            
        return None # Too soon to label

# Singleton instance
service = PriorityService()