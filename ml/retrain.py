"""
Model retraining with intelligent synthetic/real data mixing.

Key changes:
- Uses observed labels (not engineered rules)
- Synthetic data is minimal and phased out
- Real data weighted 2x higher than synthetic
- Smart threshold-based mixing strategy
"""

import random
from django.db.models import Count, Q
from django.utils import timezone
import tempfile
from Notifications.models import NotificationEvent, UserNotificationState
from ml.labels import ObservedLabeler
from ml.features import FeatureExtractor
from ml.synthetic import SyntheticDataGenerator
from ml.model import UserNotificationModel

CANONICAL_FEATURE_ORDER = [
        # ---- Temporal ----
        "hour",
        "day_of_week",
        "is_weekend",
        "is_work_hours",
        "is_sleep_hours",
        "is_morning",
        "is_afternoon",
        "is_evening",

        # ---- Text ----
        "text_length",
        "title_length",
        "has_urgent",
        "has_promo",
        "has_person",
        "has_question",
        "has_numbers",
        "has_url",
        "is_short",
        "is_long",
        "word_count",
        "uppercase_ratio",

        # ---- App ----
        "app_priority",

        # ---- User stats ----
        "app_open_rate",
        "app_avg_reaction_time",
        "user_global_open_rate",
        "notifications_today",
        "notifications_this_hour",
        "time_since_last_notif",
        "is_first_of_day",

        # ---- Derived ----
        "is_likely_otp",
        "is_likely_promo",
        "is_high_priority_app",
        "is_notification_burst",
        "is_rare_notification"
    ]

class ModelRetrainer:
    """
    Handles intelligent model retraining for users.
    """
    
    # Thresholds for mixing strategy
    THRESHOLD_PURE_REAL = 100  # Use only real data if we have this many
    THRESHOLD_MIXED = 50       # Start mixing if we have this many
    THRESHOLD_COLD_START = 20  # Below this, mostly synthetic
    
    @staticmethod
    def should_retrain(user):
        """
        Determine if we should retrain the model for this user.
        
        Returns:
            tuple: (should_retrain: bool, reason: str)
        """
        profile = user.profile

        if profile.last_model_retrain:
            age_days = (timezone.now() - profile.last_model_retrain).days
            if age_days > 3:
                return True, f"Model older than {age_days} days"
        # Count labeled interactions (where user actually did something)
        labeled_count = UserNotificationState.objects.filter(
            Q(user=user) & (Q(opened_at__isnull=False) | Q(dismissed_at__isnull=False))
        ).count()
        
        if labeled_count == 0:
            return False, "No labeled interactions yet"
        
        # Check if user has minimum data
        if labeled_count < ModelRetrainer.THRESHOLD_COLD_START:
            return False, f"Only {labeled_count} interactions (need {ModelRetrainer.THRESHOLD_COLD_START}+)"
        
        # Retrain at key milestones
        if labeled_count in [20, 50, 100, 200, 500, 1000]:
            return True, f"Milestone: {labeled_count} interactions"
        
        # Retrain every 100 interactions after 1000
        if labeled_count >= 1000 and labeled_count % 100 == 0:
            return True, f"Periodic retrain: {labeled_count} interactions"
        
        return False, "No retrain needed"
    @staticmethod
    def build_dataset(user, apps=None, target_size=500):
        """
        Always return a dataset of exactly `target_size`.

        - Use real labeled data when available.
        - If real data is insufficient, fill with synthetic data.
        - If ZERO real data exists, train on purely synthetic data.
        """

        # Get user's apps
        qs = NotificationEvent.objects.filter(app__user=user)
        if apps:
            qs = qs.filter(app__package_name__in=apps)

        user_apps = list(
            qs.values_list('app__package_name', flat=True).distinct()
        )

        # If user has NO apps in DB, still create a synthetic dataset
        if not user_apps:
            return SyntheticDataGenerator.generate_for_cold_start(
                apps=apps or ["unknown"],
                n=target_size
            )

        # === Extract LABELED real data ===
        real_data = []

        notifications = qs.select_related('app') \
            .prefetch_related('user_states') \
            .order_by('-post_time')

        for notif in notifications:
            if ObservedLabeler.can_be_labeled(notif, user):
                try:
                    features = FeatureExtractor.extract(notif, user)
                    engagement_score = ObservedLabeler.label_from_behavior(notif, user)

                    if engagement_score is not None:
                        vec = FeatureExtractor.to_vector(features)
                        real_data.append((vec, engagement_score))

                except Exception:
                    continue

        # ---- CASE 1: Plenty of real data ----
        if len(real_data) >= target_size:
            random.shuffle(real_data)
            return real_data[:target_size]

        # ---- CASE 2: Some real data, but less than target ----
        if 0 < len(real_data) < target_size:
            needed = target_size - len(real_data)

            synthetic = SyntheticDataGenerator.generate_for_cold_start(
                apps=user_apps,
                n=needed
            )
            synthetic_vectors = [
                (FeatureExtractor.to_vector(feats), label)
                for feats, label in synthetic
            ]
            dataset = real_data + synthetic_vectors
            random.shuffle(dataset)
            return dataset

        # ---- CASE 3: ZERO real data ----
        # Purely synthetic warm start
        synthetic = SyntheticDataGenerator.generate_for_cold_start(
            apps=user_apps,
            n=target_size
        )

        return [
            (FeatureExtractor.to_vector(feats), label)
            for feats, label in synthetic
        ]
    # @staticmethod
    # def build_dataset(user, apps=None, target_size=500):
    #     """
    #     Build training dataset with smart real/synthetic mixing.
        
    #     Strategy:
    #     - >= 100 labeled: Pure real data (no synthetic)
    #     - 50-99 labeled: Real data (2x weight) + minimal synthetic
    #     - 20-49 labeled: Real data + moderate synthetic
    #     - < 20 labeled: Don't train yet (return None)
        
    #     Args:
    #         user: User instance
    #         target_size: Target dataset size (default 500)
        
    #     Returns:
    #         list of (features, engagement_score) tuples, or None if insufficient data
    #     """
        
    #     # Get user's apps
    #     qs = NotificationEvent.objects.filter(app__user=user)
    #     if apps:
    #         qs = qs.filter(app__package_name__in=apps)

    #     user_apps = list(qs.values_list('app__package_name', flat=True).distinct())
                
    #     if not user_apps:
    #         return None
        
    #     # === Extract LABELED real data ===
    #     real_data = []
        
    #     # Get notifications with interactions
    #     notifications = qs.select_related('app').prefetch_related('user_states').order_by('-post_time')
        
    #     for notif in notifications:
    #         # Only use if we can generate a label from behavior
    #         if ObservedLabeler.can_be_labeled(notif, user):
    #             try:
    #                 features = FeatureExtractor.extract(notif, user)
    #                 engagement_score = ObservedLabeler.label_from_behavior(notif, user)
                    
    #                 if engagement_score is not None:
    #                     real_data.append((features, engagement_score))
                
    #             except Exception as e:
    #                 continue
        
        
    #     # Analyze real data quality
    #     if real_data:
    #         real_labels = [label for _, label in real_data]
    #         import numpy as np
    #         label_array = np.array(real_labels)
        
    #     # === Decide on mixing strategy ===
        
    #     if len(real_data) >= ModelRetrainer.THRESHOLD_PURE_REAL:            
    #         random.shuffle(real_data)
    #         dataset = real_data[:target_size]
            
    #         return dataset
        
    #     elif len(real_data) >= ModelRetrainer.THRESHOLD_MIXED:
            
    #         # Weight real data 2x by duplicating
    #         weighted_real = real_data * 2
            
    #         # Add minimal synthetic to reach target
    #         synthetic_needed = max(0, target_size - len(weighted_real))
    #         synthetic_needed = min(synthetic_needed, 100)  # Cap at 100 synthetic
            
    #         if synthetic_needed > 0:
    #             synthetic = SyntheticDataGenerator.generate_for_cold_start(
    #                 apps=user_apps,
    #                 n=synthetic_needed
    #             )
    #         else:
    #             synthetic = []
            
    #         dataset = weighted_real + synthetic
    #         random.shuffle(dataset)
            
            
    #         return dataset[:target_size]
        
    #     elif len(real_data) >= ModelRetrainer.THRESHOLD_COLD_START:
    #         synthetic_needed = max(200, target_size - len(real_data))
            
    #         synthetic = SyntheticDataGenerator.generate_for_cold_start(
    #             apps=user_apps,
    #             n=synthetic_needed
    #         )
            
    #         dataset = real_data + synthetic
    #         random.shuffle(dataset)
    #         return dataset[:target_size]
        
    #     else:
    #         return None
    
    @staticmethod
    def train_model(user, apps=None, model_type='gbm', target_size=500, validate=True):
        dataset = ModelRetrainer.build_dataset(user, apps=apps, target_size=target_size)
        if not dataset:
            return None, None

        cleaned_dataset = dataset
        
        model = UserNotificationModel(model_type=model_type)
        metrics = model.train(cleaned_dataset, validate=validate)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
        feature_dim = len(CANONICAL_FEATURE_ORDER)
        model.save_onnx(tmp.name, feature_count=feature_dim)

        user.profile.last_model_retrain = timezone.now()
        user.profile.save(update_fields=["last_model_retrain"])

        return metrics, tmp.name
    
    @staticmethod
    def evaluate_on_recent_data(model, user, n_recent=50):
        """
        Evaluate model on recent unseen notifications (sanity check).
        
        Args:
            model: Trained UserNotificationModel
            user: User instance
            n_recent: Number of recent notifications to test on
        
        Returns:
            dict: Evaluation metrics
        """
        if model is None:
            return {"error": "No model provided"}
        
        # Get recent notifications
        recent_notifs = NotificationEvent.objects.filter(
            app__user=user
        ).select_related('app').prefetch_related('user_states').order_by('-post_time')[:n_recent]
        
        # Build test data
        test_data = []
        for notif in recent_notifs:
            if ObservedLabeler.can_be_labeled(notif, user):
                try:
                    features = FeatureExtractor.extract(notif, user)
                    true_label = ObservedLabeler.label_from_behavior(notif, user)
                    
                    if true_label is not None:
                        test_data.append((features, true_label))
                except Exception as e:
                    pass
        
        if not test_data:
            return {"error": "No test data available"}
        
        # Evaluate
        from ml.model import ModelEvaluator
        metrics = ModelEvaluator.evaluate(model, test_data)        
        return metrics
    
    @staticmethod
    def get_training_stats(user):
        """
        Get statistics about available training data for a user.
        
        Returns:
            dict: Training data statistics
        """
        # Total notifications
        total_notifs = NotificationEvent.objects.filter(app__user=user).count()
        
        # Labeled notifications (user interacted)
        labeled_notifs = NotificationEvent.objects.filter(
            app__user=user,
            user_states__user=user
        ).filter(
            Q(user_states__opened_at__isnull=False) | 
            Q(user_states__dismissed_at__isnull=False)
        ).distinct().count()
        
        # Opened notifications
        opened_notifs = NotificationEvent.objects.filter(
            app__user=user,
            user_states__user=user,
            user_states__opened_at__isnull=False
        ).distinct().count()
        
        # Dismissed notifications
        dismissed_notifs = NotificationEvent.objects.filter(
            app__user=user,
            user_states__user=user,
            user_states__dismissed_at__isnull=False,
            user_states__opened_at__isnull=True  # Dismissed WITHOUT opening
        ).distinct().count()
        
        # Calculate rates
        label_rate = labeled_notifs / total_notifs if total_notifs > 0 else 0
        open_rate = opened_notifs / labeled_notifs if labeled_notifs > 0 else 0
        
        # Determine readiness
        should_train, reason = ModelRetrainer.should_retrain(user)
        
        return {
            "total_notifications": total_notifs,
            "labeled_notifications": labeled_notifs,
            "opened_notifications": opened_notifs,
            "dismissed_notifications": dismissed_notifs,
            "label_rate": label_rate,
            "open_rate": open_rate,
            "should_train": should_train,
            "train_reason": reason,
        }


class DatasetAnalyzer:
    """
    Analyze dataset quality for debugging.
    """
    
    @staticmethod
    def analyze_dataset(dataset):
        """
        Analyze a training dataset.
        
        Returns:
            dict: Dataset statistics
        """
        import numpy as np
        
        if not dataset:
            return {"error": "Empty dataset"}
        
        labels = np.array([label for _, label in dataset])
        
        # Feature analysis
        feature_keys = set()
        for features, _ in dataset:
            feature_keys.update(features.keys())
        
        # App distribution
        apps = [features.get('app', 'unknown') for features, _ in dataset]
        from collections import Counter
        app_counts = Counter(apps)
        
        return {
            "size": len(dataset),
            "num_features": len(feature_keys),
            "label_stats": {
                "mean": float(labels.mean()),
                "std": float(labels.std()),
                "min": float(labels.min()),
                "max": float(labels.max()),
                "median": float(np.median(labels)),
            },
            "engagement_distribution": {
                "high (>=0.7)": float((labels >= 0.7).mean()),
                "medium (0.4-0.7)": float(((labels >= 0.4) & (labels < 0.7)).mean()),
                "low (<0.4)": float((labels < 0.4).mean()),
            },
            "top_apps": dict(app_counts.most_common(10)),
        }
    
    @staticmethod
    def compare_synthetic_vs_real(user):
        """
        Compare synthetic data distribution to user's real data.
        
        Returns:
            dict: Comparison statistics
        """
        # Get real data
        real_dataset = []
        notifications = NotificationEvent.objects.filter(
            app__user=user
        ).select_related('app').prefetch_related('user_states')[:100]
        
        for notif in notifications:
            if ObservedLabeler.can_be_labeled(notif, user):
                features = FeatureExtractor.extract(notif, user)
                label = ObservedLabeler.label_from_behavior(notif, user)
                if label is not None:
                    real_dataset.append((features, label))
        
        # Generate synthetic data
        user_apps = list(set(f['app'] for f, _ in real_dataset))
        synthetic_dataset = SyntheticDataGenerator.generate_for_cold_start(
            apps=user_apps if user_apps else None,
            n=100
        )
        
        # Analyze both
        real_stats = DatasetAnalyzer.analyze_dataset(real_dataset)
        synthetic_stats = DatasetAnalyzer.analyze_dataset(synthetic_dataset)
        
        return {
            "real": real_stats,
            "synthetic": synthetic_stats,
            "label_mean_diff": abs(real_stats["label_stats"]["mean"] - synthetic_stats["label_stats"]["mean"]),
            "label_std_diff": abs(real_stats["label_stats"]["std"] - synthetic_stats["label_stats"]["std"]),
        }