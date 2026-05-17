from datetime import timedelta
import numpy as np
from django.utils import timezone

from ml.models import TrainingFeature

class FeatureEngineer:
    
    @staticmethod
    def fetch_training_rows(user, apps=None, lookback_days=30, max_samples=1000, min_unused=50):

        cutoff = timezone.now() - timedelta(days=lookback_days)

        base_qs = TrainingFeature.objects.filter(user=user,created_at__gte=cutoff).exclude(label__isnull=True)

        if apps:
            base_qs = base_qs.filter(package_name__in=apps)

        base_qs = base_qs.filter(feature_version=1)

        unused_qs = (base_qs.filter(used_for_training=False).order_by("-created_at")[:min_unused])
        unused_ids = list(unused_qs.values_list("id", flat=True))
        remaining = max(0,max_samples - len(unused_ids))

        historical_qs = (base_qs.exclude(id__in=unused_ids).order_by("-created_at")[:remaining])

        final_rows = list(unused_qs) + list(historical_qs)
        import random
        random.shuffle(final_rows)

        for row in final_rows:
            raw = row.features
            if (not isinstance(raw, (list, tuple))or len(raw) != 16):
                continue
            try:
                vector = np.array(
                    raw,
                    dtype=np.float32
                )
            except Exception:
                continue

            yield row.id, vector, float(row.label)