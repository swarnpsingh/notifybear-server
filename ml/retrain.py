# ml/retrain.py

import random
from Notifications.models import NotificationEvent
from ml.synthetic import generate_synthetic_for_apps
from ml.model import UserNotificationModel
import tempfile

URGENT_WORDS = ["otp", "urgent", "verify", "code"]
PROMO_WORDS = ["sale", "offer", "discount"]
HIGH_REACTION_THRESHOLD = 30  # seconds


from django.db.models import Avg


def extract_features_and_label(n):
    text = ((n.title or "") + " " + (n.text or "")).lower()

    features = {
        "app": n.app.package_name,
        "hour": n.post_time.hour,
        "has_urgent": int(any(w in text for w in URGENT_WORDS)),
        "has_promo": int(any(w in text for w in PROMO_WORDS)),
    }

    state = n.user_states.first()
    if not state:
        return None

    score = 0

    # --- Behavioral ---
    if state.opened_at:
        score += 2
        reaction = (state.opened_at - n.post_time).total_seconds()
        if reaction <= 30:
            score += 1

    if state.dismissed_at:
        score -= 1

    # --- App-level behavior ---
    app_open_rate = n.app.daily_aggregates.aggregate(avg=Avg("open_rate"))["avg"] or 0
    if app_open_rate > 0.5:
        score += 1

    # --- Semantic ---
    if features["has_urgent"]:
        score += 1
    if features["has_promo"]:
        score -= 1

    # --- Time context ---
    if n.post_time.hour < 7 or n.post_time.hour > 22:
        score -= 0.5

    # --- Bucket ---
    if score <= 0:
        label = 0
    elif score <= 2:
        label = 1
    else:
        label = 2

    return features, label



def build_dataset_for_user(user, apps, total=1000):
    qs = NotificationEvent.objects.filter(app__user=user).order_by("-post_time")

    real_data = []
    for n in qs:
        item = extract_features_and_label(n)
        if item:
            real_data.append(item)

    random.shuffle(real_data)

    if len(real_data) >= total:
        dataset = real_data[:total]
    else:
        remaining = total - len(real_data)
        synthetic = generate_synthetic_for_apps(apps, n=remaining)
        dataset = real_data + synthetic

    random.shuffle(dataset)
    return dataset


def retrain_model_for_user(user, apps, total=1000):
    dataset = build_dataset_for_user(user, apps, total=total)

    if not dataset:
        dataset = generate_synthetic_for_apps(apps, n=total)

    model = UserNotificationModel()
    model.train(dataset)

    #path = f"models/user_{user.id}.joblib"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    model.save(tmp.name)
    tmp.close()

    return tmp.name