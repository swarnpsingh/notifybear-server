# ml/train_model.py

import tempfile
import os
from ml.synthetic import generate_synthetic_for_apps
from ml.model import UserNotificationModel

os.makedirs("models", exist_ok=True)


def train_for_user(user_id, apps, total=1000):
    dataset = generate_synthetic_for_apps(apps, n=total)

    model = UserNotificationModel()
    model.train(dataset)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    #path = f"models/user_{user_id}.joblib"
    model.save(tmp.name)
    tmp.close()

    return tmp.name
