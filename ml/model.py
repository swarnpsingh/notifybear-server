# ml/model.py

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight


class UserNotificationModel:
    def __init__(self):
        self.pipeline = None

    def train(self, dataset):
        X = [x for x, y in dataset]
        y = [y for x, y in dataset]

        cat_features = ["app"]
        num_features = ["hour", "has_urgent", "has_promo"]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ])
        
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        class_weight = dict(zip(classes, weights))

        clf = LogisticRegression(
            max_iter=500,
            multi_class="auto",
            class_weight=class_weight
        )

        self.pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", clf),
        ])

        self.pipeline.fit(X, y)

    def predict(self, features):
        return int(self.pipeline.predict([features])[0])

    def save(self, path):
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
