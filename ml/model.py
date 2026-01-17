"""
ML model for notification engagement prediction.

Key changes:
- Regression instead of classification (predicts continuous 0.0-1.0 score)
- Better for ranking (preserves ordering)
- Proper validation metrics (RMSE, not accuracy)
"""

import joblib
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

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

class UserNotificationModel:
    def __init__(self, model_type="gbm"):
        self.model_type = model_type
        if model_type == "gbm":
            self.model = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=200,
                random_state=42
            )
        else:
            raise ValueError("Only 'gbm' model_type is supported for now")

    def train(self, dataset, validate=True, test_size=0.2, random_state=42):
        if not dataset or len(dataset) == 0:
            raise ValueError("Empty dataset — cannot train")

        X, y = zip(*dataset)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Clip labels for numerical stability
        y = np.clip(y, 0.05, 0.95)

        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, y_train = X, y
            X_val = y_val = None

        self.model.fit(X_train, y_train)
        
        self.pipeline = self.model
        self.feature_names = CANONICAL_FEATURE_ORDER
        
        metrics = {}

        if validate:
            preds = self.model.predict(X_val)

            mae = mean_absolute_error(y_val, preds)
            #mse = mean_squared_error(y_val, preds)
            #rmse = np.sqrt(mse)
            rmse = root_mean_squared_error(y_val, preds)

            # Rank correlation (Spearman)
            try:
                from scipy.stats import spearmanr
                rank_corr = spearmanr(y_val, preds).correlation
            except Exception:
                rank_corr = None

            metrics = {
                "val_mae": float(mae),
                "val_rmse": float(rmse),
                "rank_correlation": float(rank_corr) if rank_corr is not None else None,
                "num_train": len(X_train),
                "num_val": len(X_val),
            }
            self.val_mae = metrics.get("val_mae")
            self.val_rmse = metrics.get("val_rmse")
            self.train_rmse = float(
                root_mean_squared_error(y_train, self.model.predict(X_train))
            )
        self.is_trained = True

        return metrics
    
    def predict(self, features):
        vec = [float(features.get(k, 0.0)) for k in CANONICAL_FEATURE_ORDER]
        arr = np.array([vec], dtype=np.float32)
        score = float(self.model.predict(arr)[0])
        return max(0.0, min(1.0, score))
    
    def predict_batch(self, features_list):
        X = []

        for f in features_list:
            vec = [float(f.get(k, 0.0)) for k in CANONICAL_FEATURE_ORDER]
            X.append(vec)

        X = np.array(X, dtype=np.float32)
        scores = self.model.predict(X)
        return np.clip(scores, 0.0, 1.0)

    
    def predict_priority_bucket(self, features):
        """
        Convert engagement score to priority bucket (for backward compatibility).
        
        Args:
            features: dict of features
        
        Returns:
            int: 0 (low), 1 (medium), 2 (high)
        """
        score = self.predict(features)
        
        # Convert continuous score to buckets
        if score >= 0.7:
            return 2  # High priority
        elif score >= 0.4:
            return 1  # Medium priority
        else:
            return 0  # Low priority
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance (for GBM models).
        
        Returns:
            dict: {feature_name: importance_score}
        """        
        if self.model_type != 'gbm':
            return {}
        
        try:
            
            # Get importances
            importances = self.model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:top_n]
            
            return {
                CANONICAL_FEATURE_ORDER[i]: float(importances[i])
                for i in indices
            }        
        except Exception as e:
            return {}
    
    def save(self, path):
        """
        Save model to disk.
        
        Args:
            path: File path (e.g., 'model.joblib')
        """
        if self.pipeline is None:
            raise ValueError("No model to save")
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'train_rmse': self.train_rmse,
            'val_rmse': self.val_rmse,
            'val_mae': self.val_mae,
        }
        
        joblib.dump(model_data, path, compress=3)

    def save_onnx(self, path, feature_count):
        initial_type = [('float_input', FloatTensorType([None, feature_count]))]
        onnx_model = skl2onnx.convert_sklearn(self.model, initial_types=initial_type)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    def load(self, path):
        """
        Load model from disk.
        
        Args:
            path: File path
        """
        model_data = joblib.load(path)
        
        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data.get('model_type', 'ridge')
        self.train_rmse = model_data.get('train_rmse')
        self.val_rmse = model_data.get('val_rmse')
        self.val_mae = model_data.get('val_mae')
    
    def summary(self):
        """
        Get model summary for monitoring.
        
        Returns:
            dict: Model information
        """
        if self.pipeline is None:
            return {"trained": False}
        
        summary = {
            "trained": True,
            "model_type": self.model_type,
            "num_features": len(self.feature_names) if self.feature_names else 0,
            "train_rmse": float(self.train_rmse) if self.train_rmse else None,
            "val_rmse": float(self.val_rmse) if self.val_rmse else None,
            "val_mae": float(self.val_mae) if self.val_mae else None,
        }
        
        return summary


class ModelEvaluator:
    """
    Evaluate model performance on held-out data.
    """
    
    @staticmethod
    def evaluate(model, test_data):
        """
        Evaluate model on test data.
        
        Args:
            model: Trained UserNotificationModel
            test_data: List of (features, label) tuples
        
        Returns:
            dict: Evaluation metrics
        """
        if not test_data:
            return {"error": "Empty test data"}
        
        X_test = [x for x, y in test_data]
        y_test = np.array([y for x, y in test_data])
        
        # Predict
        y_pred = model.predict_batch(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Ranking metric
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(y_test, y_pred)
        
        # Bucketed accuracy (for comparison)
        y_test_buckets = np.digitize(y_test, bins=[0.4, 0.7])
        y_pred_buckets = np.digitize(y_pred, bins=[0.4, 0.7])
        bucket_accuracy = (y_test_buckets == y_pred_buckets).mean()
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "rank_correlation": float(rank_corr),
            "bucket_accuracy": float(bucket_accuracy),
            "test_size": len(test_data),
        }
    
    @staticmethod
    def compare_models(model1, model2, test_data):
        """
        Compare two models on the same test data.
        
        Returns:
            dict: Comparison results
        """
        metrics1 = ModelEvaluator.evaluate(model1, test_data)
        metrics2 = ModelEvaluator.evaluate(model2, test_data)
        
        return {
            "model1": metrics1,
            "model2": metrics2,
            "rmse_improvement": metrics1["rmse"] - metrics2["rmse"],
            "rank_corr_improvement": metrics2["rank_correlation"] - metrics1["rank_correlation"],
        }