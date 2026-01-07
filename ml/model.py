"""
ML model for notification engagement prediction.

Key changes:
- Regression instead of classification (predicts continuous 0.0-1.0 score)
- Better for ranking (preserves ordering)
- Proper validation metrics (RMSE, not accuracy)
"""

import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class UserNotificationModel:
    """
    Regression model for predicting notification engagement.
    
    Predicts: float 0.0-1.0 (engagement score)
    Higher score = user more likely to engage quickly
    """
    
    def __init__(self, model_type='ridge'):
        """
        Initialize model.
        
        Args:
            model_type: 'ridge' (fast, simple) or 'gbm' (better, slower)
        """
        self.pipeline = None
        self.feature_names = None
        self.model_type = model_type
        self.train_rmse = None
        self.val_rmse = None
        self.val_mae = None
    
    def train(self, dataset, validate=True):
        """
        Train regression model on engagement data.
        
        Args:
            dataset: List of (features_dict, engagement_score) tuples
                     engagement_score is float 0.0-1.0
            validate: Whether to perform train/val split
        
        Returns:
            dict: Training metrics
        """
        if not dataset:
            raise ValueError("Empty dataset")
        
        # Unpack dataset
        X_dict = [x for x, y in dataset]
        y = np.array([y for x, y in dataset], dtype=float)
        X = pd.DataFrame(X_dict)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Engagement distribution: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}")
        
        # Check for label quality
        if y.std() < 0.05:
            logger.warning("Very low label variance - all labels are similar!")
        
        # Define feature types
        cat_features = ["app", "channel_id"]
        num_features = [f for f in X.columns if f not in cat_features]
        
        logger.info(f"Categorical features: {len(cat_features)}")
        logger.info(f"Numerical features: {len(num_features)}")
        
        # Build preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
                ("num", StandardScaler(), num_features),
            ],
            remainder='drop'
        )
        
        # Choose regressor
        if self.model_type == 'ridge':
            regressor = Ridge(
                alpha=1.0,
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'gbm':
            regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ("prep", preprocessor),
            ("reg", regressor),
        ])
        
        # Train with validation split
        if validate and len(X) >= 100:
            logger.info("Training with validation split")
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42
            )
            
            # Train
            self.pipeline.fit(X_train, y_train)
            
            # Predictions
            train_pred = self.pipeline.predict(X_train)
            val_pred = self.pipeline.predict(X_val)
            
            # Clip predictions to valid range
            train_pred = np.clip(train_pred, 0.0, 1.0)
            val_pred = np.clip(val_pred, 0.0, 1.0)
            
            # Calculate metrics
            self.train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            self.val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            self.val_mae = mean_absolute_error(y_val, val_pred)
            
            # Ranking metric: Spearman correlation
            from scipy.stats import spearmanr
            rank_corr, _ = spearmanr(y_val, val_pred)
            
            logger.info(f"Train RMSE: {self.train_rmse:.4f}")
            logger.info(f"Val RMSE: {self.val_rmse:.4f}")
            logger.info(f"Val MAE: {self.val_mae:.4f}")
            logger.info(f"Val Rank Correlation: {rank_corr:.4f}")
            
            # Check for overfitting
            if self.train_rmse < self.val_rmse * 0.7:
                logger.warning("Possible overfitting detected (train RMSE much lower than val RMSE)")
            
            return {
                "train_rmse": float(self.train_rmse),
                "val_rmse": float(self.val_rmse),
                "val_mae": float(self.val_mae),
                "rank_correlation": float(rank_corr),
                "train_size": len(X_train),
                "val_size": len(X_val),
            }
        
        else:
            # Train on full dataset (no validation)
            logger.info("Training on full dataset (no validation split)")
            
            self.pipeline.fit(X, y)
            
            # Calculate train metrics only
            train_pred = self.pipeline.predict(X)
            train_pred = np.clip(train_pred, 0.0, 1.0)
            
            self.train_rmse = np.sqrt(mean_squared_error(y, train_pred))
            
            logger.info(f"Train RMSE: {self.train_rmse:.4f}")
            
            return {
                "train_rmse": float(self.train_rmse),
                "train_size": len(X),
            }
    
    def predict(self, features):
        """
        Predict engagement score for a single notification.
        
        Args:
            features: dict of features
        
        Returns:
            float: 0.0-1.0 (engagement score)
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Add missing features with safe defaults
        for fname in self.feature_names:
            if fname not in features_df.columns:
                if fname in ["app", "channel_id"]:
                    features_df[fname] = "unknown"
                else:
                    features_df[fname] = 0.0
        
        # Reorder columns to match training
        features_df = features_df[self.feature_names]
        
        try:
            # Predict
            score = float(self.pipeline.predict(features_df)[0])
            
            # Clip to valid range
            score = max(0.0, min(1.0, score))
            
            logger.debug(f"Predicted engagement: {score:.3f} for app={features.get('app', 'unknown')}")
            
            return score
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return 0.5  # Default to medium priority on error
    
    def predict_batch(self, features_list):
        """
        Predict engagement scores for multiple notifications efficiently.
        
        Args:
            features_list: List of feature dicts
        
        Returns:
            np.array: Array of engagement scores
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add missing features
        for fname in self.feature_names:
            if fname not in features_df.columns:
                if fname in ["app", "channel_id"]:
                    features_df[fname] = "unknown"
                else:
                    features_df[fname] = 0.0
        
        # Reorder columns
        features_df = features_df[self.feature_names]
        
        try:
            # Predict
            scores = self.pipeline.predict(features_df)
            
            # Clip to valid range
            scores = np.clip(scores, 0.0, 1.0)
            
            logger.debug(f"Batch predicted {len(scores)} notifications")
            
            return scores
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            return np.full(len(features_list), 0.5)  # Default array
    
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
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type != 'gbm':
            logger.warning("Feature importance only available for GBM models")
            return {}
        
        try:
            # Get regressor from pipeline
            regressor = self.pipeline.named_steps['reg']
            
            # Get feature names after preprocessing
            preprocessor = self.pipeline.named_steps['prep']
            feature_names_out = preprocessor.get_feature_names_out()
            
            # Get importances
            importances = regressor.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:top_n]
            
            importance_dict = {
                feature_names_out[i]: float(importances[i])
                for i in indices
            }
            
            return importance_dict
        
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
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
        logger.info(f"Model saved to {path}")
    
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
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Features: {len(self.feature_names)}")
        if self.val_rmse:
            logger.info(f"Validation RMSE: {self.val_rmse:.4f}")
    
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