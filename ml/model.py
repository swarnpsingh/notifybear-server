import joblib
import numpy as np
import logging
from sklearn.ensemble import HistGradientBoostingClassifier

# ONNX Imports
try:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)

class NotificationClassifier:
    def __init__(self, model_path="notification_model.pkl"):
        self.model_path = model_path
        self.onnx_path = model_path.replace(".pkl", ".onnx")
        
        # Classifier with balanced weights for extreme data skew
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=6,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False

    def train(self, X, y):
        X = np.array(X, dtype=np.float32)
        # Ensure labels are strictly binary 0 or 1
        y = np.array(y, dtype=int) 
        
        if len(X) == 0:
            logger.warning("No data. Skipping.")
            return

        logger.info(f"Training on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        
        score = self.model.score(X, y)
        logger.info(f"Training Accuracy: {score:.4f}")
        
        self.save()
        self.save_onnx()

    def predict(self, feature_vector):
        if not self.is_trained: return 0.5
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
            
        # [0][1] gets the probability of class 1.0 (Clicked)
        prob = self.model.predict_proba(feature_vector)[0][1]
        return float(prob)

    def save(self):
        joblib.dump(self.model, self.model_path)
        logger.info(f"Python model saved to {self.model_path}")

    def save_onnx(self):
        if not ONNX_AVAILABLE: return
        if not self.is_trained: return

        n_features = len(config.FEATURE_NAMES)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        try:
            onnx_model = to_onnx(self.model, initial_types=initial_type, target_opset=12)
            with open(self.onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"ONNX model saved to {self.onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")

    def load(self):
        try:
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning("No model found. Cold start.")
            self.is_trained = False