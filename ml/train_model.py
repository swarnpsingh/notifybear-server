# ml/train_model.py (UPDATED - for backward compatibility only)

"""
Legacy training function - kept for backward compatibility.

NOTE: This is DEPRECATED. Use ml.service.MLService instead.

The new architecture uses:
- Global model (not per-user)
- Real user data (not pure synthetic)
- ml/shared_model.py and ml/service.py
"""

import tempfile
import warnings

from ml.synthetic import SyntheticDataGenerator
from ml.model import UserNotificationModel



def train_for_user(user_id, apps, total=1000):
    """
    DEPRECATED: Train a per-user model with synthetic data.
    
    This function is kept for backward compatibility only.
    
    New code should use:
        from ml.service import MLService
        MLService.retrain_global_model()
    
    Args:
        user_id: User ID (ignored in new architecture)
        apps: List of app package names
        total: Dataset size (default 1000)
    
    Returns:
        str: Path to saved model
    """
    warnings.warn(
        "train_for_user() is deprecated. Use MLService.retrain_global_model() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Generate synthetic data (old way)
    dataset = SyntheticDataGenerator.generate_for_cold_start(apps, n=total)
    
    # Train model (old way)
    model = UserNotificationModel(model_type='ridge')
    model.train(dataset, validate=False)
    
    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    model.save(tmp.name)  # Use .joblib, not .onnx (ONNX conversion is buggy)
    tmp.close()
    
    return tmp.name