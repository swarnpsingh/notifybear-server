# test_ml_logic.py
import numpy as np
from ml.model import NotificationClassifier
from ml.features import FeatureEngineer

# 1. Initialize the classifier
classifier = NotificationClassifier(model_path="local_test_model.pkl")

# 2. Create a "Mock" notification object to test extraction
class MockNotification:
    def __init__(self):
        self.title = "Urgent: Your OTP is 1234"
        self.text = "Do not share this code."
        self.post_time = pd.Timestamp.now()
        self.app_id = "com.test.app"

# 3. Test extraction
# (Ensure your FeatureEngineer doesn't call the DB in this specific test)
dummy_stats = {"notifs_past_24h": 5, "app_com.test.app_ctr": 0.2}
dummy_context = {"sec_since_last_action": 10.0}

vector = FeatureEngineer.extract(MockNotification(), dummy_stats, dummy_context)
print(f"✅ Extracted V5 Vector: {vector}")

# 4. Run a dummy training cycle
X = np.random.rand(60, 8).astype(np.float32) # 60 samples to pass the 50-sample gate
y = np.random.randint(0, 2, 60)
classifier.train(X, y)
print("✅ Local training successful!")