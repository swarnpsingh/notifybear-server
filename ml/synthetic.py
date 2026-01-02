import random
import os
os.makedirs("models", exist_ok=True)

HIGH_APPS = {"whatsapp", "messages", "phone", "gmail", "slack", "teams"}
LOW_APPS = {"instagram", "youtube", "facebook", "reddit", "tiktok"}

URGENT_WORDS = ["otp", "code", "urgent", "verify", "login"]
PROMO_WORDS = ["sale", "offer", "discount", "deal", "win"]

def label_logic(app, text, hour):
    app = app.lower()
    text = (text or "").lower()

    if any(w in text for w in URGENT_WORDS):
        return 2  # High

    if any(w in text for w in PROMO_WORDS):
        return 0  # Low

    if app in HIGH_APPS:
        return 2 if 8 <= hour <= 22 else 1

    if app in LOW_APPS:
        return 0

    return 1


def generate_synthetic_for_apps(apps, n=1000):
    data = []

    for _ in range(n):
        app = random.choice(apps)
        hour = random.randint(0, 23)

        if random.random() < 0.2:
            text = random.choice(URGENT_WORDS)
        elif random.random() < 0.3:
            text = random.choice(PROMO_WORDS)
        else:
            text = "generic message"

        label = label_logic(app, text, hour)

        features = {
            "app": app,
            "hour": hour,
            "has_urgent": int(any(w in text for w in URGENT_WORDS)),
            "has_promo": int(any(w in text for w in PROMO_WORDS))
        }

        data.append((features, label))

    return data
