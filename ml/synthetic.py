import random
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic training data for cold start scenarios.
    
    Philosophy:
    - Synthetic data is a TEMPORARY crutch, not ground truth
    - Labels should be UNCERTAIN (we're guessing)
    - Real data should replace synthetic ASAP
    """
    
    # Realistic app distribution (based on typical Android usage)
    APP_DISTRIBUTION = {
        'com.whatsapp': 0.25,
        'com.instagram.android': 0.15,
        'com.google.android.gm': 0.10,
        'com.facebook.katana': 0.10,
        'com.snapchat.android': 0.08,
        'com.google.android.youtube': 0.07,
        'com.android.phone': 0.05,
        'com.google.android.calendar': 0.05,
        'com.slack': 0.03,
        'com.twitter.android': 0.04,
        'com.spotify.music': 0.03,
        'com.king.candycrushsaga': 0.05,
    }
    
    # Realistic hour distribution (when people check phones)
    HOUR_DISTRIBUTION = {
        0: 0.005, 1: 0.005, 2: 0.005, 3: 0.005, 4: 0.005, 5: 0.01, 6: 0.02,
        7: 0.05, 8: 0.06, 9: 0.05, 10: 0.05, 11: 0.05, 12: 0.06,
        13: 0.05, 14: 0.04, 15: 0.04, 16: 0.04, 17: 0.05, 18: 0.06,
        19: 0.07, 20: 0.07, 21: 0.06, 22: 0.04, 23: 0.02,
    }
    
    # Text templates for different notification types
    TEMPLATES = {
        'urgent': [
            "OTP: {code}",
            "Verification code: {code}",
            "Your login code is {code}",
            "URGENT: Action required",
            "Important: Please verify your account",
            "Security alert: Confirm this was you",
        ],
        'promo': [
            "50% OFF Sale! Limited time only",
            "Exclusive offer just for you",
            "Flash sale ending soon!",
            "Free shipping on all orders today",
            "Win a prize - Enter now!",
            "Special discount: Use code SAVE20",
        ],
        'message': [
            "{person}: Hey, are you free?",
            "{person}: Can we talk?",
            "New message from {person}",
            "{person}: Check this out",
            "You have a new message",
            "{person}: Where are you?",
        ],
        'social': [
            "{person} liked your photo",
            "{person} commented on your post",
            "{person} started following you",
            "You have 5 new notifications",
            "{person} tagged you in a photo",
            "{person} mentioned you",
        ],
        'email': [
            "Meeting reminder: Team sync in 30 minutes",
            "New email from {person}",
            "Project update: Q4 results",
            "Weekly report is ready",
            "Action required: Review document",
        ],
        'news': [
            "Breaking: Major event happened",
            "Update on {topic}",
            "Trending now: {topic}",
            "Daily digest: Top stories",
        ],
        'system': [
            "App update available",
            "Storage almost full",
            "Connected to WiFi",
            "Battery low - 15% remaining",
        ],
    }
    
    @classmethod
    def generate_for_cold_start(cls, apps=None, n=200):
        """
        Generate minimal synthetic dataset for cold start.
        
        Args:
            apps: List of app package names (or None for default set)
            n: Number of samples (default 200, much less than before)
        
        Returns:
            List of (features, engagement_score) tuples
        """
        if apps is None:
            apps = list(cls.APP_DISTRIBUTION.keys())
        
        logger.info(f"Generating {n} synthetic samples for cold start")
        
        # Build weighted app list
        weighted_apps = []
        for app in apps:
            weight = cls.APP_DISTRIBUTION.get(app, 0.05)
            count = int(weight * n)
            weighted_apps.extend([app] * max(1, count))
        
        # Pad to n if needed
        while len(weighted_apps) < n:
            weighted_apps.append(random.choice(apps))
        
        random.shuffle(weighted_apps)
        weighted_apps = weighted_apps[:n]  # Trim to exact n
        
        dataset = []
        
        for i in range(n):
            app = weighted_apps[i]
            
            # Sample hour from realistic distribution
            hour = random.choices(
                list(cls.HOUR_DISTRIBUTION.keys()),
                weights=list(cls.HOUR_DISTRIBUTION.values())
            )[0]
            
            # Generate realistic text
            text = cls._generate_text_for_app(app)
            
            # Create features
            features = cls._create_features(app, hour, text)
            
            # Generate WEAK label (with uncertainty)
            engagement_score = cls._generate_weak_label(features)
            
            dataset.append((features, engagement_score))
        
        logger.info(f"Generated {len(dataset)} synthetic samples")
        return dataset
    
    @classmethod
    def _generate_text_for_app(cls, app):
        """Generate realistic notification text based on app type."""
        app_lower = app.lower()
        
        # Messaging apps
        if 'whatsapp' in app_lower or 'telegram' in app_lower or 'messaging' in app_lower:
            template = random.choice(cls.TEMPLATES['message'])
            return template.format(
                person=random.choice(['Mom', 'Dad', 'Boss', 'Alice', 'Bob', 'Sarah', 'Friend'])
            )
        
        # Social media
        elif 'instagram' in app_lower or 'facebook' in app_lower or 'snapchat' in app_lower or 'twitter' in app_lower:
            template = random.choice(cls.TEMPLATES['social'])
            return template.format(
                person=random.choice(['john_doe', 'alice123', 'bob_smith', 'user456'])
            )
        
        # Email
        elif 'gmail' in app_lower or 'mail' in app_lower or 'outlook' in app_lower:
            rand = random.random()
            if rand < 0.2:
                # 20% urgent (OTP, verification)
                template = random.choice(cls.TEMPLATES['urgent'])
                return template.format(code=random.randint(100000, 999999))
            elif rand < 0.5:
                # 30% promo
                return random.choice(cls.TEMPLATES['promo'])
            else:
                # 50% regular email
                template = random.choice(cls.TEMPLATES['email'])
                return template.format(person=random.choice(['John', 'Sarah', 'Manager', 'Team']))
        
        # Calendar
        elif 'calendar' in app_lower:
            return random.choice([
                'Meeting in 15 minutes',
                'Event reminder: Team standup',
                'Upcoming: Doctor appointment',
            ])
        
        # Phone/Calls
        elif 'phone' in app_lower:
            return random.choice([
                'Missed call from Mom',
                'Incoming call from Unknown',
                'Voicemail from +1234567890',
            ])
        
        # Entertainment
        elif 'youtube' in app_lower or 'spotify' in app_lower or 'netflix' in app_lower:
            return random.choice([
                'New video from your subscription',
                'Recommended for you',
                'Your playlist has been updated',
            ])
        
        # Games
        elif 'game' in app_lower or 'king' in app_lower or 'clash' in app_lower:
            return random.choice([
                'New level unlocked!',
                'Daily reward available',
                'Your turn in the game',
                'Achievement unlocked!',
            ])
        
        # News
        elif 'news' in app_lower:
            template = random.choice(cls.TEMPLATES['news'])
            return template.format(
                topic=random.choice(['Technology', 'Business', 'Sports', 'Politics'])
            )
        
        # Default/System
        else:
            return random.choice(cls.TEMPLATES['system'])
    
    @classmethod
    def _create_features(cls, app, hour, text):
        """
        Create feature dict matching real feature extraction.
        
        Must match FeatureExtractor.extract() output!
        """
        from ml.features import FeatureExtractor
        
        text_lower = text.lower()
        day_of_week = random.randint(0, 6)
        
        features = {
            # Temporal features
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": int(day_of_week >= 5),
            "is_work_hours": int(9 <= hour <= 17),
            "is_sleep_hours": int(hour < 7 or hour > 22),
            "is_morning": int(6 <= hour < 12),
            "is_afternoon": int(12 <= hour < 18),
            "is_evening": int(18 <= hour < 23),
            
            # Text features
            "text_length": len(text),
            "title_length": len(text.split(':')[0]) if ':' in text else len(text),
            "has_urgent": int(any(w in text_lower for w in FeatureExtractor.URGENT_WORDS)),
            "has_promo": int(any(w in text_lower for w in FeatureExtractor.PROMO_WORDS)),
            "has_person": int(any(w in text_lower for w in FeatureExtractor.PERSON_WORDS)),
            "has_question": int(any(w in text_lower for w in FeatureExtractor.QUESTION_WORDS)),
            "has_numbers": int(any(c.isdigit() for c in text)),
            "has_url": int('http' in text_lower or 'www.' in text_lower),
            "is_short": int(len(text) < 50),
            "is_long": int(len(text) > 200),
            "word_count": len(text.split()),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())),
            
            # App features
            "app": app,
            "app_priority": FeatureExtractor.APP_PRIORITY.get(app, 0.5),
            
            # User behavioral features (use neutral defaults for synthetic)
            "app_open_rate": 0.5,  # Neutral: no prior knowledge
            "app_avg_reaction_time": 300.0,  # 5 minutes default
            "user_global_open_rate": 0.5,  # Neutral
            "notifications_today": 0,
            "notifications_this_hour": 0,
            "time_since_last_notif": 3600.0,  # 60 min
            "is_first_of_day": 0,
            
            # Channel
            "channel_id": "unknown",
            
            # Derived features
            "is_likely_otp": int(
                any(w in text_lower for w in FeatureExtractor.URGENT_WORDS) and
                any(c.isdigit() for c in text) and
                len(text) < 50
            ),
            "is_likely_promo": int(
                any(w in text_lower for w in FeatureExtractor.PROMO_WORDS) or
                (len(text) > 200 and ('http' in text_lower or 'www.' in text_lower))
            ),
            "is_high_priority_app": int(FeatureExtractor.APP_PRIORITY.get(app, 0.5) > 0.7),
            "is_notification_burst": 0,
            "is_rare_notification": 0,
        }
        
        return features
    
    @classmethod
    def _generate_weak_label(cls, features):
        """
        Generate UNCERTAIN engagement score for synthetic data.
        
        Key principle: We're GUESSING, so labels should reflect uncertainty.
        
        Returns:
            float: 0.0-1.0 engagement score with added noise
        """
        # Start with neutral prior
        base_score = 0.5
        
        # Only VERY STRONG signals move the needle
        
        # OTP/Verification (most confident prediction)
        if features.get("is_likely_otp", 0):
            base_score = 0.7  # Likely high engagement (not 1.0, we're not certain!)
        
        # Promotional (confident it's low priority)
        elif features.get("is_likely_promo", 0):
            base_score = 0.25  # Likely low engagement
        
        # High priority app + work hours
        elif features.get("is_high_priority_app", 0) and features.get("is_work_hours", 0):
            base_score = 0.65  # Moderately likely important
        
        # High priority app + sleep hours
        elif features.get("is_high_priority_app", 0) and features.get("is_sleep_hours", 0):
            base_score = 0.55  # Could be urgent, but awkward timing
        
        # Low priority app
        elif features.get("app_priority", 0.5) < 0.3:
            base_score = 0.35  # Likely low priority
        
        # Add significant noise to express UNCERTAINTY
        # This is key: synthetic labels should be fuzzy
        noise = random.gauss(0, 0.25)  # ±25% uncertainty (high!)
        score = base_score + noise
        
        # Clip to valid range
        score = max(0.0, min(1.0, score))
        
        return score
    
    @classmethod
    def analyze_synthetic_distribution(cls, dataset):
        """
        Analyze the distribution of synthetic data (for debugging).
        
        Returns:
            dict: Statistics about the synthetic dataset
        """
        import numpy as np
        
        labels = [label for features, label in dataset]
        labels_array = np.array(labels)
        
        return {
            "count": len(labels),
            "mean": float(labels_array.mean()),
            "std": float(labels_array.std()),
            "min": float(labels_array.min()),
            "max": float(labels_array.max()),
            "high_engagement_rate": float((labels_array >= 0.7).mean()),
            "low_engagement_rate": float((labels_array <= 0.3).mean()),
        }


# Example usage for debugging
if __name__ == "__main__":
    # Generate sample dataset
    dataset = SyntheticDataGenerator.generate_for_cold_start(n=200)
    
    # Analyze distribution
    stats = SyntheticDataGenerator.analyze_synthetic_distribution(dataset)
    
    print("Synthetic Data Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean engagement: {stats['mean']:.3f}")
    print(f"  Std dev: {stats['std']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  High engagement rate: {stats['high_engagement_rate']:.1%}")
    print(f"  Low engagement rate: {stats['low_engagement_rate']:.1%}")
    
    # Show a few examples
    print("\nSample synthetic notifications:")
    for i, (features, label) in enumerate(dataset[:5]):
        print(f"\n{i+1}. App: {features['app']}")
        print(f"   Hour: {features['hour']}")
        print(f"   Has urgent: {features['has_urgent']}")
        print(f"   Has promo: {features['has_promo']}")
        print(f"   Engagement score: {label:.3f}")