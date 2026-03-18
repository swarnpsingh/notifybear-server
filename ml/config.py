# The precise 8-feature V5 Edge Vector
FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "channel_ctr",
    "app_ctr",
    "is_active_session",
    "time_since_last_notif",
    "digit_density",
    "exclamation_density",
    "title_body_ratio",
    "notifs_past_24h",
    "is_otp",
    "is_transaction",
    "is_message",
    "is_promo",
    "is_urgent",
    "text_length",
]

# Bayesian Smoothing Constants
# Used to mathematically stabilize CTR for brand new apps/channels
GLOBAL_OPEN_RATE_PRIOR = 0.15
SMOOTHING_WEIGHT = 10.0