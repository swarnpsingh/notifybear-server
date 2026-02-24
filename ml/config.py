# The precise 8-feature V5 Edge Vector
FEATURE_NAMES = [
    "raw_hour", 
    "channel_historical_ctr", 
    "is_active_session", 
    "time_since_last_notif_sec", 
    "digit_density", 
    "title_body_ratio", 
    "exclamation_density", 
    "notifs_past_24h"
]

# Bayesian Smoothing Constants
# Used to mathematically stabilize CTR for brand new apps/channels
GLOBAL_OPEN_RATE_PRIOR = 0.15
SMOOTHING_WEIGHT = 10.0