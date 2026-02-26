from django.db import models
from django.conf import settings
from django.utils import timezone


class App(models.Model):
    """
    Represents an app (package) that generates notifications for a user.
    
    Normalized: All app metadata lives here, referenced by notifications.
    Changes rarely, belongs to one user.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name="apps"
    )
    package_name = models.CharField(max_length=255, db_index=True)
    app_label = models.CharField(max_length=255, blank=True, default="")
    icon_reference = models.CharField(
        max_length=512, 
        blank=True, 
        default="",
        help_text="URL or path to app icon, not binary data"
    )
    
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "package_name")
        indexes = [
            models.Index(fields=["user", "package_name"]),
        ]
        verbose_name = "App"
        verbose_name_plural = "Apps"

    def __str__(self):
        return f"{self.app_label or self.package_name}"


class NotificationEvent(models.Model):
    """
    IMMUTABLE notification event - represents "what happened".
    
    This is the ground truth for ML. Once created, never modified.
    Stores the content of a notification as it was posted.
    """
    app = models.ForeignKey(
        App, 
        on_delete=models.CASCADE, 
        related_name="notification_events"
    )
    
    # Unique identifier from Android notification system
    notif_key = models.CharField(max_length=256, db_index=True)
    
    # Notification metadata
    channel_id = models.CharField(max_length=200, null=True, blank=True)
    type = models.CharField(max_length=50, default="general")
    post_time = models.DateTimeField(default=timezone.now, db_index=True)
    
    # Primary text content (immutable)
    title = models.CharField(max_length=500, null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    big_text = models.TextField(null=True, blank=True)
    sub_text = models.CharField(max_length=500, null=True, blank=True)
    summary_text = models.CharField(max_length=500, null=True, blank=True)
    info_text = models.CharField(max_length=500, null=True, blank=True)
    
    # Multi-line text notifications (e.g., Gmail summary style)
    text_lines = models.TextField(
        null=True, 
        blank=True,
        help_text="JSON array of text lines"
    )
    
    # Image references (store URLs or base64, not recommended for large data)
    large_icon_base64 = models.TextField(null=True, blank=True)
    picture_base64 = models.TextField(null=True, blank=True)
    
    # Conversation metadata (for messaging apps)
    conversation_title = models.CharField(max_length=500, null=True, blank=True)
    people = models.TextField(
        null=True, 
        blank=True,
        help_text="JSON array of contact names/IDs"
    )
    
    # Content hash for deduplication (optional)
    content_hash = models.CharField(
        max_length=64, 
        null=True, 
        blank=True, 
        db_index=True,
        help_text="SHA256 hash of content for deduplication"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("app", "notif_key")
        indexes = [
            models.Index(fields=["app", "post_time"]),
            models.Index(fields=["app", "notif_key"]),
            models.Index(fields=["post_time"]),
        ]
        verbose_name = "Notification Event"
        verbose_name_plural = "Notification Events"

    def __str__(self):
        return f"{self.app.package_name} | {self.notif_key} | {self.title}"
    
    @property
    def user(self):
        """Convenience property to access user through app."""
        return self.app.user


class NotificationMessage(models.Model):
    """
    Individual chat messages inside messaging-style notifications.
    
    Represents structured message data (e.g., WhatsApp conversation).
    Belongs to a NotificationEvent.
    """
    notification_event = models.ForeignKey(
        NotificationEvent,
        on_delete=models.CASCADE,
        related_name="messages",
        default=None
    )
    
    sender = models.CharField(max_length=200, null=True, blank=True)
    message_text = models.TextField(null=True, blank=True)
    message_time = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["notification_event", "message_time"]),
        ]
        verbose_name = "Notification Message"
        verbose_name_plural = "Notification Messages"

    def __str__(self):
        preview = (self.message_text or "")[:30]
        return f"{self.sender}: {preview}"


class UserNotificationState(models.Model):
    """
    MUTABLE user interaction state - represents "what the user did".
    
    One row per (user, notification) pair.
    Stores behavioral signals: opened, dismissed, snoozed.
    Used for ranking and learning user preferences.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notification_states",
        db_index=True
    )
    notification_event = models.ForeignKey(
        NotificationEvent,
        on_delete=models.CASCADE,
        related_name="user_states",
        default=None
    )
    
    # User interaction timestamps (mutable)
    opened_at = models.DateTimeField(null=True, blank=True, db_index=True)
    dismissed_at = models.DateTimeField(null=True, blank=True)
    snoozed_until = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="For future snooze feature"
    )
    
    # Derived state (cached for performance)
    is_read = models.BooleanField(default=False, db_index=True)
    
    # ML scoring (computed and cached)
    ml_score = models.FloatField(
        null=True, 
        blank=True,
        db_index=True,
        help_text="Cached ML ranking score"
    )
    
    last_updated = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "notification_event")
        indexes = [
            models.Index(fields=["user", "is_read", "ml_score"]),
            models.Index(fields=["user", "opened_at"]),
            models.Index(fields=["notification_event", "user"]),
        ]
        verbose_name = "User Notification State"
        verbose_name_plural = "User Notification States"

    def __str__(self):
        user_label = getattr(self.user, "email", str(self.user))
        status = "read" if self.is_read else "unread"
        return f"{user_label} | {self.notification_event.app.package_name} | {status}"
    
    @property
    def reaction_time(self):
        """Time from notification post to user opening it."""
        if self.opened_at:
            return self.opened_at - self.notification_event.post_time
        return None
    
    def mark_opened(self, timestamp=None):
        """Mark notification as opened and update is_read."""
        self.opened_at = timestamp or timezone.now()
        self.is_read = True
        self.save(update_fields=["opened_at", "is_read", "last_updated"])
    
    def mark_dismissed(self, timestamp=None):
        """Mark notification as dismissed."""
        self.dismissed_at = timestamp or timezone.now()
        self.save(update_fields=["dismissed_at", "last_updated"])


class InteractionEvent(models.Model):
    """
    APPEND-ONLY interaction log - represents raw user actions.
    
    Event sourcing pattern: never updated or deleted.
    Used for analytics, ML feature engineering, and audit trail.
    Can reconstruct UserNotificationState if needed.
    """
    CLICK = "CLICK"
    SWIPE = "SWIPE"
    EXPAND = "EXPAND"
    APP_CANCEL = "APP_CANCEL"
    OTHER = "OTHER"
    
    INTERACTION_TYPES = [
        (CLICK, "Click"),
        (SWIPE, "Swipe"),
        (EXPAND, "Expand"),
        (APP_CANCEL, "App Cancel"),
        (OTHER, "Other"),
    ]
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="interaction_events"
    )
    notification_event = models.ForeignKey(
        NotificationEvent,
        on_delete=models.CASCADE,
        related_name="interaction_events",
        default=None
    )
    
    # Interaction details
    interaction_type = models.CharField(max_length=16, choices=INTERACTION_TYPES)
    timestamp = models.DateTimeField(db_index=True, default=timezone.now)
    
    # Optional metadata (flexible storage for additional context)
    raw_reason = models.IntegerField(
        null=True, 
        blank=True,
        help_text="Android NotificationListenerService removal reason code"
    )
    metadata = models.JSONField(
        null=True, 
        blank=True,
        help_text="Flexible JSON storage for event-specific data"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "timestamp"]),
            models.Index(fields=["notification_event", "timestamp"]),
            models.Index(fields=["user", "interaction_type", "timestamp"]),
        ]
        verbose_name = "Interaction Event"
        verbose_name_plural = "Interaction Events"
        ordering = ["-timestamp"]

    def __str__(self):
        user_label = getattr(self.user, "email", str(self.user))
        return f"{user_label} | {self.notification_event.app.package_name} | {self.interaction_type} | {self.timestamp}"


class DailyAggregate(models.Model):
    """
    Precomputed daily aggregates for fast reporting.
    
    Denormalized for performance - recalculated from InteractionEvent.
    Stores daily metrics per (user, app, day).
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="daily_aggregates"
    )
    app = models.ForeignKey(
        App,
        on_delete=models.CASCADE,
        related_name="daily_aggregates"
    )
    day = models.DateField(db_index=True)
    
    # Aggregate metrics
    posts = models.IntegerField(default=0)
    clicks = models.IntegerField(default=0)
    swipes = models.IntegerField(default=0)
    
    # Computed rates (cached)
    open_rate = models.FloatField(null=True, blank=True)
    
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "app", "day")
        indexes = [
            models.Index(fields=["user", "day"]),
            models.Index(fields=["app", "day"]),
            models.Index(fields=["day"]),
        ]
        verbose_name = "Daily Aggregate"
        verbose_name_plural = "Daily Aggregates"

    def __str__(self):
        user_label = getattr(self.user, "email", str(self.user))
        return f"{user_label} | {self.app.package_name} | {self.day} | {self.posts}p {self.clicks}c {self.swipes}s"
    
    def calculate_open_rate(self):
        """Calculate and cache open rate."""
        if self.posts > 0:
            self.open_rate = self.clicks / self.posts
        else:
            self.open_rate = None
        self.save(update_fields=["open_rate", "last_updated"])