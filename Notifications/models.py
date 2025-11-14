from django.db import models
from django.conf import settings
from django.utils import timezone

class App(models.Model):
    """
    Represents an app (package) that produced notifications for a user.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="apps")
    package_name = models.CharField(max_length=255)
    app_label = models.CharField(max_length=255, blank=True, default="")
    first_seen = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "package_name")
        indexes = [
            models.Index(fields=["user", "package_name"]),
        ]

    def __str__(self):
        return f"{self.app_label or self.package_name}"


class Notifications(models.Model):
    """
    Raw notification record (one row per posted notification).
    Extended from the existing model to include notif_key and indexes.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notifications")
    package_name = models.CharField(max_length=200)  # ex: com.whatsapp
    channel_id = models.CharField(max_length=200, null=True, blank=True)
    post_time = models.DateTimeField(default=timezone.now)

    # Add a client-provided key to dedupe/update notifications
    notif_key = models.CharField(max_length=256, null=True, blank=True, db_index=True)

    # Primary text content
    title = models.CharField(max_length=500, null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    big_text = models.TextField(null=True, blank=True)
    sub_text = models.CharField(max_length=500, null=True, blank=True)
    summary_text = models.CharField(max_length=500, null=True, blank=True)
    info_text = models.CharField(max_length=500, null=True, blank=True)

    # Multi-line text notifications (e.g., Gmail summary style)
    text_lines = models.TextField(null=True, blank=True)

    # Optional image information — store encoded or URLs
    large_icon_base64 = models.TextField(null=True, blank=True)
    picture_base64 = models.TextField(null=True, blank=True)

    # Conversation / group chat info
    conversation_title = models.CharField(max_length=500, null=True, blank=True)
    people = models.TextField(null=True, blank=True)  # array of contact names / IDs

    created_at = models.DateTimeField(auto_now_add=True)
    
    # User interaction timestamps (client can PATCH these or send via interaction endpoint)
    timestamp_opened = models.DateTimeField(null=True, blank=True)
    timestamp_dismissed = models.DateTimeField(null=True, blank=True)

    @property
    def reaction_time(self):
        if self.timestamp_opened:
            return self.timestamp_opened - self.post_time
        return None

    def __str__(self):
        # fallback if user has no email
        user_label = getattr(self.user, "email", str(self.user))
        return f"{user_label} | {self.package_name} | {self.title}"

    class Meta:
        verbose_name = "Notification"
        verbose_name_plural = "Notifications"
        indexes = [
            models.Index(fields=["user", "package_name"]),
            models.Index(fields=["user", "post_time"]),
            models.Index(fields=["package_name", "post_time"]),
        ]


class NotificationMessage(models.Model):
    """
    Individual chat messages inside messaging-style notifications.
    """
    notification = models.ForeignKey(
        Notifications,
        on_delete=models.CASCADE,
        related_name="messages"
    )

    sender = models.CharField(max_length=200, null=True, blank=True)
    message_text = models.TextField(null=True, blank=True)
    message_time = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        preview = (self.message_text or "")[:30]
        return f"{self.sender}: {preview}"


class InteractionEvent(models.Model):
    """
    Append-only interaction events captured from device when a notification is removed/acted on.
    """
    CLICK = "CLICK"
    SWIPE = "SWIPE"
    APP_CANCEL = "APP_CANCEL"
    OTHER = "OTHER"
    TYPES = [(CLICK, "Click"), (SWIPE, "Swipe"), (APP_CANCEL, "App cancel"), (OTHER, "Other")]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="interaction_events")
    # link to app if known; otherwise create on ingest
    app = models.ForeignKey(App, on_delete=models.CASCADE, related_name="interaction_events")
    notif_key = models.CharField(max_length=256, null=True, blank=True, db_index=True)
    removed_at = models.DateTimeField(db_index=True)
    raw_reason = models.IntegerField(null=True, blank=True)
    interaction_type = models.CharField(max_length=16, choices=TYPES)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "removed_at"]),
            models.Index(fields=["app", "removed_at"]),
            models.Index(fields=["user", "notif_key"]),
        ]

    def __str__(self):
        return f"{getattr(self.user,'email',str(self.user))} | {self.app.package_name} | {self.interaction_type}"


class DailyAggregate(models.Model):
    """
    Precomputed daily aggregates for fast reporting: posts / clicks / swipes per user+app+day.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="daily_aggregates")
    app = models.ForeignKey(App, on_delete=models.CASCADE, related_name="daily_aggregates")
    day = models.DateField(db_index=True)  # YYYY-MM-DD
    posts = models.IntegerField(default=0)
    clicks = models.IntegerField(default=0)
    swipes = models.IntegerField(default=0)

    class Meta:
        unique_together = ("user", "app", "day")
        indexes = [
            models.Index(fields=["user", "day"]),
            models.Index(fields=["app", "day"]),
        ]

    def __str__(self):
        return f"{getattr(self.user,'email',str(self.user))} | {self.app.package_name} | {self.day}"
