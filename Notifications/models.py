from django.db import models
from django.conf import settings
from django.utils import timezone

class Notifications(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notifications")
    package_name = models.CharField(max_length=200)  # ex: com.whatsapp
    channel_id = models.CharField(max_length=200, null=True, blank=True)
    post_time = models.DateTimeField(default=timezone.now)

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
    
    timestamp_opened = models.DateTimeField(null=True, blank=True)
    timestamp_dismissed = models.DateTimeField(null=True, blank=True)

    @property
    def reaction_time(self):
        if self.timestamp_opened:
            return self.timestamp_opened - self.post_time
        return None

    def __str__(self):
        return f"{self.user.email} | {self.package_name} | {self.title}"
    
    class Meta:
        verbose_name = "Notification"
        verbose_name_plural = "Notifications"

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
        return f"{self.sender}: {self.message_text[:30]}..."
