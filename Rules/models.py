from django.conf import settings
from django.db import models


class UserNotificationRules(models.Model):
    """The complete, current rule set for one user. Each successful
    generation replaces rules_json/summary wholesale (the LLM merges old and
    new); updated_at doubles as the 24h-cooldown anchor."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notification_rules",
    )
    rules_json = models.TextField(default="[]")
    summary = models.TextField(blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Rules for user {self.user_id}"
