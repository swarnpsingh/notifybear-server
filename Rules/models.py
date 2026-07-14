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


class RulesGenerationLog(models.Model):
    """One row per successful rules generation - who and when. A user's
    lifetime total is just their row count (surfaced as a column in admin)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="rules_generation_logs",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Generation by user {self.user_id} at {self.created_at}"


class PromptInjectionAttempt(models.Model):
    """A rules prompt the LLM flagged as attempting to manipulate its
    instructions. The generation itself still goes through the normal
    sanitizer; this exists for monitoring (admin + email alert)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prompt_injection_attempts",
    )
    prompt = models.TextField()
    note = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Injection attempt by user {self.user_id} at {self.created_at}"
