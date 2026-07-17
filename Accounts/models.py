from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

class User(AbstractUser):
    email = models.EmailField(unique=True)
    
    def __str__(self):
        return f"{self.username}"
    
    def save(self, *args, **kwargs):
        if self.email:
            self.email = self.email.strip().lower()
        super().save(*args, **kwargs)
    
    @property
    def initials(self):
        a = ""
        b = ""
        if self.first_name:
            a = self.first_name[0]
        if self.last_name:
            b = self.last_name[0]
        c = a+b
        if c=="":
            c = "NA"
        return str(c)
    

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    dp = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    last_model_retrain = models.DateTimeField(null=True, blank=True)
    address = models.TextField(blank=True, null=True)
    # Whether this user has seen (or skipped) the first-login coach-mark
    # tour in the app. Per-user, not per-device, so a user who did it on
    # one phone never sees it again on another.
    tutorial_completed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
class AuthAuditLog(models.Model):
    EVENT_CHOICES = [
        ("login_success", "Login success"),
        ("login_failed", "Login failed"),
        ("logout", "Logout"),
        ("token_refresh", "Token refresh"),
    ]

    event = models.CharField(max_length=32, choices=EVENT_CHOICES)
    username = models.CharField(max_length=150, blank=True, null=True)
    user = models.ForeignKey("User", on_delete=models.SET_NULL, null=True, blank=True)
    ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    detail = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.event} - {self.username or self.user} @ {self.ip} [{self.timestamp}]"


class DeletedAccount(models.Model):
    email = models.EmailField()
    reasons = models.JSONField(default=list)
    other_reason = models.TextField(blank=True, null=True)
    deleted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.email} deleted on {self.deleted_at.strftime('%Y-%m-%d %H:%M:%S')}"

    class Meta:
        ordering = ['-deleted_at']

class UserKey(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    wrapped_key = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class UserStreak(models.Model):
    """
    Daily streak state mirrored from the app. last_streak_date is the
    device's LOCAL calendar day (yyyy-MM-dd) as reported by the client -
    it is stored as an opaque string on purpose, never reinterpreted in
    server time, so a user near midnight doesn't lose a day to timezone
    math. Break detection (streak reads 0 after a missed day) stays
    client-side at read time.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="streak")
    streak_count = models.IntegerField(default=0)
    last_streak_date = models.CharField(max_length=10, blank=True, default="")
    # All-time best streak_count - monotonic, survives a break (unlike
    # streak_count itself, which resets). Merged as max() unconditionally,
    # independent of which side's last_streak_date is newer.
    longest_streak = models.IntegerField(default=0)
    # Freeze-token bank (see StreakManager.maybeGrantFreezes on the client) -
    # NOT monotonic, so it merges atomically with whichever side wins the
    # streak_count/last_streak_date merge rather than via max().
    freeze_count = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}: {self.streak_count} (last {self.last_streak_date or '-'})"