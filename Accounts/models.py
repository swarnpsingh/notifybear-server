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