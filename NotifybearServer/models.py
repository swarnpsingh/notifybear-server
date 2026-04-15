from django.db import models

class AppConfig(models.Model):
    latest_version = models.CharField(max_length=20)
    min_supported_version = models.CharField(max_length=20)
    force_update_message = models.TextField(default="Please update the app to continue.")