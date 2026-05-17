from django.db import models
from Accounts.models import User


class TrainingFeature(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)

    feature_version = models.IntegerField(default=1)

    features = models.JSONField()

    label = models.FloatField(null=True, blank=True)

    notification_key = models.CharField(max_length=512)

    package_name = models.CharField(max_length=256)

    created_at = models.DateTimeField(auto_now_add=True)

    synced_at = models.DateTimeField(auto_now_add=True)

    used_for_training = models.BooleanField(default=False)
    
    class Meta:
        unique_together = (
            "user",
            "notification_key"
        )