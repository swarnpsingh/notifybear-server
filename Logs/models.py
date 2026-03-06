from django.db import models
from Accounts.models import User

class MobileLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    tag = models.CharField(max_length=100)
    message = models.TextField()
    
    def __str__(self):
        return f"{self.tag} | {self.timestamp}"