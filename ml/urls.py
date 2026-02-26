from django.urls import path
from ml.views import train_model

urlpatterns = [
    # New consolidated endpoint
    path("train/", train_model, name="train"),

    # Backward compatibility for existing Android clients
    path("train_model/", train_model, name="train_model_legacy"),
    path("retrain_model/", train_model, name="retrain_model_legacy"),
]