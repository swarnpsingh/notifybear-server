from django.urls import path
from ml.views import train_model, sync_training_features

urlpatterns = [
    # New consolidated endpoint
    path("train/", train_model, name="train"),

    # Backward compatibility for existing Android clients
    path("train_model/", train_model, name="train_model_legacy"),
    path("retrain_model/", train_model, name="retrain_model_legacy"),
    path("sync_features/",sync_training_features, name="sync_training_features"),
]