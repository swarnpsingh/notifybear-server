from django.urls import path
from ml.views import train_model

urlpatterns = [
    # A single, smart endpoint that handles both custom and routine training
    path("train/", train_model, name="train_model"),
]