from django.urls import path

from ml.views import train_model_for_user, retrain_model

urlpatterns = [
    path("train_model/", train_model_for_user, name="train_model_for_user"),
    path("retrain_model/", retrain_model, name="retrain_model"),
]