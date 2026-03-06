from django.urls import path
from . import views

urlpatterns = [
    path("ingest/", views.mobile_log),
]