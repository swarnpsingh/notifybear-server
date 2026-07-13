from django.urls import path

from .views import generate_daily_insights

urlpatterns = [
    path("daily/", generate_daily_insights, name="generate_daily_insights"),
]
