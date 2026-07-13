from django.urls import path

from .views import generate_user_rules, get_user_rules

urlpatterns = [
    path("", get_user_rules, name="get_user_rules"),
    path("generate/", generate_user_rules, name="generate_user_rules"),
]
