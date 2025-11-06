from django.urls import path
from .views import signup, login, me

urlpatterns = [
    path("signup/", signup, name="signup"),
    path("login/", login, name="login"),
    path("me/", me, name="me"),
]
