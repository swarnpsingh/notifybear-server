from django.urls import path
from .views import SignupView, LoginView, LogoutView, CustomTokenRefreshView, me, GoogleLoginView

urlpatterns = [
    path("signup/", SignupView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("token/refresh/", CustomTokenRefreshView.as_view(), name="token_refresh"),
    path("me/", me, name="me"),
    path("google-login/", GoogleLoginView.as_view(), name="google_login"),
]
