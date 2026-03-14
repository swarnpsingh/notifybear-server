from django.urls import path
from .views import SignupView, LoginView, LogoutView, CustomTokenRefreshView
from .views import me, GoogleLoginView, UploadProfilePhotoView, UpdateProfileView

urlpatterns = [
    path("signup/", SignupView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("token/refresh/", CustomTokenRefreshView.as_view(), name="token_refresh"),
    path("me/", me, name="me"),
    path("google-login/", GoogleLoginView.as_view(), name="google_login"),
    path("profile/upload-photo/", UploadProfilePhotoView.as_view(), name="upload_profile_photo"),
    path("update-profile/", UpdateProfileView.as_view(), name="update_profile"),
]
