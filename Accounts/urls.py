from django.urls import path
from .views import SignupView, LoginView, LogoutView, CustomTokenRefreshView
from .views import me, GoogleLoginView, UploadProfilePhotoView, UpdateProfileView
from .views import ForgotPasswordView, ResetPasswordView, forgot_password_page, reset_password_page
from .views import DeleteAccountView
from .views import get_user_key, set_user_key

urlpatterns = [
    path("signup/", SignupView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("token/refresh/", CustomTokenRefreshView.as_view(), name="token_refresh"),
    path("me/", me, name="me"),
    path("google-login/", GoogleLoginView.as_view(), name="google_login"),
    path("profile/upload-photo/", UploadProfilePhotoView.as_view(), name="upload_profile_photo"),
    path("update-profile/", UpdateProfileView.as_view(), name="update_profile"),
    path("delete-account", DeleteAccountView.as_view(), name="delete_account"),
    
    path('auth/forgot-password/', ForgotPasswordView.as_view(), name='forgot-password'),
    path('auth/reset-password/', ResetPasswordView.as_view(), name='reset-password'),
    
    path('forgot-password/', forgot_password_page, name='forgot-password-page'),
    path('reset-password/<uid>/<token>/', reset_password_page, name='reset-password-page'),
    
    path('get-encryptor-key/', get_user_key, name='get_encryptor_key'),
    path('set-encryptor-key/', set_user_key, name='set_encryptor_key'),
]
