import os
from time import time
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView

from django.contrib.auth import authenticate, get_user_model

from django.core.mail import EmailMessage
import threading
from django.template.loader import render_to_string

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .serializers import UserSerializer, UserSignupSerializer
from .throttles import LoginRateThrottle
from .utils import log_auth_event

User = get_user_model()

def send_html_email(to, subject, html_content):
    email = EmailMessage(
        subject=subject,
        body=html_content,
        from_email='support@notifybear.com',
        to=[to],
    )
    email.content_subtype = 'html'
    email.send()

class SignupView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = UserSignupSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            if user.email:
                html = render_to_string(
                    "welcome.html",
                    {"username": user.username}
                )

                threading.Thread(
                    target=send_html_email,
                    args=(user.email, "Welcome to NotifyBear 🐻", html)
                ).start()
            refresh = RefreshToken.for_user(user)
            # log
            log_auth_event("login_success", request=request, user=user, username=user.username, detail={"via": "signup"})
            return Response({
                "refresh": str(refresh),
                "access": str(refresh.access_token),
                "user": UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)

        # serializer.errors already contains structured error codes (as per earlier)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    throttle_classes = [LoginRateThrottle]  # per-view throttle

    def post(self, request):
        email_or_username = (request.data.get("username") or "").strip()
        password = request.data.get("password") or ""

        if not email_or_username or not password:
            # structured error
            return Response({
                "detail": [{"code": "missing_fields", "message": "username and password are required"}]
            }, status=status.HTTP_400_BAD_REQUEST)

        # decide if input is email
        username = email_or_username
        if "@" in email_or_username:
            # normalize
            email_norm = email_or_username.lower()
            try:
                user_obj = User.objects.get(email=email_norm)
                username = user_obj.username
            except User.DoesNotExist:
                username = None  # let authenticate fail below

        # normalize username for case-insensitivity
        if username:
            username = username.strip()

        user = authenticate(username=username, password=password)

        # Handle failed login
        if user is None:
            # log failed attempt
            log_auth_event("login_failed", request=request, username=email_or_username, detail={"reason": "invalid_credentials"})
            return Response({
                "detail": [{"code": "invalid_credentials", "message": "Invalid username/email or password."}]
            }, status=status.HTTP_400_BAD_REQUEST)

        # block inactive accounts explicitly
        if not user.is_active:
            log_auth_event("login_failed", request=request, user=user, username=user.username, detail={"reason": "inactive"})
            # do not reveal account disabled detail to client, but return generic error
            return Response({
                "detail": [{"code": "invalid_credentials", "message": "Invalid username/email or password."}]
            }, status=status.HTTP_400_BAD_REQUEST)

        # success
        refresh = RefreshToken.for_user(user)
        log_auth_event("login_success", request=request, user=user, username=user.username, detail={})
        return Response({
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            "user": UserSerializer(user).data
        }, status=status.HTTP_200_OK)


class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response({"detail": "Refresh token required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
            # log logout event
            log_auth_event("logout", request=request, user=request.user, username=request.user.username, detail={})
        except Exception as e:
            return Response({"detail": "Invalid or expired token."}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"detail": "Logout successful."}, status=status.HTTP_200_OK)


# Subclass token refresh to add logging and return refresh token when rotation is enabled
class CustomTokenRefreshView(TokenRefreshView):
    """
    TokenRefreshView (from simplejwt) handles rotation if SIMPLE_JWT settings have ROTATE_REFRESH_TOKENS=True.
    Here we override to log the refresh usage.
    """
    def post(self, request, *args, **kwargs):
        resp = super().post(request, *args, **kwargs)
        # If successful, resp.data will contain 'access' and possibly 'refresh' (if rotation is True)
        try:
            # Attempt to find username from provided refresh token's payload (best-effort)
            incoming_refresh = request.data.get("refresh")
            username = None
            if incoming_refresh:
                # We cannot decode refresh easily without errors - skip retrieving user here.
                pass
        finally:
            # Log token_refresh event (anonymous if user not resolved)
            log_auth_event("token_refresh", request=request, detail={"response_keys": list(resp.data.keys()) if resp.data else []})
        return resp


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
@throttle_classes([])
def me(request):
    user = request.user
    serializer = UserSerializer(user, context={"request": request})
    return Response(serializer.data)

from google.oauth2 import id_token
from google.auth.transport import requests
class GoogleLoginView(APIView):
    permission_classes = []

    def post(self, request):
        token = request.data.get("id_token")

        if not token:
            return Response({"error": "Missing id_token"}, status=400)

        try:
            # Validate token with Google
            payload = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                settings.GOOGLE_WEB_ID
            )
        except Exception:
            return Response({"error": "Invalid Google token"}, status=400)

        # Extract user info
        email = payload["email"]
        first = payload.get("given_name", "")
        last = payload.get("family_name", "")

        # Create or get user
        user, created = User.objects.get_or_create(
            email=email,
            defaults={
                "username": email.split("@")[0],
                "first_name": first,
                "last_name": last
            }
        )
        
        if created:
            html = render_to_string(
                "welcome.html",
                {"username": first}
            )
            threading.Thread(
                target=send_html_email,
                args=(user.email, "Welcome to NotifyBear 🐻", html)
            ).start()

        # Issue our JWT tokens
        refresh = RefreshToken.for_user(user)

        return Response({
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            "user": UserSerializer(user).data
        })

class UploadProfilePhotoView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        profile = request.user.profile

        if "dp" not in request.FILES:
            return Response({"error":"No file"}, status=400)
        
        file = request.FILES["dp"]

        ext = os.path.splitext(file.name)[1]
        if not ext:
            ext = ".jpg"

        file.name = f"{request.user.username}_{int(time())}{ext}"
        
        if profile.dp:
            profile.dp.delete(save=False)
        
        profile.dp = file
        profile.save()

        serializer = UserSerializer(request.user, context={"request":request})
        return Response(serializer.data)

class UpdateProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):

        user = request.user
        profile = user.profile

        serializer = UserSerializer(
            user,
            data=request.data,
            partial=True,
            context={"request": request}
        )

        if serializer.is_valid():
            serializer.save()

            address = request.data.get("address")
            if address is not None:
                profile.address = address
                profile.save()

            return Response(serializer.data)

        return Response(serializer.errors, status=400)

from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import logging
logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class ForgotPasswordView(APIView):
    permission_classes = []

    def post(self, request):
        email = request.data.get("email")

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # Don't reveal user existence (security)
            return Response({"message": "If account exists, email sent"}, status=200)

        token = PasswordResetTokenGenerator().make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))

        reset_link = request.build_absolute_uri(f"/accounts/reset-password/{uid}/{token}/")

        html = render_to_string("reset_password_email.html", {
            "reset_link": reset_link,
            "username": user.username
        })

        threading.Thread(
            target=send_html_email,
            args=(email, "Reset your password", html)
        ).start()

        return Response({"message": "If account exists, email sent"}, status=200)

@method_decorator(csrf_exempt, name='dispatch')
class ResetPasswordView(APIView):
    permission_classes = []

    def post(self, request):
        
        uid = request.data.get("uid")
        token = request.data.get("token")
        new_password = request.data.get("password")

        try:
            user_id = urlsafe_base64_decode(uid).decode()
            user = User.objects.get(pk=user_id)
        except Exception as e:
            logger.error("UID ERROR:", str(e))
            return Response({"error": "Invalid link"}, status=400)

        if not PasswordResetTokenGenerator().check_token(user, token):
            return Response({"error": "Invalid or expired token"}, status=400)
        
        try:
            validate_password(new_password)
        except ValidationError as e:
            return Response({"error": list(e.messages)}, status=400)
        
        user.set_password(new_password)
        user.save()

        return Response({"message": "Password reset successful"}, status=200)

from django.shortcuts import render

def forgot_password_page(request):
    return render(request, "forgot_password.html")

def reset_password_page(request, uid, token):
    return render(request, "reset_password.html", {
        "uid": uid,
        "token": token
    })