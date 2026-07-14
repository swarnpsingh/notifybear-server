import json
import logging
import threading
from datetime import timedelta

from django.core.mail import EmailMessage
from django.utils import timezone
from django.utils.html import escape
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import UserNotificationRules, RulesGenerationLog, PromptInjectionAttempt
from .serializers import GenerateRulesRequestSerializer
from .throttles import RulesGenerateThrottle
from .gemini_client import generate_rules, RulesGenerationError

logger = logging.getLogger(__name__)

COOLDOWN = timedelta(hours=24)

INJECTION_ALERT_TO = "team@notifybear.com"


def _send_injection_alert(user, prompt, note, attempted_at):
    """Fire-and-forget email to the team; a mail failure must never affect
    the API response, so this runs on a daemon thread and swallows errors."""

    def _send():
        try:
            # The prompt is hostile by definition here - escape everything
            # user-controlled before it goes into an HTML body.
            body = (
                "<h3>Prompt injection attempt flagged in Custom Rules</h3>"
                f"<p><b>User:</b> {escape(user.username)} "
                f"(id {user.id}, {escape(user.email) if user.email else 'no email'})<br>"
                f"<b>When:</b> {attempted_at.isoformat()}<br>"
                f"<b>Model note:</b> {escape(note) if note else '-'}</p>"
                f"<p><b>Prompt:</b></p><pre>{escape(prompt)}</pre>"
            )
            email = EmailMessage(
                subject="[NotifyBear] Prompt injection attempt flagged",
                body=body,
                from_email="support@notifybear.com",
                to=[INJECTION_ALERT_TO],
            )
            email.content_subtype = "html"
            email.send()
        except Exception as e:
            logger.warning("Injection alert email failed: %s", e)

    threading.Thread(target=_send, daemon=True).start()


def _next_allowed_at(obj):
    return (obj.updated_at + COOLDOWN).isoformat()


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([RulesGenerateThrottle])
def generate_user_rules(request):
    s = GenerateRulesRequestSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    prompt = s.validated_data["prompt"]
    installed_apps = s.validated_data.get("apps") or []

    existing = UserNotificationRules.objects.filter(user=request.user).first()

    if existing is not None:
        elapsed = timezone.now() - existing.updated_at
        if elapsed < COOLDOWN:
            retry_after = int((COOLDOWN - elapsed).total_seconds())
            return Response(
                {
                    "error": "cooldown_active",
                    "retry_after_seconds": retry_after,
                    "next_allowed_at": _next_allowed_at(existing),
                },
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

    try:
        existing_rules = json.loads(existing.rules_json) if existing else []
    except ValueError:
        existing_rules = []

    try:
        rules, summary, injection_attempt, injection_note = generate_rules(
            existing_rules, prompt, installed_apps
        )
    except RulesGenerationError as e:
        logger.warning("Rules generation failed for user %s: %s", request.user.id, e)
        return Response({"error": "generation_failed"}, status=status.HTTP_502_BAD_GATEWAY)

    obj = existing or UserNotificationRules(user=request.user)
    obj.rules_json = json.dumps(rules)
    obj.summary = summary
    obj.save()  # auto_now stamps updated_at: the daily change is consumed only here

    RulesGenerationLog.objects.create(user=request.user)

    if injection_attempt:
        attempt = PromptInjectionAttempt.objects.create(
            user=request.user,
            prompt=prompt,
            note=injection_note,
        )
        logger.warning(
            "Prompt injection flagged for user %s: %s", request.user.id, injection_note
        )
        _send_injection_alert(request.user, prompt, injection_note, attempt.created_at)

    return Response(
        {
            "summary": summary,
            "rules": rules,
            "next_allowed_at": _next_allowed_at(obj),
        },
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_user_rules(request):
    existing = UserNotificationRules.objects.filter(user=request.user).first()

    if existing is None:
        return Response(
            {"summary": "", "rules": [], "next_allowed_at": None},
            status=status.HTTP_200_OK,
        )

    try:
        rules = json.loads(existing.rules_json)
    except ValueError:
        rules = []

    return Response(
        {
            "summary": existing.summary,
            "rules": rules,
            "next_allowed_at": _next_allowed_at(existing),
        },
        status=status.HTTP_200_OK,
    )
