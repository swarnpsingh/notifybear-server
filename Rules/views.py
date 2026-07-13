import json
import logging
from datetime import timedelta

from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import UserNotificationRules
from .serializers import GenerateRulesRequestSerializer
from .throttles import RulesGenerateThrottle
from .gemini_client import generate_rules, RulesGenerationError

logger = logging.getLogger(__name__)

COOLDOWN = timedelta(hours=24)


def _next_allowed_at(obj):
    return (obj.updated_at + COOLDOWN).isoformat()


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([RulesGenerateThrottle])
def generate_user_rules(request):
    s = GenerateRulesRequestSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    prompt = s.validated_data["prompt"]

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
        rules, summary = generate_rules(existing_rules, prompt)
    except RulesGenerationError as e:
        logger.warning("Rules generation failed for user %s: %s", request.user.id, e)
        return Response({"error": "generation_failed"}, status=status.HTTP_502_BAD_GATEWAY)

    obj = existing or UserNotificationRules(user=request.user)
    obj.rules_json = json.dumps(rules)
    obj.summary = summary
    obj.save()  # auto_now stamps updated_at: the daily change is consumed only here

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
