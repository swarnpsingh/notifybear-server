import logging

from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .serializers import DailyInsightsRequestSerializer
from .throttles import DailyInsightsThrottle
from .gemini_client import generate_insights, GeminiGenerationError

logger = logging.getLogger(__name__)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([DailyInsightsThrottle])
def generate_daily_insights(request):
    s = DailyInsightsRequestSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    v = s.validated_data

    try:
        result = generate_insights(
            total=v["total"],
            high=v["high"],
            medium=v["medium"],
            low=v["low"],
            notifications=v["notifications"],
        )
    except GeminiGenerationError as e:
        logger.warning("Daily insights generation failed for user %s: %s", request.user.id, e)
        return Response({"error": "generation_failed"}, status=status.HTTP_502_BAD_GATEWAY)

    return Response(
        {
            "summary": result.summary,
            "insights": [
                {"notificationId": i.notificationId, "title": i.title, "body": i.body}
                for i in result.insights
            ],
        },
        status=status.HTTP_200_OK,
    )
