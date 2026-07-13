import logging
from typing import List, Optional

from django.conf import settings
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_FIELD_LEN = 300
TIMEOUT_MS = 25000  # leaves headroom under the client's 60s read timeout


class GeminiGenerationError(Exception):
    pass


class InsightItem(BaseModel):
    notificationId: int
    title: Optional[str] = None
    body: Optional[str] = None


class DailyInsightsSchema(BaseModel):
    summary: str
    insights: List[InsightItem]


_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise GeminiGenerationError("GEMINI_API_KEY is not configured")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _truncate(value, length=MAX_FIELD_LEN):
    return (value or "")[:length]


def _build_prompt(total, high, medium, low, notifications):
    lines = [
        "You are generating a short end-of-day notification digest for a mobile app called NotifyBear.",
        f"Today the user received {total} notifications: {high} high priority, "
        f"{medium} medium priority, {low} low priority.",
        "",
        "Today's notifications (id | app | priority | title | body):",
    ]

    for n in notifications:
        lines.append(
            f"{n['notificationId']} | {_truncate(n.get('appLabel'))} | "
            f"{_truncate(n.get('priority'))} | {_truncate(n.get('title'))} | "
            f"{_truncate(n.get('body'))}"
        )

    lines += [
        "",
        "Write:",
        "1. `summary`: 1-2 sentences summarizing the user's notification day - volume and the dominant "
        "theme/pattern. Be specific, not generic filler.",
        "2. `insights`: notifications that genuinely need a reply, confirmation, or action from the "
        "user - at most 3, but 3 is a ceiling, not a target. Do NOT pad the list to reach 3: include an "
        "item only if it truly requires action. It is normal and expected for this list to contain 0, "
        "1, or 2 items - most days will not have 3 genuinely actionable notifications. Reuse the exact "
        "`notificationId` from the list above - never invent one. Reword `title`/`body` to make the "
        "required action clear and concise.",
        "",
        "Return JSON only, matching the given schema.",
    ]

    return "\n".join(lines)


def generate_insights(total, high, medium, low, notifications):
    if not notifications:
        return DailyInsightsSchema(
            summary=f"You received {total} notifications today, nothing needs your attention.",
            insights=[],
        )

    prompt = _build_prompt(total, high, medium, low, notifications)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DailyInsightsSchema,
                temperature=0.4,
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            ),
        )
    except Exception as e:
        raise GeminiGenerationError(f"Gemini request failed: {e}") from e

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        try:
            parsed = DailyInsightsSchema.model_validate_json(response.text)
        except Exception as e:
            raise GeminiGenerationError(f"Malformed Gemini response: {e}") from e

    if not parsed.summary or not parsed.summary.strip():
        raise GeminiGenerationError("Gemini returned an empty summary")

    valid_ids = {n["notificationId"] for n in notifications}
    parsed.insights = [i for i in parsed.insights if i.notificationId in valid_ids][:3]

    return parsed
