from rest_framework.throttling import SimpleRateThrottle


class RulesGenerateThrottle(SimpleRateThrottle):
    """Abuse guard only - the real once-per-24h limit is enforced in the view
    against UserNotificationRules.updated_at, so failed generations don't
    consume the user's daily change."""

    scope = "rules_generate"

    def get_cache_key(self, request, view):
        if request.user and request.user.is_authenticated:
            return f"{self.scope}:{request.user.id}"

        return f"{self.scope}:{self.get_ident(request)}"
