from rest_framework.throttling import SimpleRateThrottle


class DailyInsightsThrottle(SimpleRateThrottle):
    scope = "daily_insights"

    def get_cache_key(self, request, view):
        if request.user and request.user.is_authenticated:
            return f"{self.scope}:{request.user.id}"

        return f"{self.scope}:{self.get_ident(request)}"
