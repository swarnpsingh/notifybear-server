from rest_framework.throttling import SimpleRateThrottle

class NotificationIngestThrottle(SimpleRateThrottle):
    scope = "notif_ingest"

    def get_cache_key(self, request, view):
        # Throttle per authenticated user
        if request.user and request.user.is_authenticated:
            return f"{self.scope}:{request.user.id}"

        # Fallback to IP for safety
        return f"{self.scope}:{self.get_ident(request)}"
