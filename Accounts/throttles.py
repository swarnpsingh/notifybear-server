from rest_framework.throttling import SimpleRateThrottle

class LoginRateThrottle(SimpleRateThrottle):
    scope = "login"

    def get_cache_key(self, request, view):
        # try username/email then fallback to IP
        username = None
        try:
            username = request.data.get("username") if request.data else None
        except Exception:
            username = None

        ip = self.get_ident(request)

        if username:
            username = str(username).strip().lower()
            return f"{self.scope}:{username}:{ip}"
        return f"{self.scope}:{ip}"