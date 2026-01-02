import os

class DeleteTempFileMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        path = response.headers.get("X-Delete-File")
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

        return response
