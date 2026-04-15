from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import AppConfig

@api_view(['GET'])
def app_config(request):
    config = AppConfig.objects.first()
    
    config = AppConfig.objects.first()

    if not config:
        return Response({
            "latest_version": "1.0.0",
            "min_supported_version": "1.0.0",
            "force_update_message": "Please update the app."
        })

    return Response({
        "latest_version": config.latest_version,
        "min_supported_version": config.min_supported_version,
        "force_update_message": config.force_update_message
    })