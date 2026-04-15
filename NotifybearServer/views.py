from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import AppConfig

@api_view(['GET'])
def app_config(request):
    config = AppConfig.objects.first()
    
    return Response({
        "latest_version": config.latest_version,
        "min_supported_version": config.min_supported_version,
        "force_update_message": config.force_update_message
    })