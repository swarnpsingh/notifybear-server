from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.http import FileResponse
from ml.retrain import ModelRetrainer

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def train_model(request):
    user = request.user
    apps = request.data.get("apps", [])

    # If the client didn't specify apps, treat it as a routine background retrain
    # and check if we actually have enough new data to justify the compute cost.
    if not apps:
        should_train, reason = ModelRetrainer.should_retrain(user)
        if not should_train:
            return Response({
                "status": "skipped",
                "reason": reason
            }, status=200)

        # Train on all apps
        metrics, file_path = ModelRetrainer.train_model(user)
    else:
        # If apps ARE provided, it's a targeted request. Skip the check and force train.
        metrics, file_path = ModelRetrainer.train_model(user, apps=apps)

    # Handle training failure or lack of data
    if not file_path:
        return Response({
            "status": "failed",
            "error": "Not enough data or training failed"
        }, status=400)

    # Stream the ONNX file back to the device
    file = open(file_path, "rb")
    response = FileResponse(file, content_type="application/octet-stream")
    response["Content-Disposition"] = 'attachment; filename="model.onnx"'
    
    # Assuming you have a custom middleware to delete the file after streaming
    response["X-Delete-File"] = file_path 
    
    return response