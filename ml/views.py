import os

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.http import FileResponse
from ml.models import TrainingFeature
from ml.retrain import INIT_MODEL_PATH, ModelRetrainer
from ml.serializers import SyncTrainingFeaturesSerializer

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def train_model(request):
    user = request.user
    apps = request.data.get("apps", [])

    # If the client didn't specify apps, treat it as a routine background retrain
    # and check if we actually have enough new data to justify the compute cost.
    if not apps:
        should_train, reason = ModelRetrainer.should_retrain_from_features(user)
        if not should_train:
            return Response({
                "status": "skipped",
                "reason": reason
            }, status=200)

        # Train on all apps
        metrics, clf = ModelRetrainer.train_model(user)
    else:
        # If apps ARE provided, it's a targeted request. Skip the check and force train.
        metrics, clf = ModelRetrainer.train_model(user, apps=apps)

    # Handle training failure or lack of data
    if clf is None:
        file = open(INIT_MODEL_PATH, "rb")
        response = FileResponse(
            file,
            content_type="application/octet-stream"
        )
        response["Content-Disposition"] = (
            'attachment; filename="model.onnx"'
        )
        return response

    if not os.path.exists(clf.onnx_path):
        clf.cleanup()
        return Response({
            "status": "failed",
            "error": "onnx export failed"
        }, status=500)

    # Stream the ONNX file back to the device
    file = open(clf.onnx_path, "rb")
    response = FileResponse(
        file,
        content_type="application/octet-stream"
    )
    response["Content-Disposition"] = 'attachment; filename="model.onnx"'
    original_close = response.close
    def cleanup_close():
        try:
            clf.cleanup()
        finally:
            original_close()
    
    response.close = cleanup_close
    
    return response

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def sync_training_features(request):

    serializer = SyncTrainingFeaturesSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    rows = []
    for item in serializer.validated_data["features"]:
        rows.append(
            TrainingFeature(
                user=request.user,
                package_name=item["package_name"],
                notification_key=item["notification_key"],
                feature_version=item["feature_version"],
                features=item["features"],
                label=item.get("label")
            )
        )

    TrainingFeature.objects.bulk_create(rows, ignore_conflicts=True)

    return Response({"saved": len(rows)})