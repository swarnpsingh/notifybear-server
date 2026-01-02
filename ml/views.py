import os
from django.http import FileResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ml.retrain import retrain_model_for_user
from ml.train_model import train_for_user

@api_view(["POST"])
def train_model_for_user(request):
    user = request.user
    apps = request.data.get("apps", [])

    if not apps:
        return Response({"error": "No apps sent"}, status=400)

    path = train_for_user(user.id, apps)
    if not os.path.exists(path):
        return Response({"error": "Model training failed"}, status=500)

    response = FileResponse(open(path, "rb"), as_attachment=True, filename="model.joblib")
    response["X-Delete-File"] = path
    return response


@api_view(["POST"])
def retrain_model(request):
    user = request.user
    apps = request.data.get("apps", [])

    if not apps:
        return Response({"error": "apps required"}, status=400)

    path = retrain_model_for_user(user, apps)
    if not os.path.exists(path):
        return Response({"error": "Model training failed"}, status=500)

    response = FileResponse(open(path, "rb"), as_attachment=True, filename="model.joblib")
    response["X-Delete-File"] = path
    return response
