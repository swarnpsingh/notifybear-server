from rest_framework import serializers


class TrainingFeaturePayloadSerializer(serializers.Serializer):

    feature_version = serializers.IntegerField()

    features = serializers.ListField(
        child=serializers.FloatField(),
        min_length=16,
        max_length=16,
    )

    label = serializers.FloatField(
        required=False,
        allow_null=True
    )

    notification_key = serializers.CharField()

    package_name = serializers.CharField()

class SyncTrainingFeaturesSerializer(serializers.Serializer):

    features = TrainingFeaturePayloadSerializer(
        many=True
    )