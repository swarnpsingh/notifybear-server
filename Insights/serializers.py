from rest_framework import serializers


class DailyInsightsNotificationSerializer(serializers.Serializer):
    notificationId = serializers.IntegerField()
    appLabel = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    title = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    body = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    type = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    priority = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class DailyInsightsRequestSerializer(serializers.Serializer):
    total = serializers.IntegerField(min_value=0)
    high = serializers.IntegerField(min_value=0)
    medium = serializers.IntegerField(min_value=0)
    low = serializers.IntegerField(min_value=0)
    notifications = DailyInsightsNotificationSerializer(many=True)
