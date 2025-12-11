from rest_framework import serializers
from .models import (
    Notifications,
    NotificationMessage,
    App,
    InteractionEvent,
    DailyAggregate,
)


class NotificationMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationMessage
        fields = ["id", "sender", "message_text", "message_time"]


class NotificationsSerializer(serializers.ModelSerializer):
    messages = NotificationMessageSerializer(many=True, read_only=True)
    reaction_time = serializers.DurationField(read_only=True)

    class Meta:
        model = Notifications
        fields = [
            "id",
            "user",
            "package_name",
            "channel_id",
            "notif_key",
            "post_time",
            "timestamp_opened",
            "timestamp_dismissed",
            "reaction_time",
            "title",
            "text",
            "big_text",
            "sub_text",
            "summary_text",
            "info_text",
            "text_lines",
            "large_icon_base64",
            "picture_base64",
            "conversation_title",
            "people",
            "created_at",
            "messages",
        ]
        read_only_fields = ("user", "created_at", "reaction_time")


#
# New serializers for ingestion + reporting
#

class IngestNotificationSerializer(serializers.Serializer):
    package_name = serializers.CharField()
    app_label = serializers.CharField(required=False, allow_blank=True)
    notif_key = serializers.CharField(required=False, allow_blank=True)
    posted_at = serializers.DateTimeField(required=False)
    title = serializers.CharField(required=False, allow_blank=True)
    text = serializers.CharField(required=False, allow_blank=True)


class IngestInteractionSerializer(serializers.Serializer):
    package_name = serializers.CharField()
    app_label = serializers.CharField(required=False, allow_blank=True)
    notif_key = serializers.CharField(required=False, allow_blank=True)
    removed_at = serializers.DateTimeField()
    raw_reason = serializers.IntegerField(required=False, allow_null=True)
    interaction_type = serializers.ChoiceField(choices=[t[0] for t in InteractionEvent.TYPES])


class AppSerializer(serializers.ModelSerializer):
    class Meta:
        model = App
        fields = ("package_name", "app_label", "first_seen")


class DailyAggregateSerializer(serializers.ModelSerializer):
    app_label = serializers.CharField(source="app.app_label", read_only=True)
    package_name = serializers.CharField(source="app.package_name", read_only=True)

    class Meta:
        model = DailyAggregate
        fields = ("day", "package_name", "app_label", "posts", "clicks", "swipes")
