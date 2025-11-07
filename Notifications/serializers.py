from rest_framework import serializers
from .models import Notifications, NotificationMessage


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
