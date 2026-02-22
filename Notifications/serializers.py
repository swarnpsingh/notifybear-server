from rest_framework import serializers
from .models import (
    NotificationEvent,
    UserNotificationState,
    NotificationMessage,
    App,
    InteractionEvent,
    DailyAggregate,
)


# -------------------------
# NotificationMessage Serializer
# -------------------------

class NotificationMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationMessage
        fields = ["id", "sender", "message_text", "message_time"]


# -------------------------
# NotificationEvent Serializer (immutable event data)
# -------------------------

class NotificationEventSerializer(serializers.ModelSerializer):
    """
    Serializes the immutable notification event (what happened).
    Includes app information for context.
    """
    messages = NotificationMessageSerializer(many=True, read_only=True)
    package_name = serializers.CharField(source='app.package_name', read_only=True)
    app_label = serializers.CharField(source='app.app_label', read_only=True)
    app_id = serializers.IntegerField(source='app.id', read_only=True)
    content_hash = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    type = serializers.CharField(read_only=True)
    
    class Meta:
        model = NotificationEvent
        fields = [
            "id",
            "app_id",
            "package_name",
            "app_label",
            "notif_key",
            "channel_id",
            "post_time",
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
            "content_hash",
            "created_at",
            "messages",
            "type",
        ]
        read_only_fields = ("id", "created_at", "app_id", "package_name", "app_label")


# -------------------------
# UserNotificationState Serializer (mutable user interaction)
# -------------------------

class UserNotificationStateSerializer(serializers.ModelSerializer):
    """
    Serializes user-specific notification state (what the user did).
    Includes the full notification event data nested.
    """
    notification = NotificationEventSerializer(source='notification_event', read_only=True)
    reaction_time = serializers.SerializerMethodField()
    
    # For backward compatibility, expose these at top level
    package_name = serializers.CharField(source='notification_event.app.package_name', read_only=True)
    app_label = serializers.CharField(source='notification_event.app.app_label', read_only=True)
    title = serializers.CharField(source='notification_event.title', read_only=True)
    text = serializers.CharField(source='notification_event.text', read_only=True)
    post_time = serializers.DateTimeField(source='notification_event.post_time', read_only=True)
    
    class Meta:
        model = UserNotificationState
        fields = [
            "id",
            "user",
            "notification",
            "is_read",
            "opened_at",
            "dismissed_at",
            "snoozed_until",
            "ml_score",
            "reaction_time",
            "last_updated",
            "created_at",
            # Backward compatibility fields
            "package_name",
            "app_label",
            "title",
            "text",
            "post_time",
        ]
        read_only_fields = ("user", "created_at", "last_updated", "reaction_time")
    
    def get_reaction_time(self, obj):
        """
        Returns reaction time in seconds (null if not opened).
        """
        rt = obj.reaction_time
        if rt:
            return rt.total_seconds()
        return None

class UserNotificationStateUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserNotificationState
        fields = ["id", "ml_score"]

# -------------------------
# Compact UserNotificationState Serializer (for lists)
# -------------------------

class UserNotificationStateCompactSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for listing notifications.
    Doesn't nest the full notification object.
    """
    package_name = serializers.CharField(source='notification_event.app.package_name', read_only=True)
    app_label = serializers.CharField(source='notification_event.app.app_label', read_only=True)
    title = serializers.CharField(source='notification_event.title', read_only=True)
    text = serializers.CharField(source='notification_event.text', read_only=True)
    post_time = serializers.DateTimeField(source='notification_event.post_time', read_only=True)
    notif_key = serializers.CharField(source='notification_event.notif_key', read_only=True)
    reaction_time_seconds = serializers.SerializerMethodField()
    
    class Meta:
        model = UserNotificationState
        fields = [
            "id",
            "notification_event_id",
            "package_name",
            "app_label",
            "notif_key",
            "title",
            "text",
            "post_time",
            "is_read",
            "opened_at",
            "dismissed_at",
            "ml_score",
            "reaction_time_seconds",
        ]
    
    def get_reaction_time_seconds(self, obj):
        rt = obj.reaction_time
        return rt.total_seconds() if rt else None


# -------------------------
# Ingestion Serializers
# -------------------------

class IngestNotificationSerializer(serializers.Serializer):
    """
    Validates incoming notification data from Android client.
    """
    package_name = serializers.CharField(max_length=255)
    app_label = serializers.CharField(max_length=255, required=False, allow_blank=True)
    notif_key = serializers.CharField(max_length=256, required=False, allow_blank=True, allow_null=True)
    posted_at = serializers.DateTimeField(required=False)
    
    # Content fields
    title = serializers.CharField(max_length=500, required=False, allow_blank=True, allow_null=True)
    text = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    big_text = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    sub_text = serializers.CharField(max_length=500, required=False, allow_blank=True, allow_null=True)
    summary_text = serializers.CharField(max_length=500, required=False, allow_blank=True, allow_null=True)
    info_text = serializers.CharField(max_length=500, required=False, allow_blank=True, allow_null=True)
    text_lines = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    
    # Metadata
    channel_id = serializers.CharField(max_length=200, required=False, allow_blank=True, allow_null=True)
    type = serializers.CharField(max_length=50, required=False, allow_blank=True, allow_null=True)
    conversation_title = serializers.CharField(max_length=500, required=False, allow_blank=True, allow_null=True)
    people = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    
    # Images (base64 or URLs)
    large_icon_base64 = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    picture_base64 = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class IngestInteractionSerializer(serializers.Serializer):
    """
    Validates incoming interaction event data from Android client.
    """
    package_name = serializers.CharField(max_length=255)
    app_label = serializers.CharField(max_length=255, required=False, allow_blank=True)
    notif_key = serializers.CharField(max_length=256, required=False, allow_blank=True, allow_null=True)
    removed_at = serializers.DateTimeField()
    raw_reason = serializers.IntegerField(required=False, allow_null=True)
    interaction_type = serializers.ChoiceField(
        choices=[t[0] for t in InteractionEvent.INTERACTION_TYPES]
    )
    metadata = serializers.JSONField(required=False, allow_null=True)


# -------------------------
# App Serializer
# -------------------------

class AppSerializer(serializers.ModelSerializer):
    """
    Serializes app information.
    """
    notification_count = serializers.SerializerMethodField()
    
    class Meta:
        model = App
        fields = [
            "id",
            "package_name",
            "app_label",
            "icon_reference",
            "first_seen",
            "last_seen",
            "notification_count",
        ]
        read_only_fields = ("id", "first_seen", "last_seen")
    
    def get_notification_count(self, obj):
        """
        Returns count of notifications from this app.
        Only included if 'include_counts' context is True.
        """
        if self.context.get('include_counts', False):
            return obj.notification_events.count()
        return None


# -------------------------
# DailyAggregate Serializer
# -------------------------

class DailyAggregateSerializer(serializers.ModelSerializer):
    """
    Serializes daily aggregate statistics.
    """
    app_label = serializers.CharField(source="app.app_label", read_only=True)
    package_name = serializers.CharField(source="app.package_name", read_only=True)
    
    class Meta:
        model = DailyAggregate
        fields = [
            "id",
            "day",
            "package_name",
            "app_label",
            "posts",
            "clicks",
            "swipes",
            "open_rate",
            "last_updated",
        ]
        read_only_fields = ("id", "last_updated")


# -------------------------
# InteractionEvent Serializer (for debugging/analytics)
# -------------------------

class InteractionEventSerializer(serializers.ModelSerializer):
    """
    Serializes interaction events (for analytics endpoints).
    """
    package_name = serializers.CharField(source='notification_event.app.package_name', read_only=True)
    app_label = serializers.CharField(source='notification_event.app.app_label', read_only=True)
    notification_title = serializers.CharField(source='notification_event.title', read_only=True)
    
    class Meta:
        model = InteractionEvent
        fields = [
            "id",
            "user",
            "notification_event",
            "package_name",
            "app_label",
            "notification_title",
            "interaction_type",
            "timestamp",
            "raw_reason",
            "metadata",
            "created_at",
        ]
        read_only_fields = ("id", "created_at")

class NotificationAnalyticsSerializer(serializers.Serializer):
    thisWeekCount = serializers.IntegerField()
    ignoreRate = serializers.FloatField()
    avgResponse = serializers.CharField()
    weeklyActivity = serializers.ListField(
        child=serializers.IntegerField()
    )
    timeDistribution = serializers.ListField(
        child=serializers.IntegerField()
    )
    insights = serializers.DictField() 