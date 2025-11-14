from django.contrib import admin
from .models import (
    Notifications,
    NotificationMessage,
    App,
    InteractionEvent,
    DailyAggregate,
)


class NotificationMessageInline(admin.TabularInline):
    model = NotificationMessage
    extra = 0
    fields = ("sender", "message_text", "message_time")
    readonly_fields = ("sender", "message_text", "message_time")


@admin.register(Notifications)
class NotificationsAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "package_name",
        "title",
        "post_time",
        "timestamp_opened",
        "reaction_time",
    )
    list_filter = ("package_name", "post_time", "timestamp_opened")
    search_fields = ("user__email", "title", "text", "package_name")
    readonly_fields = ("reaction_time", "created_at", "post_time")
    inlines = [NotificationMessageInline]

    fieldsets = (
        ("User & Source", {
            "fields": ("user", "package_name", "channel_id", "notif_key")
        }),
        ("Timestamps", {
            "fields": ("post_time", "timestamp_opened", "timestamp_dismissed", "reaction_time", "created_at")
        }),
        ("Text Content", {
            "fields": (
                "title", "text", "big_text", "sub_text", "summary_text", "info_text", "text_lines"
            )
        }),
        ("Images", {
            "fields": ("large_icon_base64", "picture_base64")
        }),
        ("Conversation / Group Info", {
            "fields": ("conversation_title", "people")
        }),
    )


@admin.register(NotificationMessage)
class NotificationMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "notification", "sender", "message_text", "message_time")
    search_fields = ("sender", "message_text")
    list_filter = ("message_time",)


@admin.register(App)
class AppAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "app_label", "package_name", "first_seen")
    search_fields = ("app_label", "package_name", "user__email")
    list_filter = ("first_seen",)
    readonly_fields = ("first_seen",)


@admin.register(InteractionEvent)
class InteractionEventAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "app", "interaction_type", "removed_at", "raw_reason", "created_at")
    search_fields = ("notif_key", "user__email", "app__package_name")
    list_filter = ("interaction_type", "removed_at")
    readonly_fields = ("created_at",)


@admin.register(DailyAggregate)
class DailyAggregateAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "app", "day", "posts", "clicks", "swipes")
    search_fields = ("app__package_name", "user__email")
    list_filter = ("day",)
    readonly_fields = ("day", "posts", "clicks", "swipes")
