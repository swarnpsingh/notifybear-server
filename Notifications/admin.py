from django.contrib import admin
from .models import Notifications, NotificationMessage


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
            "fields": ("user", "package_name", "channel_id")
        }),
        ("Timestamps", {
            "fields": ("post_time", "timestamp_opened", "reaction_time", "created_at")
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
