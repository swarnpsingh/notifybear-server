from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.db.models import Count
from .models import (
    NotificationEvent,
    UserNotificationState,
    NotificationMessage,
    App,
    InteractionEvent,
    DailyAggregate,
)


# -------------------------
# Inline Admin Classes
# -------------------------

class NotificationMessageInline(admin.TabularInline):
    model = NotificationMessage
    extra = 0
    fields = ("sender", "message_text", "message_time")
    readonly_fields = ("sender", "message_text", "message_time")
    can_delete = False


class UserNotificationStateInline(admin.TabularInline):
    """Show user states for a notification event."""
    model = UserNotificationState
    extra = 0
    fields = ("user", "is_read", "opened_at", "dismissed_at", "ml_score")
    readonly_fields = ("user", "opened_at", "dismissed_at")
    can_delete = True


class InteractionEventInline(admin.TabularInline):
    """Show interaction events for a notification."""
    model = InteractionEvent
    extra = 0
    fields = ("user", "interaction_type", "timestamp", "raw_reason")
    readonly_fields = ("user", "interaction_type", "timestamp", "raw_reason", "created_at")
    can_delete = False


# -------------------------
# App Admin
# -------------------------

@admin.register(App)
class AppAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "app_label",
        "package_name",
        "notification_count",
        "total_interactions",
        "first_seen",
        "last_seen"
    )
    list_filter = ("first_seen", "last_seen", "user")
    search_fields = ("app_label", "package_name", "user__email", "user__username")
    readonly_fields = ("first_seen", "last_seen")
    
    fieldsets = (
        ("App Information", {
            "fields": ("user", "package_name", "app_label", "icon_reference")
        }),
        ("Timestamps", {
            "fields": ("first_seen", "last_seen")
        }),
        ("Statistics", {
            "fields": (),
            "classes": ("collapse",)
        }),
    )
    
    def notification_count(self, obj):
        """Count of notifications from this app."""
        count = obj.notification_events.count()
        url = reverse(
            f"admin:{NotificationEvent._meta.app_label}_{NotificationEvent._meta.model_name}_changelist"
        ) + f"?app__id__exact={obj.id}"
        return format_html('<a href="{}">{} notifications</a>', url, count)
    notification_count.short_description = "Notifications"
    
    def total_interactions(self, obj):
        """Count of interactions for this app."""
        count = obj.notification_events.aggregate(
            total=Count('interaction_events')
        )['total'] or 0
        return f"{count} interactions"
    total_interactions.short_description = "Total Interactions"


# -------------------------
# NotificationEvent Admin (Immutable)
# -------------------------

@admin.register(NotificationEvent)
class NotificationEventAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "app_link",
        "user_email",
        "notif_key_short",
        "title",
        "post_time",
        "user_count",
        "interaction_count",
        "created_at",
        "type"
    )
    list_filter = ("post_time", "created_at", "app__package_name", "app__user", "type")
    search_fields = (
        "title",
        "text",
        "notif_key",
        "app__package_name",
        "app__app_label",
        "app__user__email"
    )
    readonly_fields = (
        "id",
        "app",
        "notif_key",
        "post_time",
        "created_at",
        "content_hash",
        "user_count",
        "interaction_count"
    )
    inlines = [NotificationMessageInline, UserNotificationStateInline, InteractionEventInline]
    
    fieldsets = (
        ("Source & Identity", {
            "fields": ("app", "notif_key", "channel_id", "content_hash")
        }),
        ("Timestamps", {
            "fields": ("post_time", "created_at")
        }),
        ("Text Content", {
            "fields": (
                "title", "text", "big_text", "sub_text", 
                "summary_text", "info_text", "text_lines"
            )
        }),
        ("Images", {
            "fields": ("large_icon_base64", "picture_base64"),
            "classes": ("collapse",)
        }),
        ("Conversation Metadata", {
            "fields": ("conversation_title", "people"),
            "classes": ("collapse",)
        }),
        ("Statistics", {
            "fields": ("user_count", "interaction_count"),
            "classes": ("collapse",)
        }),
    )
    
    def has_change_permission(self, request, obj=None):
        """
        Prevent editing immutable notification events.
        Only allow viewing.
        """
        if obj:  # Editing existing object
            return False
        return True  # Allow creation (though unlikely via admin)
    
    def app_link(self, obj):
        """Link to the app admin page."""
        url = reverse(
            f"admin:{obj.app._meta.app_label}_{obj.app._meta.model_name}_change",
            args=[obj.app.id]
        )
        return format_html('<a href="{}">{}</a>', url, obj.app.app_label or obj.app.package_name)
    app_link.short_description = "App"
    
    def user_email(self, obj):
        """Get user email through app relationship."""
        return obj.app.user.email if hasattr(obj.app.user, 'email') else str(obj.app.user)
    user_email.short_description = "User"
    
    def notif_key_short(self, obj):
        """Truncated notification key."""
        if obj.notif_key and len(obj.notif_key) > 30:
            return f"{obj.notif_key[:30]}..."
        return obj.notif_key or "—"
    notif_key_short.short_description = "Notif Key"
    
    def user_count(self, obj):
        """Number of users who received this notification."""
        count = obj.user_states.count()
        return f"{count} user(s)"
    user_count.short_description = "Seen By"
    
    def interaction_count(self, obj):
        """Number of interactions with this notification."""
        count = obj.interaction_events.count()
        return f"{count} interaction(s)"
    interaction_count.short_description = "Interactions"


# -------------------------
# UserNotificationState Admin (Mutable)
# -------------------------

@admin.register(UserNotificationState)
class UserNotificationStateAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "is_read",
        "user_link",
        "notification_link",
        "opened_at",
        "dismissed_at",
        "ml_score",
    )
    list_filter = (
        "is_read",
        "opened_at",
        "dismissed_at",
        "notification_event__app__package_name",
        "user",
        "notification_event__type"
    )
    search_fields = (
        "user__email",
        "user__username",
        "notification_event__title",
        "notification_event__text",
        "notification_event__notif_key"
    )
    readonly_fields = (
        "user",
        "notification_event",
        "created_at",
        "last_updated",
    )
    
    fieldsets = (
        ("Identity", {
            "fields": ("user", "notification_event")
        }),
        ("User State", {
            "fields": ("is_read", "opened_at", "dismissed_at", "snoozed_until")
        }),
        ("ML & Scoring", {
            "fields": ("ml_score",)
        }),
        ("Metadata", {
            "fields": ("created_at", "last_updated"),
            "classes": ("collapse",)
        }),
    )
    @admin.display(description="User")
    def user_link(self, obj):
        """Link to user admin page."""
        url = reverse(
            f"admin:{obj.user._meta.app_label}_{obj.user._meta.model_name}_change",
            args=[obj.user.id]
        )
        return format_html('<a href="{}">{}</a>', url, obj.user.email or obj.user.username)
    user_link.short_description = "User"
    @admin.display(description="Notification")
    def notification_link(self, obj):
        """Link to notification event admin page."""
        url = reverse(
    f"admin:{obj.notification_event._meta.app_label}_{obj.notification_event._meta.model_name}_change",
        args=[obj.notification_event.id]
    )
        title = obj.notification_event.title or "No title"
        if len(title) > 40:
            title = f"{title[:40]}..."
        return format_html('<a href="{}">{}</a>', url, title)
    notification_link.short_description = "Notification"
    @admin.display(description="App")
    def app_name(self, obj):
        """Display app name."""
        return obj.notification_event.app.app_label or obj.notification_event.app.package_name
    app_name.short_description = "App"
    @admin.display(description="Reaction Time")
    def reaction_time_display(self, obj):
        """Display reaction time in human-readable format."""
        rt = obj.reaction_time
        if rt:
            seconds = rt.total_seconds()
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        return "—"
    reaction_time_display.short_description = "Reaction Time"


# -------------------------
# NotificationMessage Admin
# -------------------------

@admin.register(NotificationMessage)
class NotificationMessageAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "notification_event",
        "sender",
        "message_preview",
        "message_time"
    )
    list_filter = ("message_time",)
    search_fields = ("sender", "message_text", "notification_event__title")
    readonly_fields = ("notification_event", "sender", "message_text", "message_time")
    @admin.display(description="NotificationMessage")
    def message_preview(self, obj):
        """Show truncated message text."""
        if obj.message_text and len(obj.message_text) > 50:
            return f"{obj.message_text[:50]}..."
        return obj.message_text or "—"
    message_preview.short_description = "Message"


# -------------------------
# InteractionEvent Admin (Append-only)
# -------------------------

@admin.register(InteractionEvent)
class InteractionEventAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "interaction_type",
        "user_link",
        "app_name",
        "timestamp",
        "raw_reason",
        "created_at"
    )
    list_filter = ("interaction_type", "timestamp", "notification_event__app__package_name")
    search_fields = (
        "user__email",
        "user__username",
        "notification_event__title",
        "notification_event__notif_key"
    )
    readonly_fields = (
        "user",
        "notification_event",
        "interaction_type",
        "timestamp",
        "raw_reason",
        "metadata",
        "created_at"
    )
    
    fieldsets = (
        ("Identity", {
            "fields": ("user", "notification_event")
        }),
        ("Interaction Details", {
            "fields": ("interaction_type", "timestamp", "raw_reason", "metadata")
        }),
        ("Metadata", {
            "fields": ("created_at",)
        }),
    )
    
    def has_change_permission(self, request, obj=None):
        """
        Prevent editing append-only interaction events.
        Only allow viewing.
        """
        return False
    
    def has_delete_permission(self, request, obj=None):
        """
        Prevent deleting interaction events (append-only).
        """
        return request.user.is_superuser  # Only superusers can delete
    @admin.display(description="InteractionEvent")
    def user_link(self, obj):
        """Link to user admin page."""
        url = reverse(
            f"admin:{obj.user._meta.app_label}_{obj.user._meta.model_name}_change",
            args=[obj.user.id]
        )
        return format_html('<a href="{}">{}</a>', url, obj.user.email or obj.user.username)
    user_link.short_description = "User"
    @admin.display(description="InteractionEvent")
    def app_name(self, obj):
        """Display app name."""
        return obj.notification_event.app.app_label or obj.notification_event.app.package_name
    app_name.short_description = "App"


# -------------------------
# DailyAggregate Admin
# -------------------------

@admin.register(DailyAggregate)
class DailyAggregateAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user_link",
        "app_link",
        "day",
        "posts",
        "clicks",
        "swipes",
        "open_rate_display",
        "last_updated"
    )
    list_filter = ("day", "user", "app__package_name")
    search_fields = ("app__package_name", "app__app_label", "user__email")
    readonly_fields = ("last_updated",)
    
    fieldsets = (
        ("Identity", {
            "fields": ("user", "app", "day")
        }),
        ("Metrics", {
            "fields": ("posts", "clicks", "swipes", "open_rate")
        }),
        ("Metadata", {
            "fields": ("last_updated",)
        }),
    )
    
    def user_link(self, obj):
        """Link to user admin page."""
        url = reverse(
            f"admin:{obj.user._meta.app_label}_{obj.user._meta.model_name}_change",
            args=[obj.user.id]
        )
        return format_html('<a href="{}">{}</a>', url, obj.user.email or obj.user.username)
    user_link.short_description = "User"
    
    def app_link(self, obj):
        """Link to app admin page."""
        url = reverse(
            f"admin:{obj.app._meta.app_label}_{obj.app._meta.model_name}_change",
            args=[obj.app.id]
        )
        return format_html('<a href="{}">{}</a>', url, obj.app.app_label or obj.app.package_name)
    app_link.short_description = "App"
    
    def open_rate_display(self, obj):
        """Display open rate as percentage."""
        if obj.open_rate is not None:
            return f"{obj.open_rate * 100:.1f}%"
        return "—"
    open_rate_display.short_description = "Open Rate"


# -------------------------
# Customize Admin Site
# -------------------------

admin.site.site_header = "NotifyBear Admin"
admin.site.site_title = "NotifyBear"
admin.site.index_title = "Notification Intelligence System"