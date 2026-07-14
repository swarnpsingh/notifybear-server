from django.contrib import admin
from django.db.models import Count

from .models import UserNotificationRules, RulesGenerationLog, PromptInjectionAttempt


@admin.register(UserNotificationRules)
class UserNotificationRulesAdmin(admin.ModelAdmin):
    list_display = ("user", "short_summary", "updated_at")
    list_filter = ("updated_at",)
    search_fields = ("user__username", "user__email")
    readonly_fields = ("updated_at",)
    ordering = ("-updated_at",)

    def short_summary(self, obj):
        if not obj.summary:
            return "-"
        return obj.summary[:80] + ("..." if len(obj.summary) > 80 else "")
    short_summary.short_description = "Summary"


@admin.register(RulesGenerationLog)
class RulesGenerationLogAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "user_total")
    list_filter = ("created_at",)
    search_fields = ("user__username", "user__email")
    ordering = ("-created_at",)
    readonly_fields = ("user", "created_at")

    def get_queryset(self, request):
        # Joins back through the user to all of their log rows, so every row
        # carries that user's lifetime generation count.
        return super().get_queryset(request).annotate(
            _user_total=Count("user__rules_generation_logs")
        )

    def user_total(self, obj):
        return obj._user_total
    user_total.short_description = "Total for user"
    user_total.admin_order_field = "_user_total"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(PromptInjectionAttempt)
class PromptInjectionAttemptAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "short_prompt", "short_note")
    list_filter = ("created_at",)
    search_fields = ("user__username", "user__email", "prompt")
    ordering = ("-created_at",)
    readonly_fields = ("user", "prompt", "note", "created_at")

    def short_prompt(self, obj):
        return obj.prompt[:80] + ("..." if len(obj.prompt) > 80 else "")
    short_prompt.short_description = "Prompt"

    def short_note(self, obj):
        if not obj.note:
            return "-"
        return obj.note[:80] + ("..." if len(obj.note) > 80 else "")
    short_note.short_description = "Note"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False