from django.contrib import admin
from .models import User, UserKey, UserProfile, AuthAuditLog, DeletedAccount

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email',)
    search_fields = ('username', 'email',)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user',)
    search_fields = ('user__username', 'user__email',)

@admin.register(AuthAuditLog)
class AuthAuditLogAdmin(admin.ModelAdmin):
    list_display = ("event", "username", "ip", "timestamp")
    list_filter = ("event", "timestamp")
    search_fields = ("username", "ip", "user__username")
    ordering = ("-timestamp",)
    readonly_fields = ("event", "username", "user", "ip", "user_agent", "timestamp", "detail")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

@admin.register(DeletedAccount)
class DeletedAccountAdmin(admin.ModelAdmin):
    list_display = ('email', 'deleted_at')
    list_filter = ('deleted_at',)
    search_fields = ('email',)
    readonly_fields = ('email', 'reasons', 'other_reason', 'deleted_at')
    ordering = ('-deleted_at',)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

@admin.register(UserKey)
class UserKeyAdmin(admin.ModelAdmin):
    list_display = ("user", "short_key", "created_at")
    search_fields = ("user__username", "user__email")
    readonly_fields = ("created_at",)

    def short_key(self, obj):
        if not obj.wrapped_key:
            return "-"
        return obj.wrapped_key[:20] + "..."
    short_key.short_description = "Wrapped Key"