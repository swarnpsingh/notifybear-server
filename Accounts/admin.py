from django.contrib import admin
from .models import User, UserProfile, AuthAuditLog

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