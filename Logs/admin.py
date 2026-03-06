from django.contrib import admin

from .models import MobileLog


@admin.register(MobileLog)
class MobileLogAdmin(admin.ModelAdmin):
	list_display = ("id", "user", "tag", "timestamp")
	list_filter = ("tag", "timestamp")
	search_fields = ("user__username", "user__email", "tag", "message")
	ordering = ("-timestamp",)
	readonly_fields = ("user", "tag", "message", "timestamp")

	def has_add_permission(self, request):
		return False

	def has_change_permission(self, request, obj=None):
		return False
