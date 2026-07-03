from django.contrib import admin

from .models import TrainingFeature


@admin.register(TrainingFeature)
class TrainingFeatureAdmin(admin.ModelAdmin):

    list_display = (
        "id",
        "user",
        "package_name",
        "feature_version",
        "label",
        "original_score",
        "used_for_training",
    )

    list_filter = (
        "feature_version",
        "used_for_training",
        "created_at",
    )

    search_fields = (
        "user__username",
        "package_name",
        "notification_key",
    )

    readonly_fields = (
        "created_at",
        "original_score",
        "synced_at",
    )

    ordering = ("-created_at",)

    list_per_page = 50