from rest_framework import serializers
from .models import User, UserProfile


class UserProfileSerializer(serializers.ModelSerializer):
    dp_url = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = ["dp", "dp_url"]

    def get_dp_url(self, obj):
        if obj.dp and hasattr(obj.dp, 'url'):
            return obj.dp.url
        return None


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)
    initials = serializers.ReadOnlyField()

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "first_name",
            "last_name",
            "email",
            "initials",
            "profile",
        ]

class UserSignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ["username", "email", "first_name", "last_name", "password"]

    def create(self, validated_data):
        password = validated_data.pop("password")
        user = User.objects.create(**validated_data)
        user.set_password(password)
        user.save()

        # create UserProfile automatically
        UserProfile.objects.get_or_create(user=user)

        return user
