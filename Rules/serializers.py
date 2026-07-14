from rest_framework import serializers

# ~150 words. Enforced here regardless of what the client sends.
MAX_PROMPT_CHARS = 1000

# Launcher apps on a typical phone number ~100-200; anything beyond this is
# either a broken client or padding, and only bloats the LLM prompt.
MAX_APPS = 400
MAX_APP_FIELD_CHARS = 100


class InstalledAppSerializer(serializers.Serializer):
    packageName = serializers.CharField(max_length=MAX_APP_FIELD_CHARS, trim_whitespace=True)
    appName = serializers.CharField(max_length=MAX_APP_FIELD_CHARS, trim_whitespace=True)


class GenerateRulesRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(
        min_length=3,
        max_length=MAX_PROMPT_CHARS,
        trim_whitespace=True,
    )
    # Optional so older app versions that don't send it keep working.
    apps = InstalledAppSerializer(many=True, required=False, max_length=MAX_APPS)