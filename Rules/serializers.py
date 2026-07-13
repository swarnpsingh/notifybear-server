from rest_framework import serializers

# ~150 words. Enforced here regardless of what the client sends.
MAX_PROMPT_CHARS = 1000


class GenerateRulesRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(
        min_length=3,
        max_length=MAX_PROMPT_CHARS,
        trim_whitespace=True,
    )
