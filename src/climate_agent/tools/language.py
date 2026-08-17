from langdetect import LangDetectException, detect

DEFAULT_LANGUAGE = "English"

# Not exhaustive — common languages likely to show up in real queries. Anything undetected or
# unmapped falls back to English rather than passing a raw ISO code into the narration prompt.
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "hi": "Hindi",
}


def detect_language(text: str) -> str:
    """Detect the human-readable language of a query, so narration can respond in kind.

    Explicit detection (langdetect), not implicit LLM inference — real, loggable, traceable
    value (shows up as a real trace attribute), not something to just trust the model got
    right. Known limitation: statistical language ID is less reliable on short text (most
    queries here are one sentence) — falls back to English on any detection failure or
    unmapped code rather than guessing.

    Args: text — the user's raw query.
    Returns: a human-readable language name (e.g. "Spanish"), defaulting to "English".
    """
    try:
        code = detect(text)
    except LangDetectException:
        return DEFAULT_LANGUAGE
    return LANGUAGE_NAMES.get(code, DEFAULT_LANGUAGE)
