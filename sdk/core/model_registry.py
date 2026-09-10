"""The Gemini Live models this server can serve, keyed by their real names.

Model ids here are the exact strings `live.connect(model=...)` expects, so a
client configures the same name it would use against Gemini directly — there is
no alias layer to keep in sync.

The list is deliberately curated rather than read from `client.models.list()`.
Seven models advertise `bidiGenerateContent`, but only these four actually hold
a text conversation over an audio-transcription session; the other three
(`gemini-3.5-transcribe-live`, `gemini-robotics-er-2-streaming-preview`,
`gemini-3.5-live-translate-preview`) either reject the AUDIO modality with
`1007` or never answer. Listing them would only hand clients ids that fail at
request time.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelInfo:
    """One servable model. Capability flags are what this server can honour."""

    id: str
    description: str
    supports_tools: bool = True
    supports_vision: bool = True


DEFAULT_MODEL_ID = "gemini-3.1-flash-live-preview"

MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gemini-3.1-flash-live-preview",
        description="Gemini 3.1 Flash Live — mặc định, độ trễ thấp.",
    ),
    ModelInfo(
        id="gemini-2.5-flash-native-audio-preview-12-2025",
        description="Gemini 2.5 Flash native audio, bản xem trước 12-2025.",
    ),
    ModelInfo(
        id="gemini-2.5-flash-native-audio-latest",
        description="Gemini 2.5 Flash native audio, bản mới nhất.",
    ),
    ModelInfo(
        id="gemini-2.5-flash-native-audio-preview-09-2025",
        description="Gemini 2.5 Flash native audio, bản xem trước 09-2025.",
    ),
)

_BY_ID = {model.id: model for model in MODELS}


def strip_tag(name: str) -> str:
    """Drop an Ollama-style `:tag` suffix.

    Ollama clients address models as `name:latest`, and Gemini model ids never
    contain a colon, so anything after the first one is a tag.
    """
    return name.split(":", 1)[0] if name else ""


def find(name: Optional[str]) -> Optional[ModelInfo]:
    """Look up a model by id or `id:tag`, or None if it is not servable."""
    return _BY_ID.get(strip_tag(name or ""))


def resolve(name: Optional[str]) -> str:
    """Model id to actually connect with.

    Unknown ids fall back to the default instead of erroring: the Ollama and
    OpenAI clients pointed at this server frequently send a name they picked
    from their own config, and refusing those would break setups that work
    today.
    """
    model = find(name)
    return model.id if model else DEFAULT_MODEL_ID


DEFAULT_MODEL = _BY_ID[DEFAULT_MODEL_ID]
