from dataclasses import dataclass
from typing import Any

import requests

from chatbot.config import Settings


# Represent API failures with messages that can be displayed in the interface.
class OllamaError(RuntimeError):
    pass


# Describe the installed model without loading its weights into Streamlit.
@dataclass(frozen=True)
class ModelResources:
    name: str
    quantization: str


# Send one JSON request and translate transport or server failures into clear feedback.
def request_json(
    settings: Settings,
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{settings.ollama_base_url}{endpoint}"
    timeout = settings.request_timeout if endpoint == "/api/chat" else 10

    # Close each response and use a short connection timeout for unavailable servers.
    try:
        with requests.request(
            method,
            url,
            json=payload,
            timeout=(5, timeout),
        ) as response:
            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaError(
                    f"Ollama returned invalid JSON (HTTP {response.status_code}). "
                    "Check OLLAMA_BASE_URL."
                ) from exc

            # Inspect API errors before returning a successful JSON body.
            if not isinstance(data, dict):
                raise OllamaError("Ollama returned an unexpected response format.")
            if not response.ok or data.get("error"):
                detail = data.get("error") or response.reason
                raise OllamaError(
                    f"Ollama request failed (HTTP {response.status_code}): {detail}"
                )
            return data
    except requests.Timeout as exc:
        raise OllamaError(
            "Ollama timed out. Try again, shorten the conversation, or increase OLLAMA_TIMEOUT."
        ) from exc
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Start Ollama and check OLLAMA_BASE_URL."
        ) from exc
    except requests.RequestException as exc:
        raise OllamaError(f"Could not complete the Ollama request: {exc}") from exc


# Verify the selected model without downloading or switching models.
def get_model(settings: Settings) -> ModelResources:
    data = request_json(settings, "GET", "/api/tags")
    models = data.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama did not return a valid model list.")

    # Accept an implicit latest tag while keeping custom tags exact.
    requested = settings.ollama_model
    if ":" not in requested.rsplit("/", 1)[-1]:
        requested = f"{requested}:latest"
    for model in models:
        if isinstance(model, dict) and model.get("name") == requested:
            details = model.get("details") or {}
            return ModelResources(requested, details.get("quantization_level", "GGUF"))

    raise OllamaError(
        f"Model '{settings.ollama_model}' is not installed in Ollama. "
        "Run 'ollama list' and set OLLAMA_MODEL to the exact installed name."
    )
