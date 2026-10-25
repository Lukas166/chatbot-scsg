import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


# Resolve project paths independently of the current working directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = "Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft-Q4_K_M-GGUF"
OLLAMA_MODEL = f"hf.co/{MODEL_ID}:latest"
DATASET_FILENAME = "alpaca-gpt4-indonesian-cleaned.csv"
DEFAULT_DATASET_PATH = PROJECT_DIR / "dataset" / DATASET_FILENAME
SUGGESTION_COUNT = 2
MAX_RESPONSE_TOKENS = 4096
MIN_QUESTION_LENGTH = 12
MAX_QUESTION_LENGTH = 180
SYSTEM_PROMPT = (
    "You are an Indonesian chatbot powered by a fine-tuned Qwen2.5 1.5B model, "
    "served locally through Ollama in GGUF Q4_K_M format. "
    f"Your model repository is {MODEL_ID}. "
    "Use this identity when asked about your model. Reply in Indonesian unless the user requests "
    "another language. All earlier user and assistant messages are important "
    "conversation context. When asked what the user previously asked, quote every "
    "previous user question or request in a numbered list, oldest first. Include "
    "all relevant details. Do not include the current recap request. Never leave "
    "out an earlier question that appears in the supplied history. Do not invent "
    "missing history. If older messages were omitted, clearly say the recap covers "
    "only the available history. For other requests, answer clearly and concisely. "
    "If unsure, say so."
)


# Store application settings separately from the interface and API client.
@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = OLLAMA_MODEL
    request_timeout: int = 120
    context_tokens: int = 8192
    max_new_tokens: int = 2048
    max_history_turns: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.08
    system_prompt: str = SYSTEM_PROMPT
    dataset_path: Path = DEFAULT_DATASET_PATH


# Read environment variables and validate settings before opening the chat.
def load_settings() -> Settings:
    dataset_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_DIR / dataset_path

    # Use the exact installed model rather than falling back to another model.
    settings = Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL).strip(),
        request_timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        context_tokens=int(os.getenv("CONTEXT_TOKENS", "8192")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "2048")),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "50")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.08")),
        system_prompt=os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT).strip(),
        dataset_path=dataset_path,
    )

    # Reject invalid endpoints and generation limits with actionable messages.
    endpoint = urlsplit(settings.ollama_base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("OLLAMA_BASE_URL must be a valid HTTP or HTTPS URL.")
    if endpoint.query or endpoint.fragment or endpoint.username or endpoint.password:
        raise ValueError("OLLAMA_BASE_URL must not include credentials, a query, or a fragment.")
    if not settings.ollama_model:
        raise ValueError("OLLAMA_MODEL must not be empty.")
    if not 1 <= settings.request_timeout <= 600:
        raise ValueError("OLLAMA_TIMEOUT must be between 1 and 600 seconds.")
    if not 512 <= settings.context_tokens <= 32768:
        raise ValueError("CONTEXT_TOKENS must be between 512 and 32768.")
    if not 32 <= settings.max_new_tokens <= min(MAX_RESPONSE_TOKENS, settings.context_tokens // 2):
        raise ValueError(
            f"MAX_NEW_TOKENS must be 32-{MAX_RESPONSE_TOKENS} "
            "and at most half the context size."
        )
    if not 0 <= settings.max_history_turns <= 50:
        raise ValueError("MAX_HISTORY_TURNS must be between 0 and 50.")
    if not 0 <= settings.temperature <= 2:
        raise ValueError("TEMPERATURE must be between 0 and 2.")
    if not 0 < settings.top_p <= 1:
        raise ValueError("TOP_P must be greater than 0 and at most 1.")
    if not 1 <= settings.repetition_penalty <= 2:
        raise ValueError("REPETITION_PENALTY must be between 1 and 2.")
    if not settings.system_prompt:
        raise ValueError("SYSTEM_PROMPT must not be empty.")

    return settings
