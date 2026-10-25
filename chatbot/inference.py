from dataclasses import dataclass

from chatbot.config import Settings
from chatbot.model_loader import ModelResources, OllamaError, request_json


# Keep API response metadata alongside the answer for the conversation footer.
@dataclass(frozen=True)
class Reply:
    text: str
    removed_turns: int
    input_tokens: int
    output_tokens: int
    reached_limit: bool


# Estimate context conservatively without installing a second model tokenizer.
def estimate_tokens(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"].encode("utf-8")) + 32 for message in messages) + 32


# Preserve the system prompt and newest question while trimming complete old turns.
def prepare_context(
    messages: list[dict[str, str]],
    settings: Settings,
) -> tuple[list[dict[str, str]], int]:
    if not messages or len(messages) % 2 == 0:
        raise ValueError("The conversation must end with a user message.")

    # Reject malformed history so user and assistant messages stay aligned.
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            raise ValueError("History must contain alternating user and assistant messages.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Messages must contain non-empty text.")

    # Copy messages so trimming never changes the visible session history.
    context = [dict(message) for message in messages]
    system = {"role": "system", "content": settings.system_prompt}
    removed_turns = 0
    token_budget = settings.context_tokens - settings.max_new_tokens

    # Reserve answer space and apply both the turn limit and the size estimate.
    while (
        len(context) // 2 > settings.max_history_turns
        or estimate_tokens([system, *context]) > token_budget
    ):
        if len(context) == 1:
            raise ValueError(
                "The question or system prompt is too long for the context limit. "
                "Shorten it or increase CONTEXT_TOKENS."
            )
        context = context[2:]
        removed_turns += 1

    return [system, *context], removed_turns


# Request a complete answer from Ollama using the selected conversation context.
def generate_reply(
    resources: ModelResources,
    messages: list[dict[str, str]],
    settings: Settings,
) -> Reply:
    context, removed_turns = prepare_context(messages, settings)
    payload = {
        "model": resources.name,
        "messages": context,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": settings.context_tokens,
            "num_predict": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "repeat_penalty": settings.repetition_penalty,
        },
    }
    data = request_json(settings, "POST", "/api/chat", payload)

    # Accept only completed, non-empty assistant replies into saved history.
    message = data.get("message")
    if data.get("done") is not True:
        raise OllamaError("Ollama returned an incomplete answer. Please try again.")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise OllamaError("Ollama did not return an assistant message.")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise OllamaError("The model returned an empty answer. Please try again.")

    return Reply(
        text=text.strip(),
        removed_turns=removed_turns,
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        reached_limit=data.get("done_reason") == "length",
    )
