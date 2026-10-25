import logging
from dataclasses import replace
from pathlib import Path

import streamlit as st

from chatbot.config import MAX_RESPONSE_TOKENS, MODEL_ID, SUGGESTION_COUNT, Settings, load_settings
from chatbot.dataset_utils import read_questions, select_questions
from chatbot.inference import generate_reply
from chatbot.model_loader import ModelResources, OllamaError, get_model


logger = logging.getLogger(__name__)


# Cache prompts and invalidate cached data when the CSV modification time changes.
@st.cache_data(show_spinner=False, max_entries=4)
def cached_questions(path: str, modified_ns: int) -> list[str]:
    return read_questions(Path(path))


# Create independent conversation state for each Streamlit browser session.
def initialize_session() -> None:
    for key, default in {
        "messages": [],
        "pending_question": None,
        "generation_error": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


# Queue typed questions, suggestions, and retries through the same generation path.
def queue_question(question: str) -> None:
    if question.strip():
        st.session_state.pending_question = question.strip()
        st.session_state.generation_error = None


# Reset conversation state while preserving the suggestion selection.
def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.generation_error = None
    st.session_state.pop("last_reply", None)


# Render generation controls and return the settings selected for this request.
def render_sidebar(settings: Settings, busy: bool) -> Settings:
    with st.sidebar:
        st.header("Settings")
        st.markdown(f"[Fine-tuned GGUF model](https://huggingface.co/{MODEL_ID})")
        st.caption(f"Ollama: {settings.ollama_base_url}")
        max_new_tokens = st.slider(
            "Maximum response tokens",
            min_value=32,
            max_value=min(MAX_RESPONSE_TOKENS, settings.context_tokens // 2),
            value=settings.max_new_tokens,
            disabled=busy,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.temperature,
            step=0.05,
            disabled=busy,
            help="Lower values give more predictable answers; higher values add variety.",
        )
        st.caption(f"Context limit: {settings.context_tokens:,} tokens, including the answer.")
        st.button("Clear conversation", on_click=clear_chat, disabled=busy)

    return replace(settings, max_new_tokens=max_new_tokens, temperature=temperature)


# Verify the configured model and allow connection failures to be retried.
def ensure_model(settings: Settings) -> ModelResources:
    try:
        resources = get_model(settings)
    except OllamaError as exc:
        logger.warning("Ollama is unavailable: %s", exc)
        st.error(str(exc))
        if st.button("Retry connection"):
            st.rerun()
        st.stop()

    st.sidebar.caption(f"Model: {resources.name} | {resources.quantization}")
    return resources


# Render local suggestions without making the dataset a requirement for chatting.
def render_suggestions(settings: Settings, busy: bool) -> None:
    st.caption("Suggested questions")
    try:
        path = settings.dataset_path
        modified_ns = path.stat().st_mtime_ns
        questions = cached_questions(str(path), modified_ns)
        source_key = (str(path), modified_ns, SUGGESTION_COUNT)

        # Refresh stale selections when the dataset or configured count changes.
        expected_count = min(SUGGESTION_COUNT, len(questions))
        if (
            st.session_state.get("suggestion_source") != source_key
            or len(st.session_state.get("suggestions", [])) != expected_count
        ):
            st.session_state.suggestions = select_questions(questions, count=SUGGESTION_COUNT)
            st.session_state.suggestion_source = source_key

        # Send suggestion clicks through the same callback as typed questions.
        suggestions = st.session_state.suggestions
        if len(suggestions) < SUGGESTION_COUNT:
            st.caption(f"The CSV provides only {len(suggestions)} valid suggestions.")
        if not suggestions:
            return
        columns = st.columns(min(2, len(suggestions)))
        for index, question in enumerate(suggestions):
            columns[index % len(columns)].button(
                question,
                key=f"suggestion_{index}",
                width="stretch",
                disabled=busy,
                on_click=queue_question,
                args=(question,),
            )
    except (OSError, UnicodeError, ValueError) as exc:
        logger.warning("Could not load suggested questions: %s", exc)
        st.warning(f"Suggestions are unavailable: {exc}. You can still type a message.")


# Generate a pending answer and save the turn only after the API call succeeds.
def process_question(resources: ModelResources, settings: Settings) -> None:
    pending = st.session_state.pending_question
    if pending is None:
        return

    with st.chat_message("user"):
        st.markdown(pending)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Writing a response..."):
                reply = generate_reply(
                    resources,
                    st.session_state.messages + [{"role": "user", "content": pending}],
                    settings,
                )
        except (OllamaError, ValueError) as exc:
            logger.warning("Could not generate an answer: %s", exc)
            st.session_state.generation_error = {"question": pending, "message": str(exc)}
        else:
            # Commit both messages together so retries never duplicate a user turn.
            st.session_state.messages.extend(
                [
                    {"role": "user", "content": pending},
                    {"role": "assistant", "content": reply.text},
                ]
            )
            st.session_state.last_reply = reply
            st.session_state.generation_error = None

    st.session_state.pending_question = None
    st.rerun()


# Show retry controls or metadata for the last successfully completed answer.
def render_feedback() -> None:
    error = st.session_state.generation_error
    if error:
        st.error(error["message"])
        st.caption(f"Unanswered question: {error['question']}")
        st.button("Retry answer", on_click=queue_question, args=(error["question"],))
    elif st.session_state.messages and "last_reply" in st.session_state:
        reply = st.session_state.last_reply
        st.caption(f"Prompt: {reply.input_tokens} tokens | Answer: {reply.output_tokens} tokens")
        if reply.removed_turns:
            st.caption(
                f"{reply.removed_turns} older conversation turns were omitted "
                "from the model context. "
                "The full history remains visible above."
            )
        if reply.reached_limit:
            st.info("The answer reached the token limit. Increase the limit or ask to continue.")


# Assemble the interface and keep history within the current browser session.
def main() -> None:
    st.set_page_config(page_title="Indonesian Chatbot", layout="centered")
    st.title("Indonesian Chatbot")
    st.caption("A simple conversation with your fine-tuned Qwen2.5 model through Ollama.")

    # Stop early when environment settings cannot be used by the interface.
    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        st.stop()

    initialize_session()
    busy = st.session_state.pending_question is not None
    settings = render_sidebar(settings, busy)
    resources = ensure_model(settings)

    # Replay completed messages before accepting or processing the next question.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    render_feedback()

    # Keep the configured suggestions and chat input visible while the page scrolls.
    with st.bottom:
        render_suggestions(settings, busy)
        question = st.chat_input("Type your message...", disabled=busy)
    if question and question.strip():
        queue_question(question)
        st.rerun()

    process_question(resources, settings)


if __name__ == "__main__":
    main()
