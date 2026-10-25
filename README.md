# Indonesian Chatbot

A simple Streamlit chatbot built around the fine-tuned [Qwen2.5 1.5B Indonesian model](https://huggingface.co/Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft). The original model was converted to GGUF for local inference through Ollama.

## Model origin and conversion

1. **Original fine-tuned model:** [Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft](https://huggingface.co/Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft). The training notebook exports the fine-tuned model as merged 16-bit weights in Safetensors format.
2. **GGUF conversion:** The original repository was converted using [GGUF My Repo by ggml-org](https://huggingface.co/spaces/ggml-org/gguf-my-repo), with Q4_K_M quantization for the converted model.
3. **Converted model:** [Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft-Q4_K_M-GGUF](https://huggingface.co/Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft-Q4_K_M-GGUF). This is the model installed in Ollama and used by the chatbot.

The app sends conversations to Ollama's `POST /api/chat` endpoint. Ollama loads the converted GGUF model and generates the answers.

## Features

- Conversation history per browser session, with follow-up questions.
- An English system prompt that requests Indonesian answers unless another language is requested.
- Clear conversation, retry failed answers, and adjust response length and temperature.
- Two suggested question bubbles pinned above the chat input while scrolling.
- Clear feedback for connection failures, missing models, timeouts, and long questions.

## Requirements

- Python 3.11 or newer, with pip.
- Ollama installed and running on the same computer, or an accessible Ollama server.
- The GGUF model already available in Ollama.

The application only needs Streamlit and Requests. PyTorch, Transformers, and training packages are not required.
Streamlit 1.66 or newer is required for the pinned chat controls.

## Installation and startup

Run these commands from the project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. Keep Ollama running while using the chatbot.

Check the installed and currently loaded models:

```powershell
ollama list
ollama ps
```

The default model name matches the installed model:

```text
hf.co/Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft-Q4_K_M-GGUF:latest
```

An empty `ollama ps` is normal when the model is unloaded. Ollama loads it on the first chat request. If Ollama is not running, open Ollama or run `ollama serve`. If the model is missing, install it once:

```powershell
ollama pull hf.co/Lukas16/qwen2-5-1-5b-indonesian-chatbot-sft-Q4_K_M-GGUF:latest
```

## Usage

Type a message or click a suggestion. The answer appears after Ollama finishes generating it. Ask follow-up questions in the same session to use previous conversation context.

The suggestion bubbles stay above the chat input while you scroll. Set `SUGGESTION_COUNT` in `chatbot/config.py` to change their number; the current setting is `2`. Suggestions remain unchanged during the session unless their count or dataset changes, with no shuffle button. **Clear conversation** removes its history and errors. **Retry answer** resends a failed question without duplicating messages.

Answers allow up to 2048 tokens by default. Increase **Maximum response tokens** in the sidebar to allow up to 4096 tokens. Longer answers may take more time. Ask the chatbot to continue if an answer still reaches the limit.

The interface, warnings, code comments, and system prompt use English. Dataset suggestions remain in Indonesian.

## Configuration

Defaults work with the installed model at `http://localhost:11434`. Set environment variables only when needed; `.env` files are not loaded automatically.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL, without `/api/chat`. |
| `OLLAMA_MODEL` | Model name shown above | Exact model name from `ollama list`. |
| `OLLAMA_TIMEOUT` | `120` | Chat response timeout in seconds; 1-600. |
| `CONTEXT_TOKENS` | `8192` | Context size including the answer; 512-32768. |
| `MAX_NEW_TOKENS` | `2048` | Answer limit; 32-4096, at most half the context. |
| `MAX_HISTORY_TURNS` | `50` | Maximum previous user/assistant pairs sent; 0-50, subject to the context budget. |
| `TEMPERATURE` | `0.7` | Response variation; 0-2. |
| `TOP_P` | `0.9` | Sampling probability; greater than 0, at most 1. |
| `REPETITION_PENALTY` | `1.08` | Repetition penalty; 1-2. |
| `SYSTEM_PROMPT` | Defined in `chatbot/config.py` | Assistant instructions. |
| `DATASET_PATH` | `alpaca-gpt4-indonesian-cleaned.csv` | Suggestions CSV; relative paths start at the project root. |

For a different Ollama server or model name:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "your-installed-model:latest"
python -m streamlit run app.py
```

## Conversation behavior

Each request includes the system prompt, recent completed turns, and the latest question. Older complete turns are removed when the history or context limit is reached; the full session history stays visible. An oversized question is rejected with a clear message.

The system prompt treats history as essential context. When asked for a full recap, the assistant is instructed to list every earlier user question available in the supplied history in chronological order. Set **Temperature** to **0** for more predictable recaps. Completeness still depends on the model; a recap cannot include messages that have already been removed from the model context.

Context sizing uses a conservative estimate based on UTF-8 byte length plus message overhead, rather than an exact tokenizer count. The footer displays actual prompt and output token counts returned by Ollama. Ollama applies the model's chat template.

Only successful user/assistant pairs enter history. History lives in Streamlit session state and is not stored in a database or on disk; it can be lost when the session ends or the server restarts. Recall depends on the model and the context retained.

## Dataset source

The fine-tuning notebook uses [OpenPathAI/alpaca-gpt4-indonesian-cleaned](https://huggingface.co/datasets/OpenPathAI/alpaca-gpt4-indonesian-cleaned), an Indonesian dataset with `prompt` and `response` columns. The local `alpaca-gpt4-indonesian-cleaned.csv` contains these question-answer pairs.

The chatbot uses the local CSV's `prompt` column for suggestions, deduplicated and filtered to 12-180 characters. Answers are generated by the Ollama model. A missing or invalid CSV disables suggestions while typed chat stays available. No dataset is downloaded automatically.

## Project structure

```text
chatbot-scsg/                                   # Project root
├── .streamlit/                                 # Streamlit project settings
│   └── config.toml                             # Polling-based file watcher configuration
├── app.py                                      # Streamlit interface and session state
├── chatbot/                                    # Chatbot services
│   ├── config.py                               # Environment settings and system prompt
│   ├── dataset_utils.py                        # Local CSV filtering and suggestions
│   ├── inference.py                            # Conversation context and answer generation
│   └── model_loader.py                         # Ollama API client and model verification
├── alpaca-gpt4-indonesian-cleaned.csv          # Local question suggestion dataset
├── Fine_Tuning_Chatbot_Bahasa_Indonesia.ipynb  # Original fine-tuning reference notebook
├── requirements.txt                            # Streamlit and HTTP client dependencies
├── .gitignore                                  # Generated files excluded from Git
└── README.md                                   # Setup and usage instructions
```

The original notebook is retained as a training reference and has separate training dependencies. It is not executed by the chatbot.

API reference: [Ollama chat](https://docs.ollama.com/api/chat), [installed models](https://docs.ollama.com/api/tags), and [running models](https://docs.ollama.com/api/ps).
