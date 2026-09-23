# Alibaba Cloud Model Studio (Bailian) Setup

The application uses Bailian through its OpenAI-compatible Chat Completions
endpoint. The API key is loaded at runtime and must never be committed.

## 1. Protect the account

Rotate any key that has been pasted into chat, a document, or a terminal log.
In the Bailian model-usage page, enable free-tier exhaustion protection for the
models used by this demo.

## 2. Configure the local file

The repository contains a Git-ignored `.env` file. Open it locally and set:

```dotenv
LLM_API_KEY=your-new-key
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-vl-plus
LLM_FALLBACK_MODELS=qwen-vl-max,qwen3.5-omni-plus
LLM_RESPONSE_FORMAT=json_object
LLM_TIMEOUT_SECONDS=90
```

Use the exact Base URL shown by the API example in the Bailian workspace. A
workspace-specific URL can replace the generic DashScope URL above.

`LLM_MODEL` is always attempted first. `LLM_FALLBACK_MODELS` is an ordered,
comma-separated allowlist. The client moves to the next model only for a
connection failure, rate limit, provider 5xx failure, unreadable response, or
an explicit model quota error. Authentication failures do not trigger model
switching.

For multimodal Qwen models, `json_object` is the compatibility default. The
returned object is still strictly validated by the local Pydantic schema. Use
`json_schema` only for a model documented to support it, or `auto` for provider
detection and a schema-to-object compatibility retry.

## 3. Install and validate

Run these commands in PowerShell from the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts\smoke_bailian.py
```

The last command validates configuration without sending a request. It prints
only whether the key is present, never the key value.

## 4. Send one controlled request

```powershell
python scripts\smoke_bailian.py --live
```

This sends one request using the controlled `C151` image fixture and prints the
decision plus provider, selected model, attempted models, token counts, request
ID, response format, and fallback reason. It never prints the API key.

## 5. Run the app

```powershell
streamlit run app.py
```

The same `.env` settings appear in the sidebar. A key entered into the password
field remains in the Streamlit process/session and is not written by the app.

Docker Compose also reads the same `.env` file:

```powershell
docker compose up --build
```

Environment values can be inspected by administrators of the local machine or
container host. Keeping a secret out of Git does not make it invisible to a
local administrator.

