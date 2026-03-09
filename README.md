# SillyTavern Nebula Proxy

A lightweight reverse proxy that lets SillyTavern use Nebula as its AI backend for roleplay and creative fiction.

## Architecture

SillyTavern sends OpenAI-format chat completion requests to this proxy. The proxy opens a DM thread with a specific Nebula agent, sends the conversation as a single prompt, polls for the agent's response, and returns it in OpenAI-compatible format.

## Setup

1. Clone this repo:
```bash
git clone https://github.com/R4INN/sillytavern-nebula-proxy.git
cd sillytavern-nebula-proxy
```

2. Install dependencies:
```bash
pip install flask requests python-dotenv
```

3. Create a `.env` file:
```
NEBULA_BEARER_TOKEN=<your Nebula JWT token>
NEBULA_AGENT_ID=<your Nebula agent ID>
PROXY_PORT=5001
```

**Finding your Bearer token:** Open Chrome DevTools (F12) on nebula.gg while logged in, go to the Network tab, filter by Fetch/XHR, and copy the token from any `Authorization: Bearer <token>` header on requests to `api.nebula.gg`.

4. Run:
```bash
python nebula_proxy.py
```

5. In SillyTavern, add a Chat Completion API (OpenAI-compatible) pointing to `http://localhost:5001`.

## Token Expiry

The Nebula JWT expires approximately every 30 days. When the proxy starts returning auth errors, grab a fresh token from DevTools.

## How It Works

- Converts SillyTavern's multi-message format into a single formatted prompt
- Creates a dedicated agent DM thread on first run
- Polls Nebula's message API for the assistant's response
- Returns responses in OpenAI chat completion format
- Supports both streaming and non-streaming modes
