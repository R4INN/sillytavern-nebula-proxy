#!/usr/bin/env python3
"""SillyTavern <-> Nebula Proxy (v2 — Direct Agent DM)

A lightweight reverse proxy that lets SillyTavern use Nebula as its AI backend.

Architecture:
    SillyTavern sends OpenAI-format chat completion requests to this proxy.
    The proxy opens a DM thread with a specific Nebula agent, sends the
    conversation as a single prompt, polls for the agent's response, and
    returns it in OpenAI-compatible format.

    Messages go directly to the sillytavern-roleplay agent — not to the
    main Nebula orchestrator. This keeps roleplay traffic isolated.

    No webhooks, no callbacks, no async task chains. Just direct API calls.

Auth:
    The Nebula API uses a Bearer JWT token for authentication. To get yours:
    1. Log into nebula.gg
    2. Open Chrome DevTools (F12) -> Network tab -> filter XHR
    3. Send any message and click one of the api.nebula.gg requests
    4. Copy the Authorization header value (after "Bearer ")

    NOTE: This JWT expires after ~30 days. When it stops working, repeat the
    steps above to get a fresh token.

Setup:
    1. pip install flask requests python-dotenv
    2. Create a .env file with:
         NEBULA_BEARER_TOKEN=eyJhb...<your JWT from Chrome DevTools>
         NEBULA_AGENT_ID=agt_069ae0c0325978858000e7d919400eff
         PROXY_PORT=5001
    3. python nebula_proxy.py
    4. In SillyTavern, set the API to "Chat Completion (OpenAI)"
       and point it at http://localhost:5001
"""

import json
import logging
import os
import time
import uuid

import requests as http_requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

load_dotenv()

# ---- Configuration ---------------------------------------------------------

NEBULA_API_BASE = os.getenv("NEBULA_API_BASE", "https://api.nebula.gg")
NEBULA_BEARER_TOKEN = os.getenv("NEBULA_BEARER_TOKEN", "")
NEBULA_AGENT_ID = os.getenv(
    "NEBULA_AGENT_ID", "agt_069ae0c0325978858000e7d919400eff"
)
PROXY_PORT = int(os.getenv("PROXY_PORT", "5001"))

# Polling settings
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.0"))  # seconds between polls
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "300"))    # max wait time

# ---- Logging ----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nebula-proxy")

# ---- Flask App --------------------------------------------------------------

app = Flask(__name__)

# ---- State ------------------------------------------------------------------

_dm_thread_id: str | None = None  # cached DM thread ID


# ---- API Helpers ------------------------------------------------------------

def _headers() -> dict:
    """Standard headers for Nebula API requests."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NEBULA_BEARER_TOKEN}",
        "x-user-timezone": "America/New_York",
    }


def _get_dm_thread() -> str:
    """Get or create a DM thread with the roleplay agent.

    Uses GET /agents/{agent_id}/dm which returns (or creates) a thread
    where is_agent_dm=true and target_agent_id points to our agent.
    The result is cached for the lifetime of the process.
    """
    global _dm_thread_id
    if _dm_thread_id:
        return _dm_thread_id

    logger.info("Getting DM thread with agent %s...", NEBULA_AGENT_ID)
    resp = http_requests.get(
        f"{NEBULA_API_BASE}/agents/{NEBULA_AGENT_ID}/dm",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # The response is {"result": {"id": "thrd_...", ...}}
    result = data.get("result", data)
    thread_id = result.get("id") or result.get("thread_id")

    if not thread_id:
        raise ValueError(
            f"Could not extract thread ID from DM response: "
            f"{json.dumps(data)[:500]}"
        )

    _dm_thread_id = thread_id
    logger.info(
        "DM thread ready: %s (is_agent_dm=%s)",
        thread_id,
        result.get("is_agent_dm"),
    )
    return thread_id


# ---- Message Formatting -----------------------------------------------------

def _format_messages_for_nebula(messages: list) -> str:
    """Convert OpenAI-format messages array into a single prompt for Nebula.

    SillyTavern sends the full conversation as an array of messages with roles
    (system, user, assistant). We concatenate them into a structured prompt
    that the Nebula agent can understand and roleplay from.
    """
    parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not content or not content.strip():
            continue

        if role == "system":
            parts.append(f"[System Instructions]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(f"[{role}]\n{content}")

    return "\n\n---\n\n".join(parts)


# ---- Nebula API Interaction -------------------------------------------------

def _get_existing_message_ids(thread_id: str, limit: int = 10) -> set:
    """Fetch recent message IDs from the thread (for change detection)."""
    try:
        resp = http_requests.get(
            f"{NEBULA_API_BASE}/threads/{thread_id}/messages",
            headers=_headers(),
            params={"limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        messages = []
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict):
            messages = data.get("messages", data.get("items", data.get("data", [])))

        ids = set()
        for msg in messages:
            msg_id = msg.get("id") or msg.get("message_id")
            if msg_id:
                ids.add(msg_id)
        return ids

    except Exception as e:
        logger.warning("Failed to fetch existing messages: %s", e)
        return set()


def _send_message(thread_id: str, message: str) -> dict:
    """Send a message to a Nebula thread. Returns the full API response."""
    logger.info(
        "Sending message to thread %s (%d chars)...", thread_id, len(message)
    )

    resp = http_requests.post(
        f"{NEBULA_API_BASE}/threads/{thread_id}/messages",
        headers=_headers(),
        json={
            "message": message,
            "plan_mode": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    logger.info(
        "Message sent. Response keys: %s",
        list(data.keys()) if isinstance(data, dict) else type(data).__name__,
    )
    return data


def _poll_for_response(thread_id: str, known_ids: set) -> str:
    """Poll the thread for a new assistant message.

    Compares message IDs against the set taken before we sent our message
    to detect the new assistant reply.
    """
    logger.info("Polling for response (timeout: %ss)...", POLL_TIMEOUT)
    start_time = time.time()
    poll_count = 0

    while time.time() - start_time < POLL_TIMEOUT:
        poll_count += 1
        time.sleep(POLL_INTERVAL)

        try:
            resp = http_requests.get(
                f"{NEBULA_API_BASE}/threads/{thread_id}/messages",
                headers=_headers(),
                params={"limit": 10},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            messages = []
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict):
                messages = data.get(
                    "messages",
                    data.get("items", data.get("data", [])),
                )

            # Look for new assistant messages not in our known set
            for msg in messages:
                msg_role = msg.get("role", "")
                msg_content = msg.get("content", "")
                msg_id = msg.get("id") or msg.get("message_id", "")

                if (
                    msg_role == "assistant"
                    and msg_content
                    and msg_content.strip()
                    and msg_id
                    and msg_id not in known_ids
                ):
                    elapsed = time.time() - start_time
                    logger.info(
                        "Got response after %d polls (%.1fs): %d chars",
                        poll_count,
                        elapsed,
                        len(msg_content),
                    )
                    return msg_content

        except Exception as e:
            logger.warning("Poll %d failed: %s", poll_count, e)

        if poll_count % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Still waiting... (%d polls, %.0fs elapsed)",
                poll_count,
                elapsed,
            )

    raise TimeoutError(
        f"No response after {POLL_TIMEOUT}s ({poll_count} polls)"
    )


# ---- OpenAI-Compatible Response Formatting ----------------------------------


def _make_openai_response(content: str, model: str = "nebula") -> dict:
    """Format the response in OpenAI chat completion format."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _make_openai_error(message: str, status_code: int = 500) -> tuple:
    """Format an error in OpenAI API error format."""
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": "server_error",
                    "code": status_code,
                }
            }
        ),
        status_code,
    )


# ---- Routes -----------------------------------------------------------------


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        body = request.get_json(force=True)
    except Exception:
        return _make_openai_error("Invalid JSON body", 400)

    messages = body.get("messages", [])
    model = body.get("model", "nebula")
    stream = body.get("stream", False)

    if not messages:
        return _make_openai_error("No messages provided", 400)

    logger.info(
        "Received request: %d messages, model=%s, stream=%s",
        len(messages),
        model,
        stream,
    )

    # -- Step 1: Ensure we have the DM thread ---------------------------------
    try:
        thread_id = _get_dm_thread()
    except Exception as e:
        logger.error("Failed to get DM thread: %s", e)
        return _make_openai_error(f"Failed to connect to Nebula: {e}", 502)

    # -- Step 2: Snapshot current message IDs ---------------------------------
    known_ids = _get_existing_message_ids(thread_id)
    logger.info("Thread has %d existing messages", len(known_ids))

    # -- Step 3: Format and send the message ----------------------------------
    prompt = _format_messages_for_nebula(messages)

    try:
        _send_message(thread_id, prompt)
    except Exception as e:
        logger.error("Failed to send message: %s", e)
        return _make_openai_error(
            f"Failed to send message to Nebula: {e}", 502
        )

    # -- Step 4: Poll for the response ----------------------------------------
    try:
        response_content = _poll_for_response(thread_id, known_ids)
    except TimeoutError as e:
        logger.error("Response timeout: %s", e)
        return _make_openai_error("Nebula did not respond in time", 504)
    except Exception as e:
        logger.error("Polling error: %s", e)
        return _make_openai_error(
            f"Error waiting for Nebula response: {e}", 502
        )

    # -- Step 5: Return in OpenAI format --------------------------------------
    if stream:

        def generate_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            # Send the full content in one chunk
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": response_content,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # Stop chunk
            stop_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            generate_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = _make_openai_response(response_content, model)
    logger.info("Returning response: %d chars", len(response_content))
    return jsonify(result)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Return available models (SillyTavern checks this endpoint)."""
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": "nebula",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nebula",
                }
            ],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "version": "2.0.0",
            "architecture": "agent-dm",
            "agent_id": NEBULA_AGENT_ID,
            "dm_thread_id": _dm_thread_id,
        }
    )


# ---- Startup ----------------------------------------------------------------

if __name__ == "__main__":
    if not NEBULA_BEARER_TOKEN:
        logger.error(
            "NEBULA_BEARER_TOKEN is not set! Add it to your .env file."
        )
        logger.error("")
        logger.error("To get your token:")
        logger.error("  1. Log into nebula.gg")
        logger.error(
            "  2. Open Chrome DevTools (F12) -> Network tab -> XHR filter"
        )
        logger.error(
            "  3. Send any message, click an api.nebula.gg request"
        )
        logger.error(
            "  4. Copy the Authorization header value (after 'Bearer ')"
        )
        logger.error("")
        logger.error("NOTE: The token expires after ~30 days.")
        exit(1)

    logger.info("=" * 60)
    logger.info("Nebula SillyTavern Proxy v2.0 (Agent DM)")
    logger.info("=" * 60)
    logger.info("API Base: %s", NEBULA_API_BASE)
    logger.info("Agent ID: %s", NEBULA_AGENT_ID)
    logger.info(
        "Poll interval: %ss, timeout: %ss", POLL_INTERVAL, POLL_TIMEOUT
    )
    logger.info("Listening on port %d", PROXY_PORT)
    logger.info("")
    logger.info(
        "Point SillyTavern at: http://localhost:%d", PROXY_PORT
    )
    logger.info("=" * 60)

    # Pre-fetch the DM thread on startup so errors are caught early
    try:
        thread = _get_dm_thread()
        logger.info("Ready! DM thread: %s", thread)
    except Exception as e:
        logger.error("Failed to initialize DM thread: %s", e)
        logger.error(
            "Check your NEBULA_BEARER_TOKEN and NEBULA_AGENT_ID."
        )
        exit(1)

    app.run(host="0.0.0.0", port=PROXY_PORT, debug=False)
