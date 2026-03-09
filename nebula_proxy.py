#!/usr/bin/env python3
"""SillyTavern <-> Nebula Proxy (v4.1 — Fixed Event Parsing)

A lightweight reverse proxy that lets SillyTavern use Nebula as its AI backend.

Architecture:
    SillyTavern sends OpenAI-format chat completion requests to this proxy.
    For EACH request, the proxy:
      1. Creates a fresh DM thread with the Nebula roleplay agent
      2. Sends the full conversation as a single prompt
      3. Polls the /events endpoint for the completed assistant response
      4. Returns the response in OpenAI format
      5. Deletes the ephemeral thread (fire-and-forget cleanup)

    This means the Nebula agent is completely STATELESS between requests.
    It only sees what SillyTavern sends in the messages array each time,
    with zero Nebula-side context bleed between sessions.

SillyTavern Setup:
    1. In SillyTavern, go to API Connections
    2. Select "Chat Completion" API type, "OpenAI" source
    3. Set Custom Endpoint: http://localhost:5001/v1
    4. Any API key will work (the proxy ignores it)
    5. Model: "nebula" (or anything — it's ignored)

Environment Variables:
    NEBULA_BEARER_TOKEN  - JWT from browser cookies (required)
    NEBULA_AGENT_ID      - Target agent ID (default: sillytavern-roleplay agent)
    PROXY_PORT           - Port to listen on (default: 5001)
    POLL_INTERVAL        - Seconds between polls (default: 2.5)
    POLL_TIMEOUT         - Max seconds to wait for response (default: 300)
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
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.5"))  # seconds between polls
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "300"))     # max wait time

# ---- Logging ----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nebula-proxy")

# ---- Flask App --------------------------------------------------------------

app = Flask(__name__)


# ---- API Helpers ------------------------------------------------------------

def _headers() -> dict:
    """Standard headers for Nebula API requests."""
    return {
        "Authorization": f"Bearer {NEBULA_BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _create_fresh_thread() -> str:
    """Create a brand-new ephemeral DM thread with the roleplay agent.

    Calls POST /threads with target_agent_id to create a single-agent DM.
    Each SillyTavern request gets its own thread so the agent starts fresh
    with no prior Nebula-side context.

    Returns:
        The new thread ID string.
    """
    session_tag = uuid.uuid4().hex[:8]
    title = f"st-session-{session_tag}"

    logger.info("Creating fresh thread '%s' for agent %s...", title, NEBULA_AGENT_ID)

    resp = http_requests.post(
        f"{NEBULA_API_BASE}/threads",
        headers=_headers(),
        json={
            "title": title,
            "target_agent_id": NEBULA_AGENT_ID,
            "extended_thinking_enabled": False,
            "source": "api",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Response: {"result": {"id": "thrd_...", ...}}
    result = data.get("result", data)
    thread_id = result.get("id", "")

    if not thread_id:
        raise RuntimeError(f"No thread ID in create response: {data}")

    logger.info("Created ephemeral thread: %s ('%s')", thread_id, title)
    return thread_id


def _delete_thread(thread_id: str) -> None:
    """Delete an ephemeral thread after use (fire-and-forget cleanup).

    Calls DELETE /threads/{thread_id}. Errors are logged but never raised,
    so cleanup failures don't affect the response to SillyTavern.
    """
    try:
        logger.info("Cleaning up thread %s...", thread_id)
        resp = http_requests.delete(
            f"{NEBULA_API_BASE}/threads/{thread_id}",
            headers=_headers(),
            timeout=15,
        )
        if resp.ok:
            logger.info("Thread %s deleted.", thread_id)
        else:
            logger.warning(
                "Thread cleanup returned %d: %s",
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logger.warning("Thread cleanup failed (non-fatal): %s", e)


# ---- Message Formatting -----------------------------------------------------

def _format_messages_for_nebula(messages: list) -> str:
    """Convert OpenAI-format messages array into a single prompt for Nebula.

    SillyTavern sends the full conversation history including system prompts,
    character cards, and user/assistant turns. We concatenate them into one
    message since Nebula threads accept a single message string.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[System Instructions]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        else:
            parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)


def _extract_text_from_content(content) -> str:
    """Extract plain text from a NebulaMessageEvent content field.

    The content field can be:
    - A simple string (from /messages endpoint)
    - An array of ChatContentType objects (from /events endpoint)
      Each object may have {"type": "text", "text": "..."} or similar
    - None/empty
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # Try common content object shapes
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("value")
                    or ""
                )
                if text:
                    text_parts.append(str(text))
        return "\n".join(text_parts).strip()

    return str(content).strip() if content else ""


# ---- Nebula API Interaction -------------------------------------------------

def _send_message(thread_id: str, message: str) -> str:
    """Send a message to a Nebula thread. Returns the user_message_id."""
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

    # Response: {"result": {"thread_id": "...", "message_id": "...", "status": "processing"}}
    result = data.get("result", data)
    msg_id = result.get("message_id", "unknown")
    status = result.get("status", "unknown")

    logger.info(
        "Message sent. user_message_id: %s, status: %s",
        msg_id,
        status,
    )
    return msg_id


def _poll_for_response_events(thread_id: str, user_message_id: str) -> str:
    """Poll /events/{user_message_id} for the completed assistant response.

    Uses the per-turn events endpoint which returns only events for our
    specific message, avoiding any confusion with other events.

    NebulaMessageEvent schema (from API spec):
      - type: "NebulaMessageEvent"  (NOT "role" — there is no role field)
      - status: "streaming" | "completed" | "error" | "cancelled"
      - content: array of ChatContentType objects [{type: "text", text: "..."}]
      - is_thinking: bool (true for reasoning tokens — skip these)
      - is_sub_agent: bool (true for delegated agent messages — skip these)
      - tool_call_ids: list (non-empty means tool call, not final response)

    We also watch for FinalResultEvent (type: "FinalResultEvent") which
    signals the entire turn is done.
    """
    logger.info(
        "Polling events for thread %s, user_message_id=%s...",
        thread_id,
        user_message_id,
    )

    deadline = time.time() + POLL_TIMEOUT
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        time.sleep(POLL_INTERVAL)

        try:
            # Use the per-turn endpoint for targeted results
            resp = http_requests.get(
                f"{NEBULA_API_BASE}/threads/{thread_id}/events/{user_message_id}",
                headers=_headers(),
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning(
                    "Events poll #%d returned %d, retrying...",
                    poll_count,
                    resp.status_code,
                )
                continue

            data = resp.json()

            # Response shape: {"result": {"events": [...], "usage": {...}}}
            result = data.get("result", {})
            events = result.get("events", [])

            if poll_count <= 3 or poll_count % 10 == 0:
                logger.info(
                    "Events poll #%d: %d event(s) found", poll_count, len(events)
                )

            # Scan for FinalResultEvent to know the turn is done,
            # then extract the final NebulaMessageEvent content
            has_final = False
            best_text = ""

            for event in events:
                event_type = event.get("type", "")

                if event_type == "FinalResultEvent":
                    has_final = True
                    logger.info("Poll #%d: FinalResultEvent found — turn complete", poll_count)

                elif event_type == "NebulaMessageEvent":
                    status = event.get("status", "")
                    is_thinking = event.get("is_thinking", False)
                    is_sub_agent = event.get("is_sub_agent", False)
                    tool_call_ids = event.get("tool_call_ids", [])

                    # Skip thinking tokens, sub-agent messages, and tool calls
                    if is_thinking or is_sub_agent or tool_call_ids:
                        continue

                    if status == "completed":
                        content = event.get("content", "")
                        text = _extract_text_from_content(content)
                        if text:
                            best_text = text  # keep the last completed one

                    elif status == "streaming":
                        if poll_count <= 3 or poll_count % 10 == 0:
                            logger.info(
                                "Poll #%d: response still streaming...", poll_count
                            )

            # If we found the final event and have text, we're done
            if has_final and best_text:
                logger.info(
                    "Got completed response (%d chars) on poll #%d",
                    len(best_text),
                    poll_count,
                )
                return best_text

            # Even without FinalResultEvent, if we have completed text, return it
            # (FinalResultEvent might arrive slightly later)
            if best_text and poll_count >= 3:
                logger.info(
                    "Got completed response (%d chars) on poll #%d (no FinalResultEvent yet, proceeding anyway)",
                    len(best_text),
                    poll_count,
                )
                return best_text

        except http_requests.exceptions.RequestException as e:
            logger.warning("Events poll #%d failed: %s", poll_count, e)

    raise TimeoutError(
        f"No completed assistant response after {POLL_TIMEOUT}s "
        f"({poll_count} polls)"
    )


def _poll_for_response_messages(thread_id: str) -> str:
    """Fallback: poll /messages for the assistant response.

    Used if the per-turn events endpoint is unavailable. Since we use
    ephemeral threads, any assistant message in the thread is our response.
    """
    logger.info("Falling back to /messages polling for thread %s...", thread_id)

    deadline = time.time() + POLL_TIMEOUT
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        time.sleep(POLL_INTERVAL)

        try:
            resp = http_requests.get(
                f"{NEBULA_API_BASE}/threads/{thread_id}/messages",
                headers=_headers(),
                params={"limit": 10, "sort_order": "desc"},
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning(
                    "Messages poll #%d returned %d", poll_count, resp.status_code
                )
                continue

            data = resp.json()

            # Response shape: {"result": [...]}
            messages = data.get("result", [])
            if isinstance(messages, dict):
                messages = messages.get("messages", messages.get("items", []))

            logger.info(
                "Messages poll #%d: %d message(s)", poll_count, len(messages)
            )

            for msg in messages:
                role = msg.get("role", "")

                if role == "assistant":
                    content = msg.get("content", "")
                    text = _extract_text_from_content(content)

                    if text:
                        logger.info(
                            "Got response via /messages (%d chars) on poll #%d",
                            len(text),
                            poll_count,
                        )
                        return text

        except http_requests.exceptions.RequestException as e:
            logger.warning("Messages poll #%d failed: %s", poll_count, e)

    raise TimeoutError(
        f"No assistant response via /messages after {POLL_TIMEOUT}s "
        f"({poll_count} polls)"
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
                    "param": None,
                    "code": None,
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

    # -- Step 1: Create a fresh ephemeral thread -------------------------------
    thread_id = None
    try:
        thread_id = _create_fresh_thread()
    except Exception as e:
        logger.error("Failed to create thread: %s", e)
        return _make_openai_error(f"Failed to connect to Nebula: {e}", 502)

    # -- Step 2: Format and send the message ----------------------------------
    prompt = _format_messages_for_nebula(messages)

    try:
        user_message_id = _send_message(thread_id, prompt)
    except Exception as e:
        logger.error("Failed to send message: %s", e)
        _delete_thread(thread_id)  # clean up on failure too
        return _make_openai_error(
            f"Failed to send message to Nebula: {e}", 502
        )

    # -- Step 3: Poll for the response ----------------------------------------
    try:
        response_content = _poll_for_response_events(thread_id, user_message_id)
    except TimeoutError as e:
        logger.error("Response timeout: %s", e)
        _delete_thread(thread_id)
        return _make_openai_error("Nebula did not respond in time", 504)
    except Exception as e:
        logger.error("Polling error: %s", e)
        _delete_thread(thread_id)
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

            # Send the stop chunk
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

            # Clean up the thread after streaming is complete
            _delete_thread(thread_id)

        return Response(
            generate_stream(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: return full response, then clean up
    result = jsonify(_make_openai_response(response_content, model))

    # Clean up the ephemeral thread
    _delete_thread(thread_id)

    return result


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
                    "created": 1700000000,
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
            "version": "4.1-fixed-event-parsing",
            "agent_id": NEBULA_AGENT_ID,
            "mode": "stateless (fresh thread per request)",
        }
    )


# ---- Startup ----------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Nebula SillyTavern Proxy v4.1 (Fixed Event Parsing)")
    logger.info("=" * 60)
    logger.info("Agent ID   : %s", NEBULA_AGENT_ID)
    logger.info("API Base   : %s", NEBULA_API_BASE)
    logger.info("Port       : %d", PROXY_PORT)
    logger.info("Mode       : Stateless (fresh thread per request)")
    logger.info("Poll       : %.1fs interval, %.0fs timeout", POLL_INTERVAL, POLL_TIMEOUT)
    logger.info("Token      : %s...%s", NEBULA_BEARER_TOKEN[:20], NEBULA_BEARER_TOKEN[-10:])
    logger.info("=" * 60)

    if not NEBULA_BEARER_TOKEN:
        logger.error("NEBULA_BEARER_TOKEN is not set! Add it to .env")
        exit(1)

    logger.info("Ready — each request creates a fresh thread (no shared state)")
    logger.info("SillyTavern endpoint: http://localhost:%d/v1", PROXY_PORT)

    app.run(host="0.0.0.0", port=PROXY_PORT, debug=False)
