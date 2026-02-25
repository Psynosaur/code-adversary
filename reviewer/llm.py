"""Bedrock inference client for adversarial code review."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from reviewer.config import (
    BEDROCK_PROFILE,
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    BEDROCK_MAX_TOKENS,
    USAGE_LOG_PATH,
)

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (chars_so_far, elapsed_seconds, message) -> None
ProgressCallback = Callable[[int, float, str], None]

# Module-level client — created once, reused across calls
_client = None

# ── Usage log setup ──────────────────────────────────────────────────────────

_usage_logger = None


def _get_usage_logger() -> logging.Logger:
    """Lazy-init a dedicated file logger for token usage."""
    global _usage_logger
    if _usage_logger is not None:
        return _usage_logger

    _usage_logger = logging.getLogger("reviewer.usage")
    _usage_logger.setLevel(logging.INFO)
    _usage_logger.propagate = False  # Don't duplicate to root logger

    # Ensure parent directory exists
    log_path = Path(USAGE_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file needs header BEFORE creating the handler (which creates the file)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0

    # Avoid duplicate handlers on the same logger name
    if not _usage_logger.handlers:
        handler = logging.FileHandler(str(log_path), mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _usage_logger.addHandler(handler)

    if needs_header:
        _usage_logger.info(
            "timestamp\tmodel\ttool\tinput_tokens\toutput_tokens\ttotal_tokens\tlatency_ms"
        )

    return _usage_logger


def _log_usage(
    tool: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
) -> None:
    """Log token usage to both the usage log file and the standard logger."""
    total = input_tokens + output_tokens

    # Standard logger (shows in server stderr)
    logger.info(
        "Bedrock usage [%s]: input=%d output=%d total=%d latency=%dms model=%s",
        tool,
        input_tokens,
        output_tokens,
        total,
        latency_ms,
        BEDROCK_MODEL_ID,
    )

    # Dedicated usage log file (TSV for easy parsing)
    usage = _get_usage_logger()
    ts = datetime.now(timezone.utc).isoformat()
    usage.info(
        "%s\t%s\t%s\t%d\t%d\t%d\t%d",
        ts,
        BEDROCK_MODEL_ID,
        tool,
        input_tokens,
        output_tokens,
        total,
        latency_ms,
    )


# ── Bedrock client ───────────────────────────────────────────────────────────


def _get_client():
    """Lazy-init the Bedrock Runtime client using the configured AWS profile."""
    global _client
    if _client is None:
        session = boto3.Session(
            profile_name=BEDROCK_PROFILE,
            region_name=BEDROCK_REGION,
        )
        _client = session.client(
            "bedrock-runtime",
            config=BotoConfig(
                retries={"max_attempts": 2, "mode": "adaptive"},
                read_timeout=120,
                connect_timeout=10,
                max_pool_connections=4,
                tcp_keepalive=True,
            ),
        )
        logger.info(
            "Bedrock client initialized: profile=%s region=%s model=%s",
            BEDROCK_PROFILE,
            BEDROCK_REGION,
            BEDROCK_MODEL_ID,
        )
    return _client


def invoke(
    system_prompt: str,
    user_message: str,
    tool: str = "unknown",
    max_tokens: int = BEDROCK_MAX_TOKENS,
    temperature: float = 0.3,
    on_progress: ProgressCallback | None = None,
) -> tuple[str, str]:
    """
    Send a streaming inference request to Bedrock and return the text response.

    Streams tokens from Bedrock, logging progress to stderr so the server
    operator can see the review forming in real time.

    Args:
        system_prompt: The system/persona prompt
        user_message: The user message (diff, code, decision, etc.)
        tool: Name of the calling tool (for usage logging)
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (lower = more focused)
        on_progress: Optional callback called during streaming with
                     (chars_so_far, elapsed_seconds, message)

    Returns:
        Tuple of (text_response, stop_reason). stop_reason is 'end_turn',
        'max_tokens', or 'stop_sequence'.

    Raises:
        RuntimeError: If the Bedrock call fails
    """
    client = _get_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    start = time.monotonic()
    logger.info("Bedrock stream starting [%s] model=%s", tool, BEDROCK_MODEL_ID)

    # Declare outside try so partial results are accessible in except
    text_chunks: list[str] = []
    input_tokens = 0
    output_tokens = 0

    try:
        response = client.invoke_model_with_response_stream(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        # Accumulate streamed response
        stop_reason = "unknown"
        chunk_count = 0
        total_chars = 0  # Running counter -- avoids O(n^2) recounting

        for event in response["body"]:
            # Guard against non-chunk events (error events, etc.)
            if "chunk" not in event:
                # Bedrock error events have keys like internalServerException,
                # modelStreamErrorException, throttlingException
                for key in (
                    "internalServerException",
                    "modelStreamErrorException",
                    "throttlingException",
                    "validationException",
                ):
                    if key in event:
                        err_msg = event[key].get("message", str(event[key]))
                        logger.error(
                            "Bedrock stream error [%s]: %s: %s", tool, key, err_msg
                        )
                        raise RuntimeError(f"Bedrock stream error ({key}): {err_msg}")
                logger.warning(
                    "Unknown non-chunk event in stream: %s", list(event.keys())
                )
                continue

            # Parse chunk -- guard against malformed JSON
            try:
                chunk = json.loads(event["chunk"]["bytes"])
            except (json.JSONDecodeError, KeyError) as parse_err:
                logger.warning("Malformed stream chunk, skipping: %s", parse_err)
                continue

            chunk_type = chunk.get("type", "")

            if chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    text_chunks.append(text)
                    total_chars += len(text)
                    chunk_count += 1

                    # Log and report progress every 20 chunks
                    if chunk_count % 20 == 0:
                        elapsed = time.monotonic() - start
                        msg = f"streaming {total_chars} chars, {elapsed:.0f}s"
                        logger.info("  [%s] %s", tool, msg)

                        if on_progress:
                            try:
                                on_progress(total_chars, elapsed, msg)
                            except Exception as cb_err:
                                logger.warning(
                                    "on_progress callback raised: %s", cb_err
                                )

            elif chunk_type == "message_delta":
                stop_reason = chunk.get("delta", {}).get("stop_reason", "unknown")
                output_tokens = chunk.get("usage", {}).get("output_tokens", 0)

            elif chunk_type == "message_start":
                input_tokens = (
                    chunk.get("message", {}).get("usage", {}).get("input_tokens", 0)
                )

        full_text = "".join(text_chunks)
        latency_ms = int((time.monotonic() - start) * 1000)

        # Final progress report so short responses get at least one callback
        if on_progress and total_chars > 0:
            try:
                elapsed = time.monotonic() - start
                on_progress(
                    total_chars,
                    elapsed,
                    f"complete {total_chars} chars, {elapsed:.0f}s",
                )
            except Exception as cb_err:
                logger.warning(
                    "on_progress callback raised on final report: %s", cb_err
                )

        if stop_reason == "unknown" and full_text:
            logger.warning(
                "Stream ended without message_delta for tool=%s. "
                "stop_reason is unknown -- stream may have been interrupted.",
                tool,
            )

        _log_usage(
            tool=tool,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

        if stop_reason == "max_tokens":
            logger.warning(
                "Response truncated (hit max_tokens=%d) for tool=%s. "
                "Output may be incomplete.",
                max_tokens,
                tool,
            )

        if full_text:
            return full_text, stop_reason

        logger.warning("Empty response from Bedrock stream for tool=%s", tool)
        return "", stop_reason

    except RuntimeError:
        raise  # Re-raise our own errors (e.g., from stream error events)
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        # If we got partial text before the failure, return it with a warning
        partial = "".join(text_chunks) if text_chunks else ""
        if partial:
            logger.error(
                "Bedrock stream failed after %dms with %d chars received: %s",
                latency_ms,
                len(partial),
                e,
            )
            _log_usage(
                tool=tool,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            return partial, "stream_error"
        logger.error("Bedrock inference failed after %dms: %s", latency_ms, e)
        raise RuntimeError(f"Bedrock inference failed: {e}") from e
