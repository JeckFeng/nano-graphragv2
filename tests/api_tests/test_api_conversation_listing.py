"""API test script for listing conversations and messages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.parse import urljoin

try:
    import httpx
except ImportError as exc:
    raise SystemExit("Missing dependency httpx. Install with: pip install httpx") from exc


DEFAULT_BASE_URL = os.environ.get("NANO_GRAPHRAG_API_URL", "http://localhost:8001")
DEFAULT_USER_ID = os.environ.get("NANO_GRAPHRAG_USER_ID", "api-test-user")
DEFAULT_TIMEOUT_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="List conversations and messages API test")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="External user id used by the API",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Thread id to fetch messages; if omitted, uses the first conversation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Conversation list limit",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Conversation list offset",
    )
    parser.add_argument(
        "--message-limit",
        type=int,
        default=50,
        help="Message list limit",
    )
    parser.add_argument(
        "--message-offset",
        type=int,
        default=0,
        help="Message list offset",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP request timeout in seconds",
    )
    return parser.parse_args()


def build_url(base_url: str, path: str) -> str:
    """Build a full URL for the API endpoint.

    Args:
        base_url: API base URL.
        path: Endpoint path.

    Returns:
        str: Full URL.
    """
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def request_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a GET request and parse JSON response.

    Args:
        client: httpx client.
        url: Request URL.
        params: Query parameters.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        RuntimeError: When response status is not 200.
    """
    response = client.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(
            f"Request failed: status={response.status_code}, url={url}, body={response.text}"
        )
    return response.json()


def list_conversations(
    client: httpx.Client,
    base_url: str,
    user_id: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Call the list conversations API.

    Args:
        client: httpx client.
        base_url: API base URL.
        user_id: External user id.
        limit: Max number of conversations.
        offset: Pagination offset.

    Returns:
        dict[str, Any]: Conversation list payload.
    """
    url = build_url(base_url, "v1/conversations")
    params = {"user_id": user_id, "limit": limit, "offset": offset}
    return request_json(client, url, params)


def list_messages(
    client: httpx.Client,
    base_url: str,
    user_id: str,
    thread_id: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Call the list messages API.

    Args:
        client: httpx client.
        base_url: API base URL.
        user_id: External user id.
        thread_id: Conversation thread id.
        limit: Max number of messages.
        offset: Pagination offset.

    Returns:
        dict[str, Any]: Message list payload.
    """
    url = build_url(base_url, f"v1/conversations/{thread_id}/messages")
    params = {"user_id": user_id, "limit": limit, "offset": offset}
    return request_json(client, url, params)


def select_thread_id(payload: dict[str, Any], fallback: str | None) -> str:
    """Select a thread id from payload or fallback.

    Args:
        payload: Conversation list payload.
        fallback: Thread id provided by user.

    Returns:
        str: Selected thread id.

    Raises:
        SystemExit: If no thread id is available.
    """
    if fallback:
        return fallback
    conversations = payload.get("conversations") or []
    if not conversations:
        raise SystemExit("No conversations found; provide --thread-id or create one first.")
    return str(conversations[0].get("thread_id"))


def print_payload(title: str, payload: dict[str, Any]) -> None:
    """Print a JSON payload with a title.

    Args:
        title: Section title.
        payload: JSON payload to print.
    """
    print(f"\n{title}")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(timeout=timeout) as client:
        conversations = list_conversations(
            client,
            args.base_url,
            args.user_id,
            args.limit,
            args.offset,
        )
        print_payload("Conversations", conversations)

        thread_id = select_thread_id(conversations, args.thread_id)
        messages = list_messages(
            client,
            args.base_url,
            args.user_id,
            thread_id,
            args.message_limit,
            args.message_offset,
        )
        print_payload(f"Messages (thread_id={thread_id})", messages)


if __name__ == "__main__":
    sys.exit(main())
