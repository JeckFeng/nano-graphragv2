"""API test script for approvals endpoints."""

from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Approvals API test script")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL, e.g. http://localhost:8001",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="External user id used by the API",
    )
    parser.add_argument(
        "--status",
        default="pending",
        help="Approval status filter",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Approval list limit",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Approval list offset",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve a specific approval id",
    )
    parser.add_argument(
        "--approval-id",
        default=None,
        help="Approval id to resolve",
    )
    parser.add_argument(
        "--decision",
        default="approve",
        help="Decision to submit (approve/reject/edit)",
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


def request_json(client: httpx.Client, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    """Send a request and parse JSON response.

    Args:
        client: httpx client.
        method: HTTP method name.
        url: Request URL.
        **kwargs: Additional request arguments.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        RuntimeError: When response status is not 200.
    """
    response = client.request(method, url, **kwargs)
    if response.status_code != 200:
        raise RuntimeError(
            f"Request failed: status={response.status_code}, url={url}, body={response.text}"
        )
    return response.json()


def list_approvals(
    client: httpx.Client,
    base_url: str,
    user_id: str,
    status: str,
    thread_id: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Call the list approvals API.

    Args:
        client: httpx client.
        base_url: API base URL.
        user_id: External user id.
        status: Approval status filter.
        thread_id: Optional thread id filter.
        limit: Max number of approvals.
        offset: Pagination offset.

    Returns:
        dict[str, Any]: Approval list payload.
    """
    url = build_url(base_url, "v1/approvals")
    params = {
        "user_id": user_id,
        "status": status,
        "limit": limit,
        "offset": offset,
    }
    if thread_id:
        params["thread_id"] = thread_id
    return request_json(client, "GET", url, params=params)


def resolve_approval(
    client: httpx.Client,
    base_url: str,
    user_id: str,
    approval_id: str,
    decision: str,
) -> dict[str, Any]:
    """Call the resolve approval API.

    Args:
        client: httpx client.
        base_url: API base URL.
        user_id: External user id.
        approval_id: Approval record id.
        decision: Decision string.

    Returns:
        dict[str, Any]: Resolution payload.
    """
    url = build_url(base_url, f"v1/approvals/{approval_id}")
    return request_json(
        client,
        "POST",
        url,
        params={"user_id": user_id},
        json={"decision": decision},
    )


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
        approvals = list_approvals(
            client,
            args.base_url,
            args.user_id,
            args.status,
            args.thread_id,
            args.limit,
            args.offset,
        )
        print_payload("Approvals", approvals)

        if args.resolve:
            if not args.approval_id:
                raise SystemExit("--resolve requires --approval-id")
            resolution = resolve_approval(
                client,
                args.base_url,
                args.user_id,
                args.approval_id,
                args.decision,
            )
            print_payload(f"Resolution (approval_id={args.approval_id})", resolution)


if __name__ == "__main__":
    main()
