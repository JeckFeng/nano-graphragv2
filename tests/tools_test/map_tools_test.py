"""Gaode driving route tool test script.

Purpose:
- Verify that the Gaode driving route tool works with a real API call.
- Persist the raw API response to tests/tools_test/test_map_tools_result.json.

Constraints:
- No LLM usage.
- Build the request payload that matches Gaode API requirements.
- Print the raw API response in full.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Tools.gaode_map_tool import gaode_driving_route
    from core.tool_errors import ToolError
except ModuleNotFoundError as exc:
    missing_module = exc.name or "unknown"
    if missing_module in {"aiohttp", "dotenv"}:
        raise SystemExit(
            f"Missing dependency '{missing_module}'. Install it before running the test."
        ) from exc
    raise


DEFAULT_ORIGIN = "116.3907203448,39.916580438797"
DEFAULT_DESTINATION = "116.0107203448,38.110580438797"
RESULT_PATH = PROJECT_ROOT / "tests" / "tools_test" / "test_map_tools_result.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the test script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Gaode driving route tool test")
    parser.add_argument(
        "--origin",
        default=DEFAULT_ORIGIN,
        help="Origin coordinate in 'lng,lat' format",
    )
    parser.add_argument(
        "--destination",
        default=DEFAULT_DESTINATION,
        help="Destination coordinate in 'lng,lat' format",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gaode API key; if omitted, GAODE_API_KEY env var is used",
    )
    return parser.parse_args()


def build_request_payload(origin: str, destination: str, api_key: str) -> Dict[str, str]:
    """Build the request payload matching Gaode API requirements.

    Args:
        origin: Origin coordinate string in "lng,lat" format.
        destination: Destination coordinate string in "lng,lat" format.
        api_key: Gaode API key.

    Returns:
        Dict[str, str]: Payload dictionary for the tool call.
    """
    return {
        "origin": origin,
        "destination": destination,
        "api_key": api_key,
    }


async def run_test(origin: str, destination: str, api_key: str) -> Dict[str, Any]:
    """Call the Gaode driving route tool and return raw API response.

    Args:
        origin: Origin coordinate string in "lng,lat" format.
        destination: Destination coordinate string in "lng,lat" format.
        api_key: Gaode API key.

    Returns:
        Dict[str, Any]: Raw Gaode API response JSON.

    Raises:
        RuntimeError: If the API response has no route data.
        ToolError: If the tool encounters a handled failure.
    """
    payload = build_request_payload(origin, destination, api_key)
    result = await gaode_driving_route(**payload)
    route_data = result.get("route", {})
    if not isinstance(route_data, dict) or not route_data:
        raise RuntimeError("Gaode API returned empty route data")
    return route_data


def main() -> int:
    """Run the Gaode map tool test and print the raw API response.

    Returns:
        int: Process exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    api_key = args.api_key or os.getenv("GAODE_API_KEY")
    if not api_key:
        print("Missing GAODE_API_KEY env var or --api-key argument")
        return 1

    try:
        route_data = asyncio.run(run_test(args.origin, args.destination, api_key))
    except ToolError as exc:
        print(f"ToolError: {exc}")
        if exc.details:
            print(json.dumps(exc.details, ensure_ascii=False, indent=2))
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(route_data, file, ensure_ascii=False, indent=2)

    print(json.dumps(route_data, ensure_ascii=False, indent=2))
    print(f"Raw Gaode API response saved to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
