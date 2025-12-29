import argparse
import asyncio
from pathlib import Path

import pandas as pd

from graphrag.api.query import global_search, local_search
from graphrag.config.load_config import load_config


def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Load a parquet file if it exists, otherwise return None."""
    return pd.read_parquet(path) if path.exists() else None


async def run_global(config: Path, root: Path, query: str) -> None:
    cfg = load_config(root, config)
    entities = pd.read_parquet(root / "output" / "entities.parquet")
    communities = pd.read_parquet(root / "output" / "communities.parquet")
    community_reports = pd.read_parquet(root / "output" / "community_reports.parquet")
    resp, _ = await global_search(
        config=cfg,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=None,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
    )
    print(resp)


async def run_local(config: Path, root: Path, query: str, community_level: int) -> None:
    cfg = load_config(root, config)
    entities = pd.read_parquet(root / "output" / "entities.parquet")
    communities = pd.read_parquet(root / "output" / "communities.parquet")
    community_reports = pd.read_parquet(root / "output" / "community_reports.parquet")
    text_units = pd.read_parquet(root / "output" / "text_units.parquet")
    relationships = pd.read_parquet(root / "output" / "relationships.parquet")
    covariates = load_parquet_safe(root / "output" / "covariates.parquet")

    resp, _ = await local_search(
        config=cfg,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        text_units=text_units,
        relationships=relationships,
        covariates=covariates,
        community_level=community_level,
        response_type="Multiple Paragraphs",
        query=query,
    )
    print(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG queries programmatically")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "christmas",
        help="Root directory containing settings.yaml and output parquet files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to settings.yaml (default: <root>/settings.yaml)",
    )
    parser.add_argument(
        "--method",
        choices=["global", "local"],
        default="global",
        help="Query method to execute.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query text.",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="Community level for local search (ignored for global).",
    )
    args = parser.parse_args()

    root = args.root
    config = args.config or root / "settings.yaml"

    if args.method == "global":
        asyncio.run(run_global(config, root, args.query))
    else:
        asyncio.run(run_local(config, root, args.query, args.community_level))


if __name__ == "__main__":
    main()
