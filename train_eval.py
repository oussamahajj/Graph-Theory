"""Entry point for quick Cora loading and sanity checks."""

from __future__ import annotations

from pathlib import Path

from utils import describe_cora, load_cora


def main() -> None:
    data_dir = Path("data")
    content = data_dir / "cora.content"
    cites = data_dir / "cora.cites"

    features, labels, edges = load_cora(content, cites)
    stats = describe_cora(features, labels, edges)

    print("=== Cora dataset loaded ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nTop labels:")
    print(labels.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
