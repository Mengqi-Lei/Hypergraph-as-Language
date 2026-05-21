import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperalign_bench.ogbn_arxiv_hg import BuildConfig, build_ogbn_arxiv_hg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Arxiv-HG for HyperAlign-Bench VC/nc and HEC/hecls tasks."
    )
    parser.add_argument("--raw-root", type=str, default="data/raw/ogbn_arxiv")
    parser.add_argument("--output-dir", type=str, default="dataset/arxiv_hg")
    parser.add_argument(
        "--titleabs-path",
        type=str,
        default=None,
        help="Path to titleabs.tsv used to recover title and abstract text",
    )
    parser.add_argument("--min-hyperedge-size", type=int, default=2)
    parser.add_argument("--max-hyperedge-size", type=int, default=64)
    parser.add_argument("--max-incident-hyperedges", type=int, default=8)
    parser.add_argument("--max-members-per-hyperedge", type=int, default=8)
    parser.add_argument("--max-child-hyperedges", type=int, default=1)
    parser.add_argument("--formal-hidt-depth", type=int, default=3)
    parser.add_argument("--overview-hops", type=int, default=2)
    parser.add_argument("--overview-order-buckets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        titleabs_path=args.titleabs_path,
        min_hyperedge_size=args.min_hyperedge_size,
        max_hyperedge_size=args.max_hyperedge_size,
        max_incident_hyperedges=args.max_incident_hyperedges,
        max_members_per_hyperedge=args.max_members_per_hyperedge,
        max_child_hyperedges=args.max_child_hyperedges,
        formal_hidt_depth=args.formal_hidt_depth,
        overview_hops=args.overview_hops,
        overview_order_buckets=args.overview_order_buckets,
        seed=args.seed,
    )
    meta = build_ogbn_arxiv_hg(config)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
