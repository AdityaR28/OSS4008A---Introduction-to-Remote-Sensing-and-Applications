"""Run with python -m processing --project YOUR_PROJECT --authenticate."""

import argparse
import json
import logging
from pathlib import Path

from .event import Config, Window
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Compute the McDougall Creek optical pilot in Earth Engine and export durable assets.")
    parser.add_argument("--project", required=True, help="Your Earth Engine-enabled Cloud Project ID; no project is assumed.")
    parser.add_argument("--authenticate", action="store_true", help="Start the provider's interactive OAuth flow.")
    parser.add_argument("--output", type=Path, default=Path("data/generated/mcdougall-creek-2023"))
    parser.add_argument("--reference", type=Path, default=Path("data/reference/mcdougall-creek-2023.geojson"))
    parser.add_argument("--pre-start", default="2023-07-15")
    parser.add_argument("--pre-end", default="2023-08-15", help="Exclusive end date")
    parser.add_argument("--post-start", default="2023-09-15")
    parser.add_argument("--post-end", default="2023-10-16", help="Exclusive end date")
    parser.add_argument("--threshold", type=float, default=0.27)
    parser.add_argument("--cloud-threshold", type=float, default=0.6)
    parser.add_argument("--min-valid-fraction", type=float, default=0.4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = Config(pre=Window(args.pre_start, args.pre_end), post=Window(args.post_start, args.post_end),
                    threshold=args.threshold, cloud_score_threshold=args.cloud_threshold,
                    min_valid_fraction=args.min_valid_fraction)
    manifest = run_pipeline(args.project, args.output, config=config, authenticate=args.authenticate,
                            reference_path=args.reference if args.reference.exists() else None)
    print(json.dumps({"output": str(args.output.resolve()), "metrics": manifest["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
