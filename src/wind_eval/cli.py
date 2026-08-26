from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .coherence import evaluate_coherence
from .common import jsonable, write_json
from .wsci import evaluate_wsci


def _write_summary_csv(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, values in result.get("components", {}).items():
        rows.append({"component": name, "value": values.get("score")})
    if "WSCI_0_100" in result:
        rows.append({"component": "WSCI_0_100", "value": result["WSCI_0_100"]})
    if "coherence_0_100" in result:
        rows.append({"component": "coherence_0_100", "value": result["coherence_0_100"]})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "value"])
        writer.writeheader(); writer.writerows(rows)


def _output(result, output, csv_output):
    if output:
        write_json(output, result)
    if csv_output:
        _write_summary_csv(csv_output, result)
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))


def build_parser():
    p = argparse.ArgumentParser(prog="wind-eval")
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("wsci", help="calculate WSCI")
    w.add_argument("input", type=Path)
    w.add_argument("-o", "--output", type=Path)
    w.add_argument("--csv", type=Path)
    w.add_argument("--max-lag", type=int, default=8)
    w.add_argument("--gaussian-sigma", type=float, default=2.0)
    w.add_argument("--missing-policy", choices=["reweight", "legacy-perfect"], default="reweight")
    w.add_argument("--min-vertical-coverage", type=float, default=0.5)
    c = sub.add_parser("coherence", help="calculate space-time-terrain coherence")
    c.add_argument("input", type=Path)
    c.add_argument("-o", "--output", type=Path)
    c.add_argument("--csv", type=Path)
    c.add_argument("--max-space-lag", type=int, default=8)
    c.add_argument("--max-time-lag", type=int, default=8)
    c.add_argument("--dt", type=float, default=1.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "wsci":
            result = evaluate_wsci(args.input, max_lag=args.max_lag,
                gaussian_sigma=args.gaussian_sigma,
                min_vertical_coverage=args.min_vertical_coverage,
                missing_policy=args.missing_policy)
        else:
            result = evaluate_coherence(args.input,
                max_space_lag=args.max_space_lag,
                max_time_lag=args.max_time_lag, dt=args.dt)
        _output(result, args.output, args.csv)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def wsci_main():
    return main(["wsci", *sys.argv[1:]])


def coherence_main():
    return main(["coherence", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
