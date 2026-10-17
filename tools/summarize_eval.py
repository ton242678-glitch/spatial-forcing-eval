"""Compare structured evaluation reports and optionally export task-level CSV."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openvla-SF"))
from experiments.robot.libero.eval_support import summarize_episodes


def summarize_files(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            report = json.load(stream)
        if report.get("schema_version") != 1:
            raise ValueError(f"Unsupported report schema: {path}")
        if report.get("status") not in ("running", "completed", "aborted"):
            raise ValueError(f"Invalid report status: {path}")
        if report.get("data_kind") not in ("evaluation", "synthetic_example"):
            raise ValueError(f"Invalid report data_kind: {path}")
        # Recompute statistics from records instead of trusting cached totals.
        summary, tasks = summarize_episodes(report["episodes"])
        config = report["config"]
        metadata = {"source": str(path), "run_id": report["run_id"],
                    "data_kind": report["data_kind"], "status": report["status"],
                    "task_suite": config.get("task_suite_name", ""), "seed": config.get("seed", ""),
                    "checkpoint": config.get("pretrained_checkpoint", "")}
        rows.append({**metadata, "scope": "overall", "task_id": "", "task_description": "", **summary})
        rows.extend({**metadata, "scope": "task", **task} for task in tasks)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, help="Write UTF-8 task and run summaries")
    args = parser.parse_args()
    try:
        rows = summarize_files(args.reports)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for row in rows:
        rate = "N/A" if row["success_rate"] is None else f"{row['success_rate']:.2%}"
        task = row["task_id"] if row["scope"] == "task" else "ALL"
        print(f"{row['run_id']} [{row['data_kind']}, {row['status']}] task={task}: "
              f"{row['successes']}/{row['completed']} ({rate}), "
              f"errors={row['errors']}, skipped={row['skipped']}")


if __name__ == "__main__":
    main()
