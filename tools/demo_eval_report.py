"""Create explicitly synthetic results to try report export without a model or GPU."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openvla-SF"))
from experiments.robot.libero.eval_support import EvaluationReport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/synthetic-demo.json"))
    args = parser.parse_args()
    config = SimpleNamespace(task_suite_name="synthetic_demo", seed=7,
                             pretrained_checkpoint="SYNTHETIC_NO_MODEL_LOADED")
    report = EvaluationReport(args.output, "SYNTHETIC-DEMO", config, data_kind="synthetic_example")
    report.record(0, 0, "Synthetic pick task", success=True)
    report.record(0, 1, "Synthetic pick task")
    report.record(1, 0, "Synthetic place task", error="Synthetic policy error")
    report.record(1, 1, "Synthetic place task", skipped=True)
    report.finish()
    print(f"Synthetic example saved to {args.output}. No robot evaluation was performed.")


if __name__ == "__main__":
    main()
