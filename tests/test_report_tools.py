import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReportToolTests(unittest.TestCase):
    def run_tool(self, tool, *args):
        return subprocess.run([sys.executable, str(ROOT / "tools" / tool), *map(str, args)],
                              cwd=ROOT, text=True, capture_output=True, encoding="utf-8")

    def test_demo_and_csv_export_without_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            report, csv_path = Path(directory) / "demo.json", Path(directory) / "summary.csv"
            result = self.run_tool("demo_eval_report.py", "--output", report)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No robot evaluation", result.stdout)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["data_kind"], "synthetic_example")
            # Cached summary tampering must not affect export.
            payload["summary"]["success_rate"] = 1.0
            report.write_text(json.dumps(payload), encoding="utf-8")
            result = self.run_tool("summarize_eval.py", report, "--csv", csv_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["completed"], "3")
            self.assertAlmostEqual(float(rows[0]["success_rate"]), 1/3)
            self.assertEqual(rows[0]["data_kind"], "synthetic_example")

    def test_invalid_input_has_clear_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"schema_version": 999}', encoding="utf-8")
            result = self.run_tool("summarize_eval.py", path)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Unsupported report schema", result.stderr)


if __name__ == "__main__":
    unittest.main()
