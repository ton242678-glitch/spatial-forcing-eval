import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openvla-SF"))
from experiments.robot.libero.eval_support import (
    EvaluationReport, _atomic_json, select_action_chunk, summarize_episodes, validate_evaluation_config,
)


def config(**changes):
    values = dict(pretrained_checkpoint="haofuly/spatial-forcing-7b-finetuned-libero-spatial",
                  model_family="openvla", task_suite_name="libero_spatial", center_crop=True,
                  load_in_8bit=False, load_in_4bit=False, use_l1_regression=True, use_diffusion=False,
                  num_trials_per_task=50, num_open_loop_steps=8, env_img_res=256,
                  num_images_in_input=2, num_steps_wait=10, seed=7,
                  num_diffusion_steps_train=50, num_diffusion_steps_inference=50)
    values.update(changes)
    return SimpleNamespace(**values)


class ActionChunkTests(unittest.TestCase):
    def test_short_window_uses_first_actions_not_last(self):
        self.assertEqual(select_action_chunk(range(8), 2), [0, 1])

    def test_full_and_short_model_outputs(self):
        self.assertEqual(select_action_chunk(range(8), 8), list(range(8)))
        self.assertEqual(select_action_chunk(iter([0, 1]), 8), [0, 1])

    def test_empty_policy_and_invalid_windows(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            select_action_chunk([], 2)
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_action_chunk(range(8), size)


class ConfigurationTests(unittest.TestCase):
    def validate(self, cfg):
        validate_evaluation_config(cfg, task_suites={"libero_spatial"}, action_chunk_size=8)

    def test_valid_checkpoint_ids_and_paths(self):
        self.validate(config())
        self.validate(config(pretrained_checkpoint=Path("ckpts/model"), num_open_loop_steps=1))

    def test_invalid_configs_are_rejected(self):
        cases = [dict(pretrained_checkpoint=""), dict(pretrained_checkpoint="  "),
                 dict(pretrained_checkpoint=None), dict(model_family="unknown"),
                 dict(task_suite_name="other"), dict(num_open_loop_steps=0),
                 dict(num_open_loop_steps=9), dict(num_trials_per_task=-1),
                 dict(env_img_res=0), dict(num_images_in_input=True),
                 dict(seed=-1), dict(seed=2**32), dict(num_steps_wait=-1),
                 dict(load_in_8bit=True, load_in_4bit=True), dict(use_diffusion=True),
                 dict(pretrained_checkpoint="ckpts/image_aug", center_crop=False),
                 dict(use_l1_regression=False, use_diffusion=True, num_diffusion_steps_inference=0)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(config(**changes))

    def test_optimized_python_does_not_disable_validation(self):
        import subprocess
        code = ("from experiments.robot.libero.eval_support import validate_evaluation_config; "
                "from types import SimpleNamespace; "
                "validate_evaluation_config(SimpleNamespace(pretrained_checkpoint=''), "
                "task_suites={'libero_spatial'}, action_chunk_size=8)")
        import os
        result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True,
                                env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "openvla-SF")))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError", result.stderr)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "eval.json"
        self.report = EvaluationReport(self.path, "test", config(pretrained_checkpoint=Path("ckpts/model")))

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_incremental_counts_include_errors_exclude_skips(self):
        self.report.record(0, 0, "抓取", success=True)
        self.report.record(0, 1, "抓取")
        self.report.record(1, 0, "放置", error="RuntimeError: camera unavailable")
        self.report.record(1, 1, "放置", skipped=True)
        payload = self.read()
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["summary"], dict(completed=3, successes=1, failures=1,
                                                errors=1, skipped=1, success_rate=1/3))
        self.assertEqual(payload["tasks"][0]["success_rate"], 0.5)
        self.assertEqual(payload["config"]["pretrained_checkpoint"], str(Path("ckpts/model")))
        self.report.finish()
        self.assertEqual(self.read()["status"], "completed")

    def test_empty_and_skipped_only_reports_are_unscored(self):
        self.assertIsNone(self.read()["summary"]["success_rate"])
        self.report.record(0, 0, "task", skipped=True)
        self.report.finish()
        self.assertIsNone(self.read()["summary"]["success_rate"])

    def test_abort_keeps_completed_episodes(self):
        self.report.record(0, 0, "task", success=True)
        self.report.finish(error="KeyboardInterrupt")
        self.assertEqual(self.read()["status"], "aborted")
        self.assertEqual(self.read()["summary"]["successes"], 1)
        with self.assertRaises(ValueError):
            self.report.record(0, 1, "task")

    def test_duplicate_or_conflicting_records_do_not_corrupt_progress(self):
        self.report.record(0, 0, "task", success=True)
        for args, kwargs in [((0, 0, "task"), {}), ((0, 1, "other"), {}),
                             ((1, 0, "task"), dict(success=True, error="error")),
                             ((1, 0, "task"), dict(skipped=True, success=True)),
                             ((-1, 0, "task"), {}), ((1, 0, "task"), dict(success="False"))]:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                self.report.record(*args, **kwargs)
        self.assertEqual(len(self.read()["episodes"]), 1)
        self.report.record(0, 1, "task")
        self.assertEqual(self.read()["summary"]["completed"], 2)

    def test_failed_atomic_replace_preserves_previous_valid_json(self):
        previous = self.path.read_bytes()
        with patch("experiments.robot.libero.eval_support.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                _atomic_json(self.path, {"replacement": True})
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_unknown_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_episodes([dict(task_id=0, episode_index=0, task_description="task", outcome="maybe")])


if __name__ == "__main__":
    unittest.main()
