"""Exercise the actual evaluator with stubbed model/simulator dependencies.

These tests validate orchestration, not learned policy quality or physics.
"""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openvla-SF"))
from experiments.robot.libero.eval_support import EvaluationReport


def stub(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def load_evaluator():
    helpers = {
        "experiments.robot.libero.libero_utils": ["get_libero_dummy_action", "get_libero_env", "get_libero_image",
                                                 "get_libero_wrist_image", "quat2axisangle", "save_rollout_video"],
        "experiments.robot.openvla_utils": ["get_action_head", "get_noisy_action_projector", "get_processor",
                                          "get_proprio_projector", "resize_image_for_policy"],
        "experiments.robot.robot_utils": ["get_action", "get_image_resize_size", "get_model",
                                         "invert_gripper_action", "normalize_gripper_action", "set_seed_everywhere"],
    }
    replacements = {name: stub(name, **{key: Mock() for key in names}) for name, names in helpers.items()}
    replacements["experiments.robot.robot_utils"].DATE_TIME = "TEST"
    replacements.update({
        "draccus": stub("draccus", wrap=lambda: lambda func: func),
        "numpy": stub("numpy", array=lambda value: value),
        "tqdm": stub("tqdm", tqdm=lambda iterable: iterable),
        "PIL": stub("PIL", Image=Mock()),
        "libero": stub("libero"),
        "libero.libero": stub("libero.libero", benchmark=Mock()),
        "wandb": stub("wandb", init=Mock(), log=Mock(), save=Mock()),
        "prismatic": stub("prismatic"), "prismatic.vla": stub("prismatic.vla"),
        "prismatic.vla.constants": stub("prismatic.vla.constants", NUM_ACTIONS_CHUNK=8),
    })
    path = ROOT / "openvla-SF/experiments/robot/libero/run_libero_eval.py"
    spec = importlib.util.spec_from_file_location("sf_evaluator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    replacements[spec.name] = module
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    module.log_message = Mock()
    return module


class Action:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return [self.value]


class EvaluatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_evaluator()
        self.cfg = self.module.GenerateConfig(pretrained_checkpoint="ckpts/model", num_open_loop_steps=2,
                                               num_steps_wait=0, num_trials_per_task=1)
        self.module.prepare_observation = lambda obs, size: ({}, "frame")
        self.module.process_action = lambda action, family: action
        self.module.get_action.return_value = [Action(i) for i in range(8)]

    def environment(self, *, done_after=2):
        executed = []

        def step(action):
            executed.append(action[0])
            return {}, 0, len(executed) >= done_after, {}

        env = SimpleNamespace(reset=lambda: None, get_observation=lambda: {},
                              set_init_state=lambda state: {}, step=step)
        return env, executed

    def test_actual_episode_executes_prediction_prefix(self):
        env, executed = self.environment()
        success, frames = self.module.run_episode(self.cfg, env, "task", object(), 224)
        self.assertTrue(success)
        self.assertEqual(executed, [0, 1])
        self.assertEqual(len(frames), 2)

    def test_requery_starts_after_requested_window(self):
        env, executed = self.environment(done_after=4)
        self.module.get_action.side_effect = [[Action(i) for i in range(8)], [Action(i) for i in range(10, 18)]]
        self.module.run_episode(self.cfg, env, "task", object(), 224)
        self.assertEqual(executed, [0, 1, 10, 11])
        self.assertEqual(self.module.get_action.call_count, 2)

    def test_empty_policy_output_is_recorded_as_error(self):
        env, executed = self.environment()
        self.module.get_action.return_value = []
        metrics = {}
        success, _ = self.module.run_episode(self.cfg, env, "task", object(), 224, episode_metrics=metrics)
        self.assertFalse(success)
        self.assertEqual(executed, [])
        self.assertIn("empty action chunk", metrics["error"])

    def test_invalid_config_never_loads_model(self):
        self.cfg.num_open_loop_steps = 0
        with self.assertRaises(ValueError):
            self.module.eval_libero(self.cfg)
        self.module.get_model.assert_not_called()

    def test_task_report_is_saved_before_video_encoding_failure(self):
        env, _ = self.environment()
        self.module.get_libero_env.return_value = (env, "task")
        self.module.load_initial_states = lambda *args: ([[]], None)
        suite = SimpleNamespace(get_task=lambda task_id: object())
        self.module.save_rollout_video.side_effect = OSError("encoder unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = EvaluationReport(path, "test", self.cfg)
            with self.assertRaisesRegex(OSError, "encoder"):
                self.module.run_task(self.cfg, suite, 0, object(), 224, report=report)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["successes"], 1)

    def test_skipped_custom_demonstration_is_not_attempted(self):
        self.cfg.initial_states_path = "custom.json"
        self.module.get_libero_env.return_value = (self.environment()[0], "task")
        self.module.load_initial_states = lambda *args: ([], {"task": {"demo_0": {"success": False}}})
        suite = SimpleNamespace(get_task=lambda task_id: object())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = EvaluationReport(path, "test", self.cfg)
            self.assertEqual(self.module.run_task(self.cfg, suite, 0, object(), 224, report=report), (0, 0))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["skipped"], 1)
            self.assertIsNone(payload["summary"]["success_rate"])
            self.module.get_action.assert_not_called()

    def setup_main(self, path):
        log = io.StringIO()
        self.module.initialize_model = Mock(return_value=(object(), None, None, None, None))
        self.module.get_image_resize_size.return_value = 224
        self.module.setup_logging = Mock(return_value=(log, str(path.with_suffix(".txt")), "test"))
        self.module.benchmark.get_benchmark_dict.return_value = {
            "libero_spatial": lambda: SimpleNamespace(n_tasks=1)
        }
        return log

    def test_keyboard_interrupt_marks_report_aborted_and_closes_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            log = self.setup_main(path)
            self.module.run_task = Mock(side_effect=KeyboardInterrupt())
            with self.assertRaises(KeyboardInterrupt):
                self.module.eval_libero(self.cfg)
            self.assertEqual(json.loads(path.read_text())["status"], "aborted")
            self.assertTrue(log.closed)

    def test_successful_main_completes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            log = self.setup_main(path)

            def task(*args, report):
                report.record(0, 0, "task", success=True)
                return 1, 1

            self.module.run_task = task
            self.assertEqual(self.module.eval_libero(self.cfg), 1.0)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["summary"]["success_rate"], 1.0)
            self.assertTrue(log.closed)


if __name__ == "__main__":
    unittest.main()
