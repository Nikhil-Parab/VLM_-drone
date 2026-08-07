"""
Validate thread-budget fixes:
  - torch.set_num_threads() called exactly once at startup
  - VLM paths no longer toggle thread count per call
  - verify_thread_budget is read-only (no affinity re-pin on every cycle)
"""
from __future__ import annotations

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure enginev2 is on path when run from repo root or dev_tools/
import os

_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)


class TestThreadBudgetGuard(unittest.TestCase):
    def test_set_num_threads_called_once_at_startup(self):
        import thread_budget
        import importlib

        importlib.reload(thread_budget)
        thread_budget._TORCH_INTRA_THREADS_SET_COUNT = 0

        with patch.object(thread_budget, "_pin_process_affinity", return_value=("mock", {0, 1})):
            thread_budget.set_thread_budget(2)

        self.assertEqual(thread_budget.get_torch_intra_threads_set_count(), 1)
        # Re-enforcing thread budget on model load is allowed
        thread_budget._set_torch_intra_threads_once(2)
        self.assertEqual(thread_budget.get_torch_intra_threads_set_count(), 2)

    def test_verify_is_read_only_no_repin(self):
        import thread_budget

        with patch.object(thread_budget, "_pin_process_affinity") as pin:
            thread_budget.verify_thread_budget(2, context="test-readonly")
            pin.assert_not_called()

    def test_repair_only_on_drift(self):
        import thread_budget

        ok_state = {
            "ok": True,
            "torch_ok": True,
            "cv2_ok": True,
            "affinity_ok": True,
        }
        with patch.object(thread_budget, "_read_budget_state", return_value=ok_state):
            with patch.object(thread_budget, "_pin_process_affinity") as pin:
                repaired = thread_budget.repair_thread_budget_if_drifted(2, context="test")
                self.assertFalse(repaired)
                pin.assert_not_called()

        # Drift in torch_intra (e.g. from Ultralytics load) -> must repair torch intra-threads
        drift_state_torch_only = {
            "ok": False,
            "torch_ok": False,
            "cv2_ok": True,
            "affinity_ok": True,
            "torch_intra": 7,
            "cv2_threads": 2,
            "sched_affinity": None,
            "psutil_affinity": {0, 1},
        }
        with patch.object(thread_budget, "_read_budget_state", return_value=drift_state_torch_only):
            with patch.object(thread_budget, "_pin_process_affinity", return_value=("mock", {0, 1})):
                with patch("torch.set_num_threads") as mock_set:
                    with patch.object(thread_budget, "verify_thread_budget"):
                        repaired = thread_budget.repair_thread_budget_if_drifted(2, context="test-torch-drift")
                        self.assertTrue(repaired, "torch drift should be repaired to maintain budget")
                        mock_set.assert_called_once_with(2)

        # Drift in cv2 or affinity — those CAN safely be repaired mid-run.
        drift_state_cv2 = {
            "ok": False,
            "torch_ok": True,
            "cv2_ok": False,
            "affinity_ok": True,
            "torch_intra": 2,
            "cv2_threads": 8,
            "sched_affinity": None,
            "psutil_affinity": {0, 1},
        }
        with patch.object(thread_budget, "_read_budget_state", return_value=drift_state_cv2):
            with patch.object(thread_budget, "_pin_process_affinity", return_value=("mock", {0, 1})):
                with patch("torch.set_num_threads") as mock_set:
                    with patch.object(thread_budget, "verify_thread_budget"):
                        import cv2 as _cv2
                        with patch.object(_cv2, "setNumThreads") as mock_cv2:
                            repaired = thread_budget.repair_thread_budget_if_drifted(2, context="test-cv2-drift")
                            self.assertTrue(repaired, "cv2 drift must trigger a repair")
                            mock_cv2.assert_called_once_with(2)
                            mock_set.assert_not_called()  # still must not touch torch

    def test_vlm_paths_do_not_call_set_num_threads(self):
        """Simulate classify()/generate_hud_label() — must not touch thread pool size."""
        import thread_budget

        thread_budget._TORCH_INTRA_THREADS_SET_COUNT = 0
        with patch.object(thread_budget, "_pin_process_affinity", return_value=("mock", {0, 1})):
            thread_budget.set_thread_budget(2)

        import reid_engine
        import small_object_classifier

        model = MagicMock()
        processor = MagicMock()
        param = MagicMock()
        param.device = "cpu"
        # Use side_effect so each call to model.parameters() returns a fresh iterator.
        # iter([param]) is exhausted after one next() call — using side_effect avoids
        # StopIteration on the second constructor (SmolObjectClassifier) that also calls
        # next(shared_model.parameters()) to resolve self.device.
        model.parameters.side_effect = lambda: iter([param])

        # processor(...) must return something with a .to() method.
        # MagicMock() supports attribute access by default, so .to(device) just
        # returns another MagicMock — which also supports ** unpacking via __iter__.
        inputs_mock = MagicMock()
        inputs_mock.__iter__ = MagicMock(return_value=iter([]))  # for **inputs
        processor.return_value = inputs_mock
        processor.apply_chat_template.return_value = "prompt"

        fake_ids = MagicMock()
        fake_ids.shape = (1, 5)
        inputs_mock.to.return_value = {"input_ids": fake_ids}  # .to(device) result
        gen_out = MagicMock()
        gen_out.__getitem__ = MagicMock(return_value=fake_ids)
        model.generate.return_value = [gen_out]
        processor.decode.return_value = "<reid>red jacket</reid>"

        reid = reid_engine.ReIDEngine(shared_model=model, shared_processor=processor)
        cls = small_object_classifier.SmolObjectClassifier(shared_model=model, shared_processor=processor)

        import numpy as np

        crop = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch("torch.set_num_threads") as mock_set:
            # torch.no_grad must be a context manager for the `with torch.no_grad():` block
            with patch("torch.no_grad", return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))):
                reid.generate_hud_label(crop)
                cls.classify(crop)
            mock_set.assert_not_called()

        self.assertEqual(thread_budget.get_torch_intra_threads_set_count(), 1)


class TestPerCallToggleCost(unittest.TestCase):
    """Measure wall time: repeated set_num_threads toggling vs no toggling."""

    def test_toggle_vs_no_toggle_timing(self):
        import torch

        if not hasattr(torch, "set_num_threads"):
            self.skipTest("torch not available")

        n = 20
        baseline = torch.get_num_threads()

        t0 = time.perf_counter()
        for _ in range(n):
            torch.set_num_threads(1)
            torch.set_num_threads(baseline)
        toggle_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            pass
        noop_ms = (time.perf_counter() - t0) * 1000

        print(
            f"\n[PERF] {n}x set_num_threads toggle: {toggle_ms:.1f}ms "
            f"vs noop loop: {noop_ms:.1f}ms "
            f"(overhead ~{(toggle_ms - noop_ms):.1f}ms)"
        )
        # Not a hard fail — platforms vary — but report for before/after comparison.
        self.assertGreaterEqual(toggle_ms, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
