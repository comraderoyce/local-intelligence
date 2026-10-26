from __future__ import annotations

import unittest

import numpy as np

from li.abliteration.optimize import _directions, kernel
from li.data import _sample
from li.evals.refusal import _parse_label, _parse_wildguard


class DataSplitTests(unittest.TestCase):
    def test_named_splits_are_stable_and_disjoint(self):
        prompts = [f"prompt {i}" for i in range(600)]
        measure = _sample(prompts, 128, 7, "measure")
        select = _sample(prompts, 32, 7, "select")
        optimize = _sample(prompts, 64, 7, "optimize")
        evaluation = _sample(prompts, 100, 7, "eval")
        groups = [set(x) for x in (measure, select, optimize, evaluation)]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertFalse(left & right)
        self.assertEqual(measure, _sample(prompts, 128, 7, "measure"))

    def test_split_capacity_is_enforced(self):
        with self.assertRaises(ValueError):
            _sample([f"p{i}" for i in range(600)], 33, 0, "select")


class JudgeTests(unittest.TestCase):
    def test_label_parser_accepts_only_leading_label(self):
        self.assertEqual(_parse_label("REFUSAL"), "REFUSAL")
        self.assertEqual(_parse_label("Label: partial\nreason"), "PARTIAL")
        self.assertEqual(_parse_label("FULFILLED."), "FULFILLED")
        self.assertIsNone(_parse_label("The response is a REFUSAL"))

    def test_wildguard_parser(self):
        self.assertEqual(_parse_wildguard(
            "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"),
            "REFUSAL")
        self.assertEqual(_parse_wildguard("Response refusal: no"), "FULFILLED")
        self.assertIsNone(_parse_wildguard("refusal"))


class DirectionTests(unittest.TestCase):
    def test_per_layer_projected_directions(self):
        dirs = np.array([[[1.0, 1.0]], [[1.0, -1.0]]], dtype=np.float32)
        harmless = np.array([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
        out = _directions(dirs, harmless, 0.0, 0, "per_layer", True)
        self.assertEqual(out.shape, (2, 1, 2))
        np.testing.assert_allclose(out[:, 0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(out[:, 0], axis=1), 1.0)

    def test_kernel_peaks_and_reaches_zero(self):
        out = kernel(7, peak=3, width=2, a_max=1.0, a_min=0.2)
        self.assertEqual(out[3], 1.0)
        self.assertAlmostEqual(out[1], 0.2)
        self.assertEqual(out[0], 0.0)


if __name__ == "__main__":
    unittest.main()


class RegistryTests(unittest.TestCase):
    def test_custom_hf_id_resolves_after_init(self):
        from li.registry import resolve_spec, spec_for
        spec = resolve_spec("mistralai/Mistral-7B-Instruct-v0.3")
        # init saves only the short name; download must still find the source repo
        self.assertEqual(spec_for(spec.name, spec.repo).repo,
                         "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(spec_for("qwen3-8b", "Qwen/Qwen3-8B").mlx_repo,
                         "mlx-community/Qwen3-8B-4bit")


class EvalSummaryTests(unittest.TestCase):
    def test_unparsed_judge_labels_are_excluded_and_counted(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import li.evals as ev
        from li.config import Config
        cfg = Config(name="t", repo="t/t", model_dir="m", n_advbench_eval=4, n_xstest_eval=2)
        prompts = {"advbench": ["a1", "a2", "a3", "a4"], "xstest": ["x1", "x2"]}
        labels = iter(["REFUSAL", None, "FULFILLED", "FULFILLED", "FULFILLED", "FULFILLED"])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(ev, "load_advbench", return_value=prompts["advbench"]), \
                mock.patch.object(ev, "load_xstest", return_value=prompts["xstest"]), \
                mock.patch.object(ev, "_load", return_value=(None, None, None)), \
                mock.patch.object(ev, "complete", side_effect=lambda m, t, ps, c: ["r"] * len(ps)), \
                mock.patch.object(ev, "perplexity", return_value=10.0), \
                mock.patch.object(ev, "judge_responses", side_effect=lambda m, t, pairs, **k: [
                    {"label": next(labels), "raw": "", "version": "v"} for _ in pairs]):
            results, _ = ev.run_eval_suite({"base": "m"}, cfg, Path(tmp), judge_dir="j")
        row = results["base"]
        self.assertEqual(row["advbench_refusal_unjudged"], 1.0)
        self.assertAlmostEqual(row["advbench_refusal"], 1 / 3)
        self.assertEqual(row["xstest_overrefusal"], 0.0)
