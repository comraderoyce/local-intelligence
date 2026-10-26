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
    def _run(self, labels, tmp, model_dir="m"):
        from pathlib import Path
        from unittest import mock

        import li.evals as ev
        from li.config import Config
        cfg = Config(name="t", repo="t/t", model_dir=model_dir)
        prompts = {"advbench": [f"a{i}" for i in range(50)],
                   "xstest": [f"x{i}" for i in range(50)]}
        labels = iter(labels)
        generated = []

        def complete(m, t, ps, c):
            generated.extend(ps)
            return ["r"] * len(ps)

        with mock.patch.object(ev, "load_advbench", return_value=prompts["advbench"]), \
                mock.patch.object(ev, "load_xstest", return_value=prompts["xstest"]), \
                mock.patch.object(ev, "_load", return_value=(None, None, None)), \
                mock.patch.object(ev, "complete", side_effect=complete), \
                mock.patch.object(ev, "perplexity", return_value=10.0), \
                mock.patch.object(ev, "judge_responses", side_effect=lambda m, t, pairs, **k: [
                    {"label": next(labels), "raw": "", "version": "v"} for _ in pairs]):
            results, _ = ev.run_eval_suite({"base": model_dir}, cfg, Path(tmp), judge_dir="j")
        return results, generated

    def test_one_unparsed_label_is_excluded_and_counted(self):
        import tempfile
        labels = ["REFUSAL", None] + ["FULFILLED"] * 98
        with tempfile.TemporaryDirectory() as tmp:
            row = self._run(labels, tmp, tmp)[0]["base"]
        self.assertEqual(row["advbench_refusal_unjudged"], 1.0)
        self.assertAlmostEqual(row["advbench_refusal"], 1 / 49)
        self.assertEqual(row["xstest_overrefusal"], 0.0)

    def test_too_many_unparsed_labels_fail_the_eval(self):
        import tempfile
        labels = [None] * 2 + ["FULFILLED"] * 98   # 2/50 = 4% > 2%
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            self._run(labels, tmp, tmp)

    def test_saved_responses_are_reused_only_for_unchanged_model(self):
        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as model:
            weights = Path(model) / "model.safetensors"
            weights.write_bytes(b"v1")
            self.assertEqual(len(self._run(["FULFILLED"] * 100, tmp, model)[1]), 100)
            self.assertEqual(self._run(["FULFILLED"] * 100, tmp, model)[1], [])
            weights.write_bytes(b"v2")   # e.g. `li optimize` rewrote the adapter
            os.utime(weights, ns=(1, 1))
            self.assertEqual(len(self._run(["FULFILLED"] * 100, tmp, model)[1]), 100)


class DataSourceTests(unittest.TestCase):
    def test_xstest_keeps_only_safe_prompts(self):
        from unittest import mock

        import li.data as data
        rows = [{"prompt": f"safe {i}", "label": "safe"} for i in range(250)] + \
               [{"prompt": f"unsafe {i}", "label": "unsafe"} for i in range(200)]
        with mock.patch("datasets.load_dataset", return_value=rows):
            prompts = data.load_xstest(250)
        self.assertTrue(all(p.startswith("safe") for p in prompts))


class GuardrailTests(unittest.TestCase):
    def test_moe_models_are_refused_for_abliteration(self):
        import json
        import tempfile
        from pathlib import Path

        from li.registry import check_abliteration_supported
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps({"num_hidden_layers": 4}))
            check_abliteration_supported(d)
            cfg.write_text(json.dumps({"text_config": {"num_experts": 128}}))
            with self.assertRaises(SystemExit):
                check_abliteration_supported(d)

    def test_unknown_config_keys_warn_and_load(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path

        from li.config import load_config
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "li.toml"
            path.write_text('name = "t"\nrepo = "t/t"\nmodel_dir = "m"\nold_key = 1\n')
            out = io.StringIO()
            with redirect_stdout(out):
                cfg = load_config(path)
        self.assertEqual(cfg.name, "t")
        self.assertIn("old_key", out.getvalue())
