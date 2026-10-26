"""End-to-end smoke test on Qwen3-0.6B: every model-touching command once.

Opt-in (downloads ~2GB, takes a few minutes on Apple Silicon):

    LI_SMOKE=1 uv run pytest tests/test_smoke.py -s

Runs in a temporary workspace, so it never touches the repo's li.toml,
models/ or runs/. LI_SMOKE_WORKDIR=<dir> reuses a workspace (and its
downloads) between runs.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

SMOKE = os.environ.get("LI_SMOKE") == "1"


def li(cwd: Path, *args: str, timeout: int = 1800) -> str:
    proc = subprocess.run([sys.executable, "-m", "li.cli", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f"li {' '.join(args)} failed ({proc.returncode}):\n"
                             f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return proc.stdout


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(SMOKE, "set LI_SMOKE=1 to run the end-to-end smoke test")
class SmokeTest(unittest.TestCase):
    def test_pipeline(self):
        work = os.environ.get("LI_SMOKE_WORKDIR")
        tmp = None if work else tempfile.TemporaryDirectory()
        cwd = Path(work or tmp.name)
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            self._run(cwd)
        finally:
            if tmp:
                tmp.cleanup()

    def _run(self, cwd: Path) -> None:
        li(cwd, "init", "--model", "qwen3-0.6b")
        li(cwd, "download")
        li(cwd, "measure", "--n-harmful", "16", "--n-harmless", "16")
        li(cwd, "select", "--n", "8", "--per-pos", "1")
        li(cwd, "abliterate")
        run = cwd / "runs" / "qwen3-0.6b-4bit"
        self.assertTrue((run / "abliterated" / "adapters.safetensors").exists())

        li(cwd, "eval", "--n", "8", "--max-tokens", "48")
        summary = json.loads((cwd / "evals" / "qwen3-0.6b-4bit" / "summary.json").read_text())
        self.assertEqual(set(summary), {"base", "abliterated"})
        for row in summary.values():
            self.assertIn("advbench_refusal", row)        # judge ran
            self.assertLess(row["perplexity"], 200)       # model still coherent

        # Qwen3-0.6B ties its embedding and output head: export must untie them
        export = run / "export"
        shutil.rmtree(export, ignore_errors=True)   # reused LI_SMOKE_WORKDIR
        out = li(cwd, "export")
        cfg = json.loads((export / "config.json").read_text())
        self.assertFalse(cfg.get("tie_word_embeddings", True))
        probs = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("base", "adapter", "export"):
                probs[parts[0]] = float(parts[1])
        self.assertLess(abs(probs["export"] - probs["adapter"]),
                        0.5 * abs(probs["base"] - probs["adapter"]) + 0.02)

        port = free_port()
        server = subprocess.Popen(
            [sys.executable, "-m", "li.cli", "serve", "--no-think", "--port", str(port)],
            cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base = f"http://127.0.0.1:{port}/v1"
            for _ in range(120):
                try:
                    models = json.load(urllib.request.urlopen(f"{base}/models", timeout=2))
                    break
                except OSError:
                    time.sleep(1)
            else:
                self.fail("li serve did not come up")
            self.assertEqual([m["id"] for m in models["data"]], ["qwen3-0.6b-abliterated"])
            req = urllib.request.Request(
                f"{base}/chat/completions", method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"model": "any-client-name", "max_tokens": 16,
                                 "messages": [{"role": "user", "content": "Say hi."}]}).encode())
            reply = json.load(urllib.request.urlopen(req, timeout=120))
            self.assertTrue(reply["choices"][0]["message"]["content"].strip())
        finally:
            server.terminate()
            server.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
