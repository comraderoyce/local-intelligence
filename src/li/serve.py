"""OpenAI-compatible HTTP server for one local model, via mlx_lm.server.

mlx_lm.server is built for switching between Hugging Face repos, which trips
up clients that send their own configured model name (coding agents, chat
UIs): any `model` other than "default_model" is treated as a repo to download
and load. Here the server holds exactly one model, so every request maps to
it, and /v1/models lists that model under one stable id.

The server also wires memory for generation; mlx_lm's default (the device's
recommended working set, ~10.7GB on 16GB) is how a 16GB Mac starves
WindowServer, so the wired limit is capped to the weights plus headroom.
"""
from __future__ import annotations

import json
import sys
import time

WIRED_HEADROOM_GB = 2.0


def run_server(model_dir: str, adapter_dir: str | None, name: str, host: str,
               port: int, max_tokens: int, think: bool, weights_gb: float,
               extra: list[str]) -> None:
    import importlib

    from .stream import cap_wired_limit
    server = importlib.import_module("mlx_lm.server")

    created = int(time.time())

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        # single-model server: whatever name the client sends means this model
        return orig_load(self, "default_model", None, "default_model")

    def handle_models_request(self):
        self._set_completion_headers(200)
        self.end_headers()
        body = {"object": "list", "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "li"}]}
        self.wfile.write(json.dumps(body).encode())
        self.wfile.flush()

    orig_load = server.ModelProvider.load
    server.ModelProvider.load = load
    server.APIHandler.handle_models_request = handle_models_request
    cap_wired_limit(weights_gb + WIRED_HEADROOM_GB)

    argv = ["mlx_lm.server", "--model", model_dir, "--host", host, "--port", str(port),
            "--max-tokens", str(max_tokens)]
    if adapter_dir:
        argv += ["--adapter-path", adapter_dir]
    if not think:
        argv += ["--chat-template-args", json.dumps({"enable_thinking": False})]
    sys.argv = argv + extra
    print(f"serving {name} ({model_dir}{' + ' + adapter_dir if adapter_dir else ''})\n"
          f"  OpenAI-compatible endpoint: http://{host}:{port}/v1  (model id: {name}, "
          "any API key)\n  Ctrl-C stops the server", flush=True)
    server.main()
