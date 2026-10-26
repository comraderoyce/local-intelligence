# Security policy

`li` is alpha software. Security fixes go into the latest release only.

## Reporting a vulnerability

Report it privately through GitHub: the repository's **Security** tab →
**Report a vulnerability**. Please don't open a public issue. Include the `li`
version (`li --version`), your macOS version, and steps to reproduce.

You should get a reply within a week.

## In scope

Problems in `li`'s own code, for example:

- a crafted model, adapter, dataset or `li.toml` that makes `li` run code,
  read or write files outside the workspace, or crash the machine despite the
  memory guard;
- `li serve` binding to anything other than the requested host, or exposing
  more than the OpenAI-compatible endpoints.

## Not vulnerabilities

- **What an abliterated model says.** Removing refusals is the tool's purpose;
  see [Responsible use](README.md#responsible-use).
- **`li serve` has no authentication.** It binds to `127.0.0.1` by default and
  is not meant to be exposed to a network. Passing `--host 0.0.0.0` exposes it
  to anyone who can reach the port.
- Bugs in MLX, `mlx-lm`, Hugging Face libraries or the models themselves:
  report those upstream.
