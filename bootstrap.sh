#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# The JAX stubs are not published alongside the other shape stub packages.
# Install them from the matching Pyrefly revision without their placeholder deps.
uv pip install --python .venv/bin/python --no-deps \
  'git+https://github.com/facebook/pyrefly.git@ef08065bc7d691fd2c24884f813d42770c785c7b#subdirectory=tensor-shapes/pyrefly-jax-stubs'

echo "Select $(pwd)/.venv/bin/python as your VS Code Python interpreter."
