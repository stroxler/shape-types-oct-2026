#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "Install curl to download the pinned Pyrefly porting skill." >&2
  exit 1
fi

pyrefly_revision=ef08065bc7d691fd2c24884f813d42770c785c7b

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# The JAX stubs are not published alongside the other shape stub packages.
# Install them from the matching Pyrefly revision without their placeholder deps.
uv pip install --python .venv/bin/python --no-deps \
  "git+https://github.com/facebook/pyrefly.git@${pyrefly_revision}#subdirectory=tensor-shapes/pyrefly-jax-stubs"

# Keep the complete skill bundle at the same revision as the JAX stubs.
skill_dir=.agents/skills/add-shape-types-to-torch-model
skill_url="https://raw.githubusercontent.com/facebook/pyrefly/${pyrefly_revision}/tensor-shapes/skills/add-shape-types-to-torch-model"
mkdir -p "$skill_dir"
for skill_file in SKILL.md porting_principles.md shape_tracking_capabilities.md style_guide.md verify_port.sh; do
  curl --fail --location --silent --show-error --retry 3 \
    --output "$skill_dir/$skill_file.tmp" "$skill_url/$skill_file"
  mv "$skill_dir/$skill_file.tmp" "$skill_dir/$skill_file"
done
chmod +x "$skill_dir/verify_port.sh"

echo "Select $(pwd)/.venv/bin/python as your VS Code Python interpreter."
