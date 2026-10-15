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

pyrefly_version=1.4.0

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Keep the complete skill bundle at the same release as the shape stubs.
skill_dir=.agents/skills/add-shape-types-to-torch-model
skill_url="https://raw.githubusercontent.com/facebook/pyrefly/${pyrefly_version}/tensor-shapes/skills/add-shape-types-to-torch-model"
mkdir -p "$skill_dir"
for skill_file in SKILL.md porting_principles.md shape_tracking_capabilities.md style_guide.md verify_port.sh; do
  curl --fail --location --silent --show-error --retry 3 \
    --output "$skill_dir/$skill_file.tmp" "$skill_url/$skill_file"
  mv "$skill_dir/$skill_file.tmp" "$skill_dir/$skill_file"
done
chmod +x "$skill_dir/verify_port.sh"

echo "Select $(pwd)/.venv/bin/python as your VS Code Python interpreter."
