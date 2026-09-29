# Shape types demo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
bash bootstrap.sh
```

The script creates an ignored `.venv` with Python 3.13, installs the numerical
libraries and version-matched Pyrefly shape stubs, then installs JAX shape stubs
from a pinned Pyrefly GitHub revision. In VS Code, select `.venv/bin/python` as
the Python interpreter. The repo's `.vscode/settings.json` tells the Pyrefly
extension to run `.venv/bin/pyrefly` from this workspace.

For the October 1, 2026 demo, an up-to-date Pyrefly extension should also be
compatible: extension version 1.3.9002 already bundles Pyrefly 1.4.0-dev.2,
matching the version installed by `bootstrap.sh`. The workspace setting keeps
the demo on its pinned version even if the extension's bundled version changes.

The first shape examples are in `demo/`. Check them from the repo root with:

```sh
.venv/bin/pyrefly check demo/building_blocks.py demo/simple_numpy.py
```
