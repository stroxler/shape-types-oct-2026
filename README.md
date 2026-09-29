# Shape types demo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
bash bootstrap.sh
```

The script creates an ignored `.venv` with Python 3.13, installs the numerical
libraries and version-matched Pyrefly shape stubs, then installs JAX shape stubs
from a pinned Pyrefly GitHub revision. In VS Code, select `.venv/bin/python` as
the Python interpreter. The Pyrefly executable is `.venv/bin/pyrefly`.
If VS Code has a global `pyrefly.lspPath` override, set its workspace value to
`.venv/bin/pyrefly` so the extension uses the pinned version.

The first shape examples are in `demo/`. Check them from the repo root with:

```sh
.venv/bin/pyrefly check demo/building_blocks.py demo/simple_numpy.py
```
