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

The introductory shape examples are in `small_examples/`. Check them from the
repo root with:

```sh
.venv/bin/pyrefly check
```

Larger shape-annotated models are under
[`example_oss_models/`](example_oss_models/README.md), with notes on their
upstream origins and the demo-specific edits.

## Annotate Whisper yourself

[`whisper_original/`](whisper_original/) contains the package and assets from
[OpenAI Whisper](https://github.com/openai/whisper) at commit
`86098128c0b4f24f0e2aa2994de830614b474227`, before any shape annotations.
It includes Whisper's original MIT license. The two Whisper copies have separate
Pyrefly configs so imports resolve against the copy you open in VS Code.

After bootstrapping, run this to see the starting point:

```sh
.venv/bin/pyrefly check --config whisper_original/pyrefly.toml --output-format omit-errors
```

The original reports 103 errors in the shape-annotation scope. Its config excludes
the normalizers and Triton helper, which are outside that scope. You
can edit `whisper_original/whisper/` and rerun the check as you annotate it.

[`whisper_annotated/`](whisper_annotated/) contains the annotated version at
commit `fe99f13a1c4ff8704d23548a53ec343376095ae0` of the Whisper clone.
It keeps the same package layout and OpenAI MIT license for comparison. Check
it independently with:

```sh
.venv/bin/pyrefly check --config whisper_annotated/pyrefly.toml
```

The annotated scope checks with 0 errors. Both Whisper configs exclude the
unchanged normalizers and Triton helper; the repo-root config excludes both
Whisper copies so `.venv/bin/pyrefly check` continues to check the smaller
examples without counting the intentionally unannotated baseline as errors.
The [Whisper annotation prompt](whisper_annotated/PROMPT.md) adapts the original
agent instructions to this repository's local environment and pinned skill.
Use it to try adding the annotations yourself; it is not a verbatim historical record.
The bootstrap script also downloads the
[Pyrefly shape-annotation skill](https://github.com/facebook/pyrefly/tree/ef08065bc7d691fd2c24884f813d42770c785c7b/tensor-shapes/skills/add-shape-types-to-torch-model)
into `.agents/skills/add-shape-types-to-torch-model/`. Its files are ignored by
Git and pinned to the same Pyrefly commit as the JAX stubs. Start a new agent
session after bootstrapping so the local skill can be discovered.
Running actual speech transcription additionally needs `ffmpeg` on your PATH
and will download model weights; neither is needed for static exploration.
