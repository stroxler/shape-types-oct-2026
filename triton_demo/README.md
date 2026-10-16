# Triton shape-typing demo (Q&A / booth backup)

Two shape-annotated Triton kernels, simplified for showing in the editor:
`vector_add.py` (the universal starting point) and `layer_norm.py`. Each file
keeps its annotated kernel, its checked host boundary, and type-level probes;
the prototype's unittest scaffolding is dropped. Kernel bodies are unchanged
apart from added clarifying comments.

`typed_triton/` is a verbatim snapshot of the library and `triton-stubs`
overlay from the pyrefly-shapes-for-kernels prototype at commit `283eb24`,
so the demo keeps working even as the prototype moves on.

Static exploration only: Triton has no macOS wheels, so these kernels cannot
execute here. Check the demos with:

```sh
.venv/bin/pyrefly check --config triton_demo/pyrefly.toml
```

The config excludes the vendored support tree from diagnostics (its macOS-only
import gaps would otherwise show) while still resolving imports against it.
The repo-root check excludes this directory; use the command above.
