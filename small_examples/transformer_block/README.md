# Find the transformer-block bug

Start with [`original.py`](original.py): a pre-norm transformer block with a
fused QKV projection, configurable model width `D` and head count `H`. Each
head has width `D // H`; the constructor requires positive dimensions and `D`
to be divisible by `H`. Defaults are `D = 64` and `H = 8`, giving 8 features
per head. It maps `[B, T, D]` to `[B, T, D]` through attention, output
projection, and an MLP, with a residual around each. Try spotting the
assembly mistake before opening [`annotated.py`](annotated.py).

From the repo root, the unannotated file passes static checking:

```sh
.venv/bin/pyrefly check small_examples/transformer_block/original.py
```

Now open `annotated.py` in VS Code and inspect the Pyrefly inlay type hints and
hovers for `qkv`, `q`, `k`, `v`, and `attn`. It performs the same operations
as `original.py`, but its constructor binds `D` and `H` while its `forward`
signature supplies the symbolic dimensions `B` and `T`. The intermediate
variables have **no explicit type annotations**: Pyrefly infers their shapes
from the signature and installed PyTorch shape stubs. The completely
unannotated `original.py` lacks that starting point, so its tensor dimensions
cannot be tracked as precisely.

From the repo root, check the exercise with:

```sh
.venv/bin/pyrefly check --config small_examples/transformer_block/pyrefly.toml
```

The expected result is **1 error**. The attention section is correct: `q`,
`k`, and `v` are `[B, H, T, D // H]`, and `attn` has that shape too. But the
heads are never merged back: `self.proj` expects a trailing dimension `D`
and gets `D // H`, so the residual add fails to check. Inspect the inferred
shapes to find the missing step without relying on annotations at the
intermediate assignments.

[`fixed.py`](fixed.py) shows the same model with the merge step repaired and
the same inferred intermediate shapes. The local check above still reports 1
error because it includes the deliberately broken `annotated.py`. To check
the fix alone:

```sh
.venv/bin/pyrefly check small_examples/transformer_block/fixed.py
```

Try the fixed model with a different width/head count:

```sh
.venv/bin/python -c 'import torch; from small_examples.transformer_block.fixed import TransformerBlock; print(TransformerBlock(96, 8)(torch.randn(2, 5, 96)).shape)'
```

If you want to do the repair yourself, add the merge step in `annotated.py`
(and optionally `original.py`), then re-run the local check. It will report
0 errors once the shape-typed model is corrected.

<details>
<summary>Show the fix</summary>

The attention output needs `merged = attn.transpose(1, 2).reshape(b, t,
self.dim)` and `self.proj(merged)` instead of `self.proj(attn)`. Compare
`fixed.py`.

</details>

The repo-root `.venv/bin/pyrefly check` intentionally excludes `annotated.py`
so the rest of the demo stays green; the local `pyrefly.toml` lets VS Code
check this exercise when you open it.
