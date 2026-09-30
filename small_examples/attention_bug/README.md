# Find the cross-attention bugs

Start with [`original.py`](original.py): a four-head cross-attention layer with
32 features (8 per head). It expects a query of shape `[B, Q, 32]` and context
of shape `[B, K, 32]`; `B`, `Q`, and `K` need not be equal. Try spotting the
two axis-handling mistakes before opening [`annotated.py`](annotated.py).

From the repo root, the unannotated file passes static checking:

```sh
.venv/bin/pyrefly check small_examples/attention_bug/original.py
```

Now open `annotated.py` in VS Code and inspect the Pyrefly inlay type hints and
hovers for `q`, `k`, `v`, `scores`, `attended`, and `merged`. It performs the same
operations as `original.py`, but its `forward` signature supplies the symbolic
dimensions `B`, `Q`, and `K`. The intermediate variables have **no explicit
type annotations**: Pyrefly infers their shapes from the signature and the
installed PyTorch shape stubs. The completely unannotated `original.py` lacks
that starting point, so its tensor dimensions cannot be tracked as precisely.

From the repo root, check the exercise with:

```sh
.venv/bin/pyrefly check --config small_examples/attention_bug/pyrefly.toml
```

The expected result is **2 errors**. The `q`, `k`, and `v` shapes are correct:
`[B, 4, Q, 8]`, `[B, 4, K, 8]`, and `[B, 4, K, 8]`. But the score calculation
transposes the wrong axes of `k`, producing `[B, K, 4, 8]`; matrix multiplication
with `q` is invalid. Later, `merged` has shape `[Q, B, 32]` instead of the
declared return shape `[B, Q, 32]`. Inspect the inferred shapes to find the
two mistakes without relying on annotations at the intermediate assignments.

[`fixed.py`](fixed.py) shows the same model with those operations repaired and
the same inferred intermediate shapes. The local check above still reports 2
errors because it includes the deliberately broken `annotated.py`. To check
the fix alone:

```sh
.venv/bin/pyrefly check small_examples/attention_bug/fixed.py
```

Try the fixed model with distinct lengths:

```sh
.venv/bin/python -c 'import torch; from small_examples.attention_bug.fixed import CrossAttention; print(CrossAttention()(torch.randn(2, 3, 32), torch.randn(2, 5, 32)).shape)'
```

If you want to do the repair yourself, change the buggy operations in
`annotated.py` (and optionally `original.py`), then re-run the local check.
It will report 0 errors once the shape-typed model is corrected.

<details>
<summary>Show the fixes</summary>

The score calculation needs `k.transpose(-2, -1)` instead of
`k.transpose(1, 2)`. The final reshape needs `(batch, query_length, 32)`
instead of `(query_length, batch, 32)`. Compare `fixed.py`.

</details>

The repo-root `.venv/bin/pyrefly check` intentionally excludes `annotated.py`
so the rest of the demo stays green; the local `pyrefly.toml` lets VS Code
check this exercise when you open it.
