# Find the cross-attention bugs

Start with [`original.py`](original.py): a four-head cross-attention layer with
32 features (8 per head). It accepts a query of shape `[B, Q, 32]` and context
of shape `[B, K, 32]`; `Q` and `K` need not be equal. Try spotting the two
axis-handling mistakes before opening [`annotated.py`](annotated.py).

From the repo root, the unannotated file passes static checking:

```sh
.venv/bin/pyrefly check small_examples/attention_bug/original.py
```

A small smoke test also passes when all three dimensions happen to equal 4:

```sh
.venv/bin/python -c 'import torch; from small_examples.attention_bug.original import CrossAttention; print(CrossAttention()(torch.randn(4, 4, 32), torch.randn(4, 4, 32)).shape)'
```

Now open `annotated.py` in VS Code and inspect the shape hints. The first class,
`CrossAttention`, performs the same operations as `original.py`, but its
interface names the independent dimensions `B`, `Q`, and `K`. The typed
intermediate tensors give each mistake its own diagnostic:

```sh
.venv/bin/pyrefly check --config small_examples/attention_bug/pyrefly.toml
```

The expected result is **2 errors** in `CrossAttention`: the inferred query tensor has shape
`[Q, 4, B, 8]` rather than `[B, 4, Q, 8]`, and the inferred key tensor has
shape `[4, K, B, 8]` rather than `[B, 4, K, 8]`. Below the `# ------` separator,
`FixedCrossAttention` shows the same model with both operations repaired.
This lets you see how the annotations first catch the mistakes and then check
the fix. The file retains the broken class deliberately, so the command above
continues to report two errors even though the fixed class checks cleanly.

Try the fixed model with distinct lengths:

```sh
.venv/bin/python -c 'import torch; from small_examples.attention_bug.annotated import FixedCrossAttention; print(FixedCrossAttention()(torch.randn(2, 3, 32), torch.randn(2, 5, 32)).shape)'
```

If you want to do the repair yourself, change the two buggy operations in
`CrossAttention` (and optionally `original.py`), then re-run the check. It will
report 0 errors once both are corrected.

<details>
<summary>Show the two fixes</summary>

In `CrossAttention`, the query reshape needs `(batch, query_length, 4, 8)` instead of
`(query_length, batch, 4, 8)`. The key projection needs `.transpose(1, 2)`
instead of `.transpose(0, 2)`. Compare `FixedCrossAttention` below the separator.

</details>

The repo-root `.venv/bin/pyrefly check` intentionally excludes `annotated.py`
so the rest of the demo stays green; the local `pyrefly.toml` lets VS Code
check this exercise when you open it.
