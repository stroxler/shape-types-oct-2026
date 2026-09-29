# Find the cross-attention bugs

Start with [`original.py`](original.py): a four-head cross-attention layer with
32 features (8 per head). It accepts a query of shape `[B, Q, 32]` and context
of shape `[B, K, 32]`; `Q` and `K` need not be equal. Try spotting the
axis-handling mistakes before opening [`annotated.py`](annotated.py).

From the repo root, the unannotated file passes static checking:

```sh
.venv/bin/pyrefly check small_examples/attention_bug/original.py
```

A small smoke test also passes when all three dimensions happen to equal 4:

```sh
.venv/bin/python -c 'import torch; from small_examples.attention_bug.original import CrossAttention; print(CrossAttention()(torch.randn(4, 4, 32), torch.randn(4, 4, 32)).shape)'
```

Now open `annotated.py` in VS Code and inspect the shape hints. Its
`CrossAttention` class performs the same operations as `original.py`, but its
interface names the independent dimensions `B`, `Q`, and `K`. The typed
intermediate tensors show where the shapes go wrong:

```sh
.venv/bin/pyrefly check --config small_examples/attention_bug/pyrefly.toml
```

The expected result is **3 errors**, at `q`, `k`, and `v`. Pyrefly infers
`[4, B, Q, 8]` for `q` rather than `[B, 4, Q, 8]`, and `[4, K, B, 8]` for
both `k` and `v` rather than `[B, 4, K, 8]`. The identical-looking
`.transpose(0, 2)` calls are all wrong; `q` also has its batch and sequence
length reversed in `.reshape()`.

[`fixed.py`](fixed.py) shows the same model with those operations repaired.
This lets you see the annotations catch the mistakes and then check the fix.
The local check above still reports 3 errors because it includes the deliberately
broken `annotated.py`. To check the fix alone:

```sh
.venv/bin/pyrefly check small_examples/attention_bug/fixed.py
```

Try the fixed model with distinct lengths:

```sh
.venv/bin/python -c 'import torch; from small_examples.attention_bug.fixed import CrossAttention; print(CrossAttention()(torch.randn(2, 3, 32), torch.randn(2, 5, 32)).shape)'
```

If you want to do the repair yourself, change the buggy operations in
`annotated.py` (and optionally `original.py`), then re-run the local check.
It will report 0 errors once the annotated model is corrected.

<details>
<summary>Show the fixes</summary>

The query reshape needs `(batch, query_length, 4, 8)` instead of
`(query_length, batch, 4, 8)`. Each of `q`, `k`, and `v` needs
`.transpose(1, 2)` instead of `.transpose(0, 2)`. Compare `fixed.py`.

</details>

The repo-root `.venv/bin/pyrefly check` intentionally excludes `annotated.py`
so the rest of the demo stays green; the local `pyrefly.toml` lets VS Code
check this exercise when you open it.
