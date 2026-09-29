# Shape-annotated model examples

These models come from Pyrefly's
[PyTorch shape examples](https://github.com/facebook/pyrefly/tree/main/tensor-shapes/pyrefly-torch-stubs/examples)
and [JAX shape examples](https://github.com/facebook/pyrefly/tree/main/tensor-shapes/pyrefly-jax-stubs/examples).
The two files under `jax/` are BERT and NanoGPT; the remaining models under
`torch/` use PyTorch. The PyTorch BERT and NanoGPT ports are omitted because
their JAX counterparts already demonstrate those model families here.

The Pyrefly versions adapt models from TorchBenchmark and other upstream
projects. Each file retains its source and license header, and many include
more detailed provenance in their module docstrings. Check those notices before
reusing a particular model; the repository's MIT license does not replace any
upstream terms identified there.

This copy removes `assert_type` calls and their imports, along with empty
type-checking blocks left behind by that removal. Those assertions help test
the Pyrefly stubs, but in this demo, editor inlay hints and hover show inferred
shapes without interrupting the model code. The model implementations and shape
annotations are otherwise unchanged. The `runtime/` test variants are not
included; these examples are intended for static exploration, and some models
need additional packages to execute.

After running `bootstrap.sh`, check the models with `.venv/bin/pyrefly check`
from the repository root.
