# Annotate Whisper with Pyrefly shape types

Annotate `whisper_original/whisper/` in this repo with tensor shapes, preserving
runtime behavior and the public API. Keep `whisper_annotated/whisper/` unchanged;
consult it only after your own pass.

1. Run `./bootstrap.sh`, then read and follow
   `.agents/skills/add-shape-types-to-torch-model/SKILL.md` and its sibling docs.
   Bootstrap installs the skill, Pyrefly, shape stubs, and `shape_extensions`.
2. Annotate tensor/array modules in `whisper_original/whisper/`; leave
   `normalizers/` (no shape logic) and `triton_ops.py` (Triton boundary) alone.
   The local `pyrefly.toml` excludes both.
3. Track PyTorch `Tensor[[...]]`, NumPy `np.ndarray[[...]]`, and conversions.
   See `small_examples/simple_numpy.py` and `example_oss_models/torch/` for
   patterns. Preserve code structure; use temporary probes and focused casts
   at untracked boundaries. Do not edit shared stubs or Pyrefly; report gaps.
4. Before editing, verify the environment and baseline:

   ```sh
   .venv/bin/pyrefly dump-config --config whisper_original/pyrefly.toml whisper_original/whisper/model.py
   .venv/bin/pyrefly check small_examples/simple_numpy.py example_oss_models/torch/resnet.py
   .venv/bin/pyrefly check --config whisper_original/pyrefly.toml --output-format omit-errors
   ```

   Confirm the examples check cleanly, record Whisper's starting error count,
   and verify `.venv` and imports resolve from `whisper_original/`, not the
   annotated copy.
5. Iterate module by module using the skill's evidence, audit, probe, and
   verification steps. Run
   `.venv/bin/pyrefly check --config whisper_original/pyrefly.toml` to see
   diagnostics; fix errors while retaining useful shaped contracts. Probe
   inference, not just error counts. Run focused runtime checks with
   `PYTHONPATH=whisper_original .venv/bin/python` as needed;
   full transcription also needs `ffmpeg` and model weights.