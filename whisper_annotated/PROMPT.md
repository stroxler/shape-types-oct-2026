# Annotate Whisper with Pyrefly shape types

Work in this demo repository. Add tensor shape annotations to
`whisper_original/whisper/`, preserving its runtime behavior and public API.
Keep `whisper_annotated/whisper/` unchanged as a reference for comparison; try
to work through the original before consulting the annotated version.

1. Run `./bootstrap.sh` first. Read
   `.agents/skills/add-shape-types-to-torch-model/SKILL.md` and its sibling
   documentation, and follow its guidance for annotating existing code. The skill,
   Pyrefly, shape stubs, and `shape_extensions` are installed by bootstrap.
2. Add shape annotations to the package modules under
   `whisper_original/whisper/` that work with tensors and arrays.
   Leave `normalizers/` and `triton_ops.py` alone: the former has no tensor
   shape logic, and the latter is a separate Triton boundary. The local
   `pyrefly.toml` excludes both from the scoped project check.
3. Track both PyTorch `Tensor[[...]]` and NumPy `np.ndarray[[...]]` shapes,
   including their conversion boundaries. Consult `small_examples/simple_numpy.py`
   for NumPy syntax and `example_oss_models/torch/` for larger model patterns.
   Preserve structure and runtime behavior; use temporary type probes and
   focused casts when a library or dynamic boundary loses shape information.
   Do not edit the shared stubs or Pyrefly itself. Report any gaps you find.
4. Verify the environment and record the starting point before editing:

   ```sh
   .venv/bin/pyrefly dump-config --config whisper_original/pyrefly.toml whisper_original/whisper/model.py
   .venv/bin/pyrefly check small_examples/simple_numpy.py example_oss_models/torch/resnet.py
   .venv/bin/pyrefly check --config whisper_original/pyrefly.toml --output-format omit-errors
   ```

   The known-good examples should have 0 errors, and the unannotated Whisper
   scope starts at 103 errors. Pyrefly should resolve the repo's `.venv` and
   imports from `whisper_original/`, not the annotated copy.
5. Iterate module by module, following the skill's source-evidence, operation
   audit, probe, and verification steps. Use the scoped check without
   `--output-format omit-errors` to see and fix diagnostics:

   ```sh
   .venv/bin/pyrefly check --config whisper_original/pyrefly.toml
   ```

   Aim for 0 errors while retaining useful shaped input/output contracts.
   A clean check alone does not prove shapes were inferred; probe important
   boundaries. Run focused runtime checks after changes, importing the local
   package with `PYTHONPATH=whisper_original .venv/bin/python` when needed.
   Full transcription additionally requires `ffmpeg` and model weights.
6. Work in reviewable commits if useful. Do not commit `.venv`, downloaded
   skill files, temporary type probes, or working notes. Write a concise
   `whisper_original/SHAPE_ANNOTATION_REPORT.md` covering the final check, shaped
   component boundaries, remaining gradual/cast sites, and runtime checks.
   Compare your result with `whisper_annotated/whisper/` only after completing
   your own pass.
