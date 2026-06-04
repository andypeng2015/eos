# Compiler Inspection

Eos can now emit inspection artifacts directly from the CLI. These commands are
intended for compiler development, backend debugging, CI artifact capture, and
cross-repo triad work where the generated graph or backend kernel source needs
to be reviewed without opening a debugger.

## Compile Bundle

```bash
eos compile --bundle bundle/ embed.eos embed.mll
```

The bundle directory contains:

```text
source.eos
artifact.mll
graph.json
manifest.json
kernels/
```

`graph.json` is the same structured report produced by `eos graph --format
json`. `kernels/` contains one source file per backend kernel variant. The
top-level `manifest.json` records module identity, artifact path, entrypoint,
step, kernel, backend, and kernel-source counts.

Bundle writing is best-effort after artifact compilation. If the artifact was
written but bundle sidecars fail, `compile` prints a warning so production
artifact creation does not fail because of an inspection-only sidecar.

## Graph Reports

```bash
eos graph --format json embed.eos
eos graph --format dot embed.mll > graph.dot
```

For source inputs, graph reports include HIR, MIR, LIR, and artifact snapshots.
For `.mll` inputs, graph reports include the artifact snapshot and counts
available from the sealed plan.

The JSON format is meant for tools and regression fixtures. The DOT format is a
compact visual map of modules, entrypoints, kernels, and scheduled steps.

## Kernel Extraction

```bash
eos kernels --backend webgpu --out kernels/ embed.mll
eos kernels --out kernels/ embed.eos
```

Kernel extraction accepts either source or artifact input. The command writes a
`manifest.json` beside the extracted sources. Backend names are sorted in the
manifest for stable comparisons.

Common backend filters:

```text
cuda
metal
vulkan
directml
webgpu
```

## Doctor

```bash
eos doctor
```

`doctor` reports the artifact schema, Go runtime, registered backend kinds,
backend capabilities, relevant local tools, and environment variables that
affect runtime or profiling behavior. It is intentionally text-first so it can
be pasted into issue reports or CI logs.
