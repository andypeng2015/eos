#!/usr/bin/env python3
"""Preset-friendly entry point for BEIR vector-cache export.

This keeps external embedding providers at the vector-cache boundary expected by
`eos eval-retrieval-vectors`:

  <output_root>/<dataset_name>/doc-vectors.jsonl
  <output_root>/<dataset_name>/query-vectors.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_qwen3_retrieval_vectors as exporter  # noqa: E402


PRESETS = {
    "qwen3-0.6b": {
        "model_name": exporter.DEFAULT_MODEL,
        "query_prefix": exporter.DEFAULT_QUERY_PREFIX,
        "document_prefix": "",
    },
    "mxbai-large": {
        "model_name": "mixedbread-ai/mxbai-embed-large-v1",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
    },
}


def expand_preset(argv: list[str]) -> list[str]:
    out: list[str] = []
    preset_name = ""
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--preset":
            if index + 1 >= len(argv):
                raise SystemExit("--preset requires a value")
            preset_name = argv[index + 1]
            index += 2
            continue
        out.append(value)
        index += 1

    if not preset_name:
        return out
    preset = PRESETS.get(preset_name)
    if preset is None:
        names = ", ".join(sorted(PRESETS))
        raise SystemExit(f"unknown --preset {preset_name!r}; available presets: {names}")

    if "--model-name" not in out:
        out.extend(["--model-name", preset["model_name"]])
    if "--query-prefix" not in out:
        out.extend(["--query-prefix", preset["query_prefix"]])
    if "--document-prefix" not in out:
        out.extend(["--document-prefix", preset["document_prefix"]])
    return out


def main() -> int:
    original_argv = sys.argv
    if any(arg in ("-h", "--help") for arg in original_argv[1:]):
        names = ", ".join(sorted(PRESETS))
        print(f"Preset wrapper options:\n  --preset NAME        Available presets: {names}\n")
    try:
        sys.argv = [original_argv[0], *expand_preset(original_argv[1:])]
        return exporter.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
