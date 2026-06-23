#!/usr/bin/env python3
"""Prepare and execute audited run-artifact reclaim manifests.

The default behavior is dry-run only. Apply mode requires both explicit apply
flags and only removes exact manifest paths that pass the same safety checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUALITY_CLAIM = False
MANIFEST_SCHEMA = "eos.run_reclaim_manifest.v1"
SUMMARY_SCHEMA = "eos.run_reclaim_summary.v1"
EVIDENCE_LABEL = "diagnostic run artifact reclaim plan; not product-quality evidence"
CACHE_ROOTS = ("runs/external-vector-caches", "runs/eos-vector-caches")
SUMMARY_FIELDS = (
    "path",
    "exists",
    "kind",
    "estimated_bytes",
    "deletion_allowed",
    "deleted",
    "warnings",
)


class ReclaimError(ValueError):
    """Raised for invalid paths, manifests, or unsafe execution requests."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root_from(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def relative_text(path: Path) -> str:
    return path.as_posix().rstrip("/")


def has_parent_escape(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def cache_root_warning(rel_path: str) -> str | None:
    for root in CACHE_ROOTS:
        if rel_path == root or rel_path.startswith(root + "/"):
            return f"path is under special cache root {root}; deletion requires explicit cache-root allowance"
    return None


def normalize_manifest_path(
    value: str | Path,
    *,
    root: Path,
    allow_cache_roots: bool = False,
) -> str:
    original = Path(value)
    if has_parent_escape(original):
        raise ReclaimError(f"{value}: parent-directory escapes are not allowed")

    root = root.resolve()
    if original.is_absolute():
        try:
            relative = original.resolve().relative_to(root)
        except ValueError as exc:
            raise ReclaimError(f"{value}: absolute paths must be under repo root {root}") from exc
    else:
        relative = original

    if relative.is_absolute() or has_parent_escape(relative):
        raise ReclaimError(f"{value}: path must stay inside repo root")
    if not relative.parts:
        raise ReclaimError(f"{value}: empty path is not allowed")

    rel_path = relative_text(Path(*relative.parts))
    if rel_path in ("", "."):
        raise ReclaimError(f"{value}: empty path is not allowed")
    if rel_path.split("/", 1)[0] != "runs":
        raise ReclaimError(f"{value}: only paths under runs/ are reclaimable by default")
    if rel_path == "runs":
        raise ReclaimError("runs: refusing to reclaim the whole runs/ directory")

    first = rel_path.split("/", 1)[0]
    if first in {".git", "assets", "datasets", ".tiller", "scripts", "src", "docs"}:
        raise ReclaimError(f"{value}: protected top-level path {first!r} is not reclaimable")

    warning = cache_root_warning(rel_path)
    if warning and not allow_cache_roots:
        raise ReclaimError(f"{value}: {warning}")
    return rel_path


def manifest_entries(
    paths: list[str],
    *,
    root: Path,
    allow_cache_roots: bool = False,
) -> list[dict[str, Any]]:
    entries = []
    seen: set[str] = set()
    for raw_path in paths:
        rel_path = normalize_manifest_path(
            raw_path,
            root=root,
            allow_cache_roots=allow_cache_roots,
        )
        if rel_path in seen:
            continue
        seen.add(rel_path)
        warnings = []
        warning = cache_root_warning(rel_path)
        if warning:
            warnings.append(warning)
        entries.append({"path": rel_path, "warnings": warnings})
    return entries


def create_manifest(
    *,
    paths: list[str],
    output: Path | None,
    reason: str,
    root: Path | None = None,
    allow_cache_roots: bool = False,
) -> dict[str, Any]:
    if not reason.strip():
        raise ReclaimError("manifest reason is required")
    root = (root or repo_root_from()).resolve()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "quality_claim": QUALITY_CLAIM,
        "evidence_label": EVIDENCE_LABEL,
        "created_at": utc_now(),
        "root": str(root),
        "reason": reason,
        "paths": manifest_entries(paths, root=root, allow_cache_roots=allow_cache_roots),
    }
    if output:
        write_json(output, manifest)
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReclaimError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReclaimError(f"{path}: expected top-level object")
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ReclaimError(f"{path}: unsupported schema {data.get('schema')!r}")
    if data.get("quality_claim") is not False:
        raise ReclaimError(f"{path}: quality_claim must be false")
    if not isinstance(data.get("paths"), list):
        raise ReclaimError(f"{path}: paths must be a list")
    if not data.get("root"):
        raise ReclaimError(f"{path}: root is required")
    return data


def path_kind(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "missing"
    if path.is_symlink():
        if path.is_dir():
            return "dir"
        return "file"
    if path.is_dir():
        return "dir"
    return "file"


def estimate_bytes(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    try:
        if path.is_symlink() or path.is_file():
            return path.lstat().st_size
        if path.is_dir():
            total = 0
            for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
                base = Path(dirpath)
                total += base.lstat().st_size
                for name in list(dirnames):
                    child = base / name
                    if child.is_symlink():
                        total += child.lstat().st_size
                        dirnames.remove(name)
                for name in filenames:
                    total += (base / name).lstat().st_size
            return total
    except OSError:
        return 0
    return 0


def entry_path(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("path"), str):
        return entry["path"]
    raise ReclaimError(f"invalid manifest path entry {entry!r}")


def summarize_manifest(
    manifest: dict[str, Any],
    *,
    allow_cache_roots: bool = False,
    allow_missing: bool = False,
) -> dict[str, Any]:
    root = Path(str(manifest["root"])).resolve()
    rows = []
    total = 0
    seen: set[str] = set()
    invalid: list[str] = []
    missing_rejections: list[str] = []

    for raw_entry in manifest["paths"]:
        raw_path = entry_path(raw_entry)
        try:
            rel_path = normalize_manifest_path(
                raw_path,
                root=root,
                allow_cache_roots=allow_cache_roots,
            )
        except ReclaimError as exc:
            invalid.append(str(exc))
            continue
        if rel_path in seen:
            invalid.append(f"{rel_path}: duplicate manifest path")
            continue
        seen.add(rel_path)

        path = root / rel_path
        kind = path_kind(path)
        exists = kind != "missing"
        bytes_estimate = estimate_bytes(path)
        warnings = []
        warning = cache_root_warning(rel_path)
        if warning:
            warnings.append(warning)
        if path.is_symlink():
            warnings.append("path is a symlink; apply mode unlinks the symlink only")
        if kind == "missing" and not allow_missing:
            missing_rejections.append(f"{rel_path}: path is missing")

        deletion_allowed = exists and not invalid and kind in {"file", "dir"}
        rows.append(
            {
                "path": rel_path,
                "exists": exists,
                "kind": kind,
                "estimated_bytes": bytes_estimate,
                "deletion_allowed": deletion_allowed,
                "deleted": False,
                "warnings": warnings,
            }
        )
        total += bytes_estimate

    return {
        "schema": SUMMARY_SCHEMA,
        "quality_claim": QUALITY_CLAIM,
        "evidence_label": EVIDENCE_LABEL,
        "root": str(root),
        "reason": manifest.get("reason", ""),
        "dry_run": True,
        "apply": False,
        "allow_missing": allow_missing,
        "allow_cache_roots": allow_cache_roots,
        "paths": rows,
        "invalid": invalid,
        "missing_rejections": missing_rejections,
        "total_estimated_reclaim_bytes": total,
    }


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def execute_manifest(
    manifest: dict[str, Any],
    *,
    dry_run: bool = True,
    apply: bool = False,
    yes_delete_approved_run_artifacts: bool = False,
    allow_cache_roots: bool = False,
    allow_missing: bool = False,
) -> dict[str, Any]:
    if allow_missing and not dry_run:
        raise ReclaimError("--allow-missing is only allowed for idempotent dry-runs")
    if apply and dry_run:
        dry_run = False
    if not dry_run and (not apply or not yes_delete_approved_run_artifacts):
        raise ReclaimError("apply mode requires --apply --yes-delete-approved-run-artifacts")

    summary = summarize_manifest(
        manifest,
        allow_cache_roots=allow_cache_roots,
        allow_missing=allow_missing,
    )
    summary["dry_run"] = dry_run
    summary["apply"] = apply and not dry_run
    if summary["invalid"]:
        raise ReclaimError("; ".join(summary["invalid"]))
    if summary["missing_rejections"] and not allow_missing:
        raise ReclaimError("; ".join(summary["missing_rejections"]))
    if dry_run:
        return summary

    root = Path(str(manifest["root"])).resolve()
    for row in summary["paths"]:
        if not row["deletion_allowed"]:
            raise ReclaimError(f"{row['path']}: deletion is not allowed")
        path = root / str(row["path"])
        remove_path(path)
        row["deleted"] = True
        row["exists_after"] = path.exists() or path.is_symlink()
    return summary


def tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary["paths"]:
            writer.writerow({field: tsv_value(row.get(field)) for field in SUMMARY_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or execute audited run-artifact reclaim manifests.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    manifest = subcommands.add_parser("manifest", help="Create a reclaim manifest from exact paths.")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--reason", required=True)
    manifest.add_argument("--path", action="append", required=True, dest="paths")
    manifest.add_argument("--root", type=Path, help="Repository root. Defaults to detected current repo.")
    manifest.add_argument("--allow-cache-roots", action="store_true")

    execute = subcommands.add_parser("execute", help="Dry-run or apply a reclaim manifest.")
    execute.add_argument("manifest", type=Path)
    execute.add_argument("--dry-run", action="store_true", default=True)
    execute.add_argument("--apply", action="store_true")
    execute.add_argument("--yes-delete-approved-run-artifacts", action="store_true")
    execute.add_argument("--allow-cache-roots", action="store_true")
    execute.add_argument("--allow-missing", action="store_true")
    execute.add_argument("--output-json", type=Path)
    execute.add_argument("--output-tsv", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = create_manifest(
                paths=args.paths,
                output=args.output,
                reason=args.reason,
                root=args.root,
                allow_cache_roots=args.allow_cache_roots,
            )
            print(f"wrote reclaim manifest: paths={len(manifest['paths'])} output={args.output}")
            return 0

        manifest = load_manifest(args.manifest)
        dry_run = not args.apply
        summary = execute_manifest(
            manifest,
            dry_run=dry_run,
            apply=args.apply,
            yes_delete_approved_run_artifacts=args.yes_delete_approved_run_artifacts,
            allow_cache_roots=args.allow_cache_roots,
            allow_missing=args.allow_missing,
        )
    except ReclaimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_tsv:
        write_tsv(args.output_tsv, summary)
    if not args.output_json and not args.output_tsv:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "planned run reclaim: "
            f"paths={len(summary['paths'])} dry_run={str(summary['dry_run']).lower()} "
            f"bytes={summary['total_estimated_reclaim_bytes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
