#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DATASETS = ("scifact", "nfcorpus", "fiqa")


BASELINE = {
    "scifact": {"ndcg_at_10": 0.5645379155090131, "recall_at_100": 0.7964444444444444},
    "nfcorpus": {"ndcg_at_10": 0.20574596786076532, "recall_at_100": 0.24206606745988307},
    "fiqa": {"ndcg_at_10": 0.12126094061428457, "recall_at_100": 0.351678208622653},
}


def load_eos_rows(path):
    data = json.loads(Path(path).read_text())
    rows = {}
    for row in data.get("rows", []):
        if row.get("category") == "short_retrieval" and row.get("baseline") == "eos":
            rows[row["dataset"]] = row
    missing = [ds for ds in DATASETS if ds not in rows]
    if missing:
        raise SystemExit(f"missing eos scoreboard rows for: {','.join(missing)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--scoreboard", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    rows = load_eos_rows(args.scoreboard)
    datasets = {}
    for ds in DATASETS:
        row = rows[ds]
        base = BASELINE[ds]
        datasets[ds] = {
            "ndcg_at_10": row["ndcg_at_10"],
            "recall_at_100": row["recall_at_100"],
            "ndcg_at_10_delta": row["ndcg_at_10"] - base["ndcg_at_10"],
            "recall_at_100_delta": row["recall_at_100"] - base["recall_at_100"],
            "ndcg_floor_pass": row["ndcg_at_10"] - base["ndcg_at_10"] >= -0.0020,
            "recall_floor_pass": row["recall_at_100"] - base["recall_at_100"] >= -0.0030,
        }
    macro_ndcg = sum(datasets[ds]["ndcg_at_10"] for ds in DATASETS) / len(DATASETS)
    macro_recall = sum(datasets[ds]["recall_at_100"] for ds in DATASETS) / len(DATASETS)
    base_macro_ndcg = sum(BASELINE[ds]["ndcg_at_10"] for ds in DATASETS) / len(DATASETS)
    base_macro_recall = sum(BASELINE[ds]["recall_at_100"] for ds in DATASETS) / len(DATASETS)
    result = {
        "schema": "eos.s40_balanced_boundary_dense_gate.v1",
        "arm": args.arm,
        "scoreboard_json": str(Path(args.scoreboard).resolve()),
        "baseline": BASELINE,
        "datasets": datasets,
        "macro": {
            "ndcg_at_10": macro_ndcg,
            "recall_at_100": macro_recall,
            "ndcg_at_10_delta": macro_ndcg - base_macro_ndcg,
            "recall_at_100_delta": macro_recall - base_macro_recall,
            "ndcg_lift_pass": macro_ndcg - base_macro_ndcg >= 0.0010,
            "recall_floor_pass": macro_recall - base_macro_recall >= -0.0010,
            "continuation_signal": macro_ndcg - base_macro_ndcg >= 0.0004,
        },
    }
    result["exploration_pass"] = (
        result["macro"]["ndcg_lift_pass"]
        and result["macro"]["recall_floor_pass"]
        and all(datasets[ds]["ndcg_floor_pass"] and datasets[ds]["recall_floor_pass"] for ds in DATASETS)
    )
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        f"# {args.arm} dense gate",
        "",
        f"- exploration_pass: `{str(result['exploration_pass']).lower()}`",
        f"- macro nDCG@10 delta: `{result['macro']['ndcg_at_10_delta']:+.9f}`",
        f"- macro recall@100 delta: `{result['macro']['recall_at_100_delta']:+.9f}`",
        f"- continuation_signal_ge_0.0004: `{str(result['macro']['continuation_signal']).lower()}`",
        "",
        "| dataset | nDCG@10 delta | R@100 delta | nDCG floor | R@100 floor |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for ds in DATASETS:
        d = datasets[ds]
        lines.append(
            f"| {ds} | `{d['ndcg_at_10_delta']:+.9f}` | `{d['recall_at_100_delta']:+.9f}` | "
            f"`{str(d['ndcg_floor_pass']).lower()}` | `{str(d['recall_floor_pass']).lower()}` |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
