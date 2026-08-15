#!/usr/bin/env python
"""Print the deduplicated run matrix before any training is launched (Phase 5).

Two configs are the same run when their model config and protocol are identical.
Duplicates are reported as aliases so the fusion table can reuse an ablation
result instead of retraining it.

    python scripts/print_run_matrix.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import build_protocol, load_experiment  # noqa: E402

GROUP_ORDER = ["mechanism", "ablation", "fusion", "baseline", "misc"]


def signature(cfg: dict) -> str:
    model = dict(cfg.get("model") or {})
    proto = asdict(build_protocol(cfg))
    train = dict(cfg.get("train") or {})
    return json.dumps({"model": model, "protocol": proto, "train": train}, sort_keys=True, default=str)


def main() -> int:
    paths = sorted(p for p in glob.glob("configs/*.yaml")
                   if os.path.basename(p) not in ("_base.yaml", "corruptions.yaml")
                   and not os.path.basename(p).startswith("local"))

    entries = []
    for p in paths:
        cfg = load_experiment(p)
        entries.append({
            "path": p,
            "name": cfg.get("name", os.path.basename(p)),
            "group": cfg.get("group", "misc"),
            "description": cfg.get("description", ""),
            "model": cfg.get("model", {}),
            "train": cfg.get("train", {}),
            "declared_aliases": list(cfg.get("aliases") or []),
            "sig": signature(cfg),
        })

    by_sig: dict[str, list[dict]] = {}
    for e in entries:
        by_sig.setdefault(e["sig"], []).append(e)

    unique = []
    for sig, group in by_sig.items():
        group.sort(key=lambda e: (GROUP_ORDER.index(e["group"]) if e["group"] in GROUP_ORDER else 9,
                                  e["name"]))
        primary = group[0]
        # identical-signature duplicates, plus aliases the config declares itself
        primary["aliases"] = [g["name"] for g in group[1:]] + primary["declared_aliases"]
        unique.append(primary)

    unique.sort(key=lambda e: (GROUP_ORDER.index(e["group"]) if e["group"] in GROUP_ORDER else 9,
                               e["name"]))

    hdr = f"{'run':<24} {'group':<10} {'PE':<3} {'TF':<3} {'AE':<3} {'fusion':<10} {'extra':<18} aliases"
    print(hdr)
    print("-" * len(hdr))
    counts: dict[str, int] = {}
    for e in unique:
        m = e["model"]
        extra = []
        if m.get("legacy_lut"):
            extra.append("legacy-LUT")
        if m.get("photometric"):
            extra.append(m["photometric"])
        if e["train"].get("corruption_augmentation"):
            extra.append("aug-train")
        if m.get("classifier", "yolov8n-cls") != "yolov8n-cls":
            extra.append(m["classifier"])
        counts[e["group"]] = counts.get(e["group"], 0) + 1
        print(f"{e['name']:<24} {e['group']:<10} "
              f"{'Y' if m.get('use_pe') else '-':<3} "
              f"{'Y' if m.get('use_tf') else '-':<3} "
              f"{'Y' if m.get('use_ae') else '-':<3} "
              f"{(m.get('fusion') if m.get('use_tf') else 'identity'):<10} "
              f"{','.join(extra) or '-':<18} "
              f"{','.join(e['aliases']) or '-'}")

    print()
    print(f"config files       : {len(entries)}")
    print(f"unique training runs: {len(unique)}")
    for g in GROUP_ORDER:
        if g in counts:
            print(f"  {g:<10}: {counts[g]}")
    dupes = sum(len(e["aliases"]) for e in unique)
    print(f"deduplicated away  : {dupes} "
          f"({', '.join(a for e in unique for a in e['aliases']) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
