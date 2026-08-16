#!/usr/bin/env python
"""Prove — or disprove — that a parameter is unreachable from the forward graph.

Gate for the implementation-defect cleanup in
docs/PROTOCOL_AMENDMENT_2026-08-16.md A4. A parameter may only be removed if all
three independent tests agree it takes no part in the computation:

  1. FORWARD REACHABILITY  its owning module never executes (forward hook silent)
  2. AUTOGRAD GRAPH        it is absent from the set of AccumulateGrad leaves
                           reachable from the output tensor's grad_fn
  3. BACKWARD              it receives `grad is None` after a full backward pass

Test 2 is the decisive one: walking the autograd graph enumerates exactly the
tensors that contributed to the output, independent of any hook or heuristic.

Frozen parameters (`requires_grad=False`) are legitimately absent from the graph
and are excluded from the dead set — otherwise every frozen backbone weight would
be misreported as dead.

    LOCAL / COLAB   python scripts/prove_dead_parameters.py

Runs on CPU with scratch weights. No dataset, no downloads, no training.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.models import build_model  # noqa: E402

CANDIDATES = {
    "C2-7": dict(tf_backbone="mobilevit_xxs", ae_space="feature"),
    "C2-14": dict(tf_backbone="mobilevit_xxs", ae_space="feature", tf_stage=3),
    "C2-28": dict(tf_backbone="mobilevit_xxs", ae_space="feature", tf_stage=2),
    # Controls: paths that MUST keep the image-space head.
    "C1 (image-space AE)": dict(tf_backbone="mobilevit_xxs", ae_space="image"),
    "C0 (frozen v1, ViT-B/16)": dict(),
}


def params_in_autograd_graph(out: torch.Tensor) -> set[int]:
    """ids of every leaf tensor reachable from `out` through the autograd graph.

    `next_functions` hands back a *fresh* Python wrapper for each graph node on
    every access. Those wrappers are garbage-collected as soon as they go out of
    scope, and CPython recycles their `id()`s -- so a visited-set keyed on `id(fn)`
    alone will wrongly mark unvisited nodes as seen and truncate the walk. Every
    visited node is therefore kept alive in `keepalive` for the duration.
    """
    reachable: set[int] = set()
    seen: set[int] = set()
    keepalive: list = []
    stack = [out.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keepalive.append(fn)                     # prevents id() recycling
        var = getattr(fn, "variable", None)      # AccumulateGrad carries the leaf
        if var is not None:
            reachable.add(id(var))
        for nxt, _ in getattr(fn, "next_functions", ()):
            stack.append(nxt)
    return reachable


def analyse(label: str, over: dict) -> dict:
    cfg = dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
               num_classes=39, img_size=224, pretrained=False, vit_pretrained=False)
    cfg.update(over)
    model = build_model(cfg).train()

    executed: set[str] = set()
    handles = [mod.register_forward_hook(
        lambda m, i, o, n=name: executed.add(n))
        for name, mod in model.named_modules() if name]

    logits = model(torch.rand(2, 3, 224, 224))
    for h in handles:
        h.remove()

    reachable = params_in_autograd_graph(logits)
    logits.sum().backward()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    dead = []
    for n, p in trainable:
        owner = n.rsplit(".", 1)[0]
        in_graph = id(p) in reachable
        ran = owner in executed
        has_grad = p.grad is not None
        if not in_graph and not ran and not has_grad:
            dead.append((n, p.numel()))
        elif not (in_graph and has_grad):
            # Disagreement between the three tests -- never silently treat as dead.
            dead.append((n + "  [INCONSISTENT: "
                         f"graph={in_graph} executed={ran} grad={has_grad}]", p.numel()))

    return {"label": label, "total": sum(p.numel() for _, p in model.named_parameters()),
            "trainable": sum(p.numel() for _, p in trainable), "dead": dead}


def main() -> int:
    print(f"{'candidate':<26}{'params':>12}{'trainable':>12}{'dead tensors':>14}{'dead values':>13}")
    print("-" * 77)
    results = []
    for label, over in CANDIDATES.items():
        r = analyse(label, over)
        results.append(r)
        print(f"{r['label']:<26}{r['total']:>12,}{r['trainable']:>12,}"
              f"{len(r['dead']):>14}{sum(n for _, n in r['dead']):>13,}")

    print()
    any_dead = False
    for r in results:
        if r["dead"]:
            any_dead = True
            print(f"{r['label']}:")
            for n, cnt in r["dead"]:
                print(f"    {n:<44} {cnt:>7,} values")

    print()
    if any_dead:
        print("VERDICT: dead parameters present. All three tests agreed for each entry")
        print("         unless marked INCONSISTENT -- an INCONSISTENT entry must NOT")
        print("         be removed until it is understood.")
    else:
        print("VERDICT: no dead parameters. Every trainable parameter is reachable")
        print("         from the forward graph and receives a gradient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
