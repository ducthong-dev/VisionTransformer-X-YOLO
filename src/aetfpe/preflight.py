"""The A100 readiness gates: what must be true before the real campaign starts.

Every gate is recorded as evidence on Drive as it is proven, not asserted at the
end from memory, so the readiness verdict survives a runtime disconnect and can
be re-read later. `GateBook.report()` prints one line per gate and then exactly
one of two sentences, which is the only thing the operator has to read:

    READY FOR A100 FULL CAMPAIGN
    NOT READY FOR A100 FULL CAMPAIGN

The evaluators below derive their verdict from artifacts (manifests, provenance
stamps, metrics files) rather than from a notebook variable, so a gate cannot
pass because an earlier cell was run and then edited.
"""

from __future__ import annotations

import json
import os
import time

from . import provenance as prov

GATES = (
    ("dependency_setup", "dependency setup"),
    ("cuda", "CUDA"),
    ("dataset_counts", "dataset counts"),
    ("no_test_access", "no-test-access"),
    ("smoke_A", "architecture smoke A"),
    ("smoke_E", "architecture smoke E"),
    ("smoke_M", "architecture smoke M"),
    ("smoke_F", "architecture smoke F"),
    ("smoke_B", "architecture smoke B"),
    ("drive_persistence", "Drive persistence"),
    ("drive_only_resume", "Drive-only resume"),
    ("isolation", "preflight/scientific isolation"),
    ("scientific_manifest_clean", "scientific manifest clean"),
    ("a3_forced", "A3 forced"),
)
GATE_KEYS = tuple(k for k, _ in GATES)
LABELS = dict(GATES)

READY = "READY FOR A100 FULL CAMPAIGN"
NOT_READY = "NOT READY FOR A100 FULL CAMPAIGN"

# Which architecture family each smoke gate stands for. The five arms that ran on
# the T4 are A0, E5, M1, F1, B2 -- one per family prefix.
SMOKE_FAMILIES = {"A": "smoke_A", "E": "smoke_E", "M": "smoke_M",
                  "F": "smoke_F", "B": "smoke_B"}


class GateBook:
    """Durable gate results under <drive_root>/preflight/manifest/."""

    def __init__(self, drive_root: str):
        self.drive_root = drive_root
        self.dir = os.path.join(drive_root, prov.NS_PREFLIGHT, "manifest")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "preflight_gates.json")
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as fh:
                    return json.load(fh)
            except Exception:  # noqa: BLE001
                pass
        return {"gates": {}}

    def record(self, key: str, passed: bool, detail: str = "") -> bool:
        if key not in GATE_KEYS:
            raise KeyError(f"unknown gate {key!r}; known: {GATE_KEYS}")
        self.data.setdefault("gates", {})[key] = {
            "passed": bool(passed), "detail": detail,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)
        print(f"  gate {LABELS[key]:<32} {'PASS' if passed else 'FAIL'}"
              + (f"  -- {detail}" if detail else ""))
        return bool(passed)

    def record_many(self, results: dict) -> None:
        for k, (ok, detail) in results.items():
            self.record(k, ok, detail)

    def get(self, key: str) -> dict:
        return (self.data.get("gates") or {}).get(key, {})

    def failed(self) -> list[str]:
        return [k for k in GATE_KEYS if not self.get(k).get("passed")]

    def report(self) -> bool:
        """Print every gate, then exactly one verdict sentence. Returns readiness."""
        print("=" * 78)
        print("A100 READINESS GATES")
        print("=" * 78)
        for key in GATE_KEYS:
            g = self.get(key)
            state = "PASS" if g.get("passed") else "FAIL"
            print(f"{LABELS[key]:<34} {state}")
            if g.get("detail"):
                print(f"{'':<34}   {g['detail']}")
            elif not g:
                print(f"{'':<34}   not evaluated")
        bad = self.failed()
        print("=" * 78)
        if not bad:
            print(READY)
            return True
        print(NOT_READY)
        print("\nfailed gates:")
        for key in bad:
            g = self.get(key)
            why = g.get("detail") or ("not evaluated" if not g else "no detail recorded")
            print(f"  - {LABELS[key]}: {why}")
        return False


# --------------------------------------------------------------------------- #
# Evaluators. Each returns (passed, detail).
# --------------------------------------------------------------------------- #

def completed_preflight_runs(drive_root: str) -> dict:
    """Run ID -> its record, for every finished run in the preflight tree.

    Read from the artifacts rather than from the preflight manifest, so evidence
    migrated out of the old flat layout counts exactly like a run launched by
    the current engine.
    """
    ck = os.path.join(drive_root, prov.NS_PREFLIGHT, "checkpoints")
    out = {}
    if not os.path.isdir(ck):
        return out
    for rid in sorted(os.listdir(ck)):
        d = os.path.join(ck, rid)
        summ = os.path.join(d, "train_summary.json")
        if not os.path.isdir(d) or not os.path.exists(summ):
            continue
        try:
            with open(summ) as fh:
                js = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        rec = prov.load(d)
        if js.get("status") != "completed" or not prov.is_smoke(rec):
            continue
        out[rid] = {"summary": js, "provenance": rec}
    return out


def evaluate_architecture_smoke(drive_root: str) -> dict:
    """One gate per architecture family, satisfied by a completed preflight run."""
    done = completed_preflight_runs(drive_root)
    out = {}
    for letter, key in sorted(SMOKE_FAMILIES.items()):
        hits = [r for r in done if r[:1] == letter]
        out[key] = (bool(hits),
                    f"completed preflight run(s): {', '.join(hits)}" if hits
                    else f"no completed preflight run whose ID starts with {letter!r}")
    return out


def evaluate_drive_persistence(drive_root: str, min_runs: int = 1) -> tuple[bool, str]:
    """Artifacts survived on Drive, not only in /content scratch."""
    ck = os.path.join(drive_root, prov.NS_PREFLIGHT, "checkpoints")
    if not os.path.isdir(ck):
        return False, f"missing {ck}"
    good, bad = [], []
    for rid in sorted(os.listdir(ck)):
        d = os.path.join(ck, rid)
        if not os.path.isdir(d):
            continue
        need = ["metrics.csv", "train_summary.json"]
        missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
        (good if not missing else bad).append(rid if not missing else f"{rid}({missing})")
    if len(good) < min_runs:
        return False, f"only {len(good)} run(s) with durable artifacts on Drive; incomplete: {bad}"
    return True, f"{len(good)} run(s) with metrics.csv + train_summary.json on Drive: {', '.join(good)}"


def evaluate_resume(drive_root: str) -> tuple[bool, str]:
    """Read the Drive-only resume test's evidence file."""
    p = os.path.join(drive_root, prov.NS_PREFLIGHT, "manifest", "resume_test.json")
    if not os.path.exists(p):
        return False, "no resume_test.json -- run scripts/preflight_resume_test.py"
    try:
        js = json.load(open(p))
    except Exception as exc:  # noqa: BLE001
        return False, f"resume_test.json unreadable ({exc})"
    checks = js.get("checks") or {}
    failed = [k for k, v in checks.items() if not v.get("passed")]
    if failed or not js.get("passed"):
        return False, f"resume test failed checks: {failed or 'see resume_test.json'}"
    return True, (f"resumed at epoch {js.get('resumed_at_epoch')}, "
                  f"metrics contiguous 1..{js.get('final_epoch')}, "
                  f"best preserved {js.get('best_before')} -> {js.get('best_after')}")


def evaluate_isolation(drive_root: str) -> tuple[bool, str]:
    """No shared directory, no legacy flat tree, no misfiled artifact."""
    problems = []
    for legacy in ("checkpoints", "logs", "campaign"):
        p = os.path.join(drive_root, legacy)
        if os.path.isdir(p):
            problems.append(f"legacy flat {legacy}/ still live at the Drive root "
                            "(run scripts/migrate_campaign_namespaces.py --apply)")
    for ns in prov.NAMESPACES:
        marker = os.path.join(drive_root, ns, "NAMESPACE")
        if not os.path.exists(marker):
            problems.append(f"missing {ns}/NAMESPACE marker")
        ck = os.path.join(drive_root, ns, "checkpoints")
        if not os.path.isdir(ck):
            problems.append(f"missing {ns}/checkpoints")
            continue
        for rid in sorted(os.listdir(ck)):
            d = os.path.join(ck, rid)
            if not os.path.isdir(d):
                continue
            rec = prov.load(d)
            if not rec:
                problems.append(f"{ns}/checkpoints/{rid} carries no provenance record")
            elif rec.get("namespace") != ns:
                problems.append(f"{ns}/checkpoints/{rid} is stamped "
                                f"namespace={rec.get('namespace')!r}")
            elif ns == prov.NS_SCIENTIFIC and prov.is_smoke(rec):
                problems.append(f"scientific/checkpoints/{rid} is a smoke artifact")
            elif ns == prov.NS_PREFLIGHT and not prov.is_smoke(rec):
                problems.append(f"preflight/checkpoints/{rid} claims to be full-data")
    if problems:
        return False, "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else "")
    return True, ("preflight/ and scientific/ are disjoint trees, both stamped, "
                  "no legacy flat directories remain")


def evaluate_scientific_manifest_clean(drive_root: str,
                                       smoke_ids: tuple = ()) -> tuple[bool, str]:
    """The scientific manifest must not count a smoke run as COMPLETED."""
    p = os.path.join(drive_root, prov.NS_SCIENTIFIC, "manifest", "campaign_manifest.json")
    if not os.path.exists(p):
        return False, f"no scientific manifest at {p} -- build it before the campaign"
    try:
        js = json.load(open(p))
    except Exception as exc:  # noqa: BLE001
        return False, f"scientific manifest unreadable ({exc})"
    if js.get("namespace") != prov.NS_SCIENTIFIC:
        return False, f"manifest declares namespace {js.get('namespace')!r}"
    ck = os.path.join(drive_root, prov.NS_SCIENTIFIC, "checkpoints")
    bad = []
    completed = []
    for rid, r in sorted((js.get("runs") or {}).items()):
        if r.get("status") != "COMPLETED":
            continue
        completed.append(rid)
        rec = prov.load(os.path.join(ck, rid))
        if not rec:
            bad.append(f"{rid}: COMPLETED with no provenance on Drive")
        elif prov.is_smoke(rec):
            bad.append(f"{rid}: COMPLETED from a smoke artifact")
        elif not rec.get("full_data"):
            bad.append(f"{rid}: COMPLETED from a subset run")
    contaminated = [r for r in smoke_ids if r in completed]
    if contaminated:
        bad.append(f"T4 smoke IDs counted as COMPLETED: {contaminated}")
    if bad:
        return False, "; ".join(bad)
    return True, (f"{len(completed)} COMPLETED run(s), all full-data with verified "
                  f"provenance{'' if completed else ' (none yet -- expected before the campaign)'}")


def evaluate_a3_forced(force_ids, scientific_campaign=None) -> tuple[bool, str]:
    """A3 is forced, and F3 is still a logical reuse rather than a second training."""
    from .campaign import FORCED_FUSION_IDS, REUSE, SKIPPED_SIZE

    ids = list(force_ids or ())
    problems = []
    if "A3" not in ids:
        problems.append("A3 is not in FORCE_LARGE_IDS")
    missing = [i for i in FORCED_FUSION_IDS if i not in ids]
    if missing:
        problems.append(f"forced fusion set incomplete, missing {missing}")
    if REUSE.get("F3", ("",))[0] != "A3":
        problems.append("F3 is no longer a logical reuse of A3")
    if "F3" in ids:
        problems.append("F3 must not be trained separately -- it reuses A3")
    if scientific_campaign is not None:
        r = (scientific_campaign.manifest.get("runs") or {}).get("A3", {})
        if r.get("status") == SKIPPED_SIZE:
            problems.append("A3 is still SKIPPED_SIZE in the scientific manifest")
        f3 = (scientific_campaign.manifest.get("runs") or {}).get("F3", {})
        if f3.get("status") not in ("SKIPPED_REUSE", None):
            problems.append(f"F3 is {f3.get('status')} rather than SKIPPED_REUSE")
    if problems:
        return False, "; ".join(problems)
    return True, (f"forced physical trainings {list(FORCED_FUSION_IDS)}; "
                  "F3 remains a logical reuse of A3")
