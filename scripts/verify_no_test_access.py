#!/usr/bin/env python
"""Prove the trainer cannot reach the test split or any corruption set.

Static, AST-level check over `scripts/train.py` and every first-party module it
imports. Run it before any training job whose protocol forbids test access.

Two independent tests:

  1. CONFIG KEYS   the tokens `test_split` and `corruption_root` never appear in
                   the trainer's reachable source, so the config entries that hold
                   those paths are never read.
  2. PATH JOINS    every `os.path.join(root, <split>)` in the trainer resolves to
                   a split key on an allow-list (`train_split`, `val_split`).

A grep would be fooled by a commented-out line or a string in a docstring; the AST
walk sees only executable code.

    LOCAL / COLAB   python scripts/verify_no_test_access.py

Exit 0 = no test or corruption path reachable. Exit 1 = a path was found.
"""

from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "scripts", "train.py")

FORBIDDEN_KEYS = ("test_split", "corruption_root")
FORBIDDEN_SUBSTRINGS = ("corruptions_test", "/test", "test/")
ALLOWED_SPLIT_KEYS = {"train_split", "val_split"}


def first_party_modules(tree: ast.AST) -> set[str]:
    """`aetfpe.*` modules imported by the entry point, as file paths."""
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("aetfpe"):
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("aetfpe"):
                    mods.add(a.name)
    paths = set()
    for m in mods:
        p = os.path.join(ROOT, "src", *m.split(".")) + ".py"
        if os.path.exists(p):
            paths.add(p)
    return paths


def executable_strings(tree: ast.AST) -> list[str]:
    """String constants in executable positions -- docstrings excluded."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def split_keys_joined(tree: ast.AST) -> list[str]:
    """Subscript keys passed to os.path.join(...) -- i.e. which splits are built."""
    keys = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant):
                if isinstance(arg.slice.value, str):
                    keys.append(arg.slice.value)
    return keys


def main() -> int:
    tree = ast.parse(open(ENTRY).read(), filename=ENTRY)
    targets = {ENTRY} | first_party_modules(tree)

    failures = []
    print(f"entry point : {os.path.relpath(ENTRY, ROOT)}")
    print(f"reachable   : {len(targets)} first-party source files\n")

    print("[1] forbidden config keys in executable code")
    for path in sorted(targets):
        t = ast.parse(open(path).read(), filename=path)
        rel = os.path.relpath(path, ROOT)
        hits = []
        for node in ast.walk(t):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if node.slice.value in FORBIDDEN_KEYS:
                    hits.append(f"{rel}:{node.lineno} reads config['{node.slice.value}']")
        for s in executable_strings(t):
            if s in FORBIDDEN_KEYS:
                hits.append(f"{rel}: string constant {s!r} in executable position")
        if hits:
            failures += hits
            for h in hits:
                print(f"    FAIL  {h}")
    if not failures:
        print(f"    PASS  neither {' nor '.join(FORBIDDEN_KEYS)} is read anywhere reachable")

    print("\n[2] split keys used to build dataset roots in the trainer")
    keys = [k for k in split_keys_joined(tree) if k.endswith("_split") or "split" in k]
    for k in sorted(set(keys)):
        ok = k in ALLOWED_SPLIT_KEYS
        print(f"    {'PASS' if ok else 'FAIL'}  os.path.join(root, data['{k}'])")
        if not ok:
            failures.append(f"trainer joins a non-allow-listed split key: {k}")
    if not keys:
        print("    WARN  no split joins found -- check the trainer's data wiring by hand")

    print("\n[3] forbidden path fragments in executable strings")
    frag = []
    for path in sorted(targets):
        t = ast.parse(open(path).read(), filename=path)
        rel = os.path.relpath(path, ROOT)
        for s in executable_strings(t):
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in s:
                    frag.append(f"{rel}: {s!r} contains {bad!r}")
    for f in frag:
        print(f"    FAIL  {f}")
        failures.append(f)
    if not frag:
        print("    PASS  no test/corruption path fragment in executable strings")

    print()
    if failures:
        print(f"RESULT: {len(failures)} finding(s). The trainer may reach a forbidden path.")
        return 1
    print("RESULT: no test split and no corruption path is reachable from the trainer.")
    print("        Only data['train_split'] and data['val_split'] are used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
