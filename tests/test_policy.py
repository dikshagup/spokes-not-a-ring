"""Repository policy: every experiments/ module is reachable from a repro.sh; no machine-specific paths."""
from __future__ import annotations

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "experiments")


def _py_files(root):
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def _figure_dirs():
    return sorted(d for d in os.listdir(EXP)
                  if os.path.isdir(os.path.join(EXP, d)) and d not in ("common", "results")
                  and not d.startswith("__"))


def _module_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _imports(path):
    src = open(path, encoding="utf-8").read()
    names = set()
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", src, re.M):
        names.add((m.group(1) or m.group(2)).split(".")[0])
    for m in re.finditer(r"^\s*from\s+[\w.]+\s+import\s+\(?([^\n)]+)", src, re.M):
        for n in m.group(1).split("#")[0].split(","):
            if n.strip():
                names.add(n.strip().split(" as ")[0].strip())
    return names


def test_every_figure_dir_has_a_repro_script():
    for d in _figure_dirs():
        assert os.path.exists(os.path.join(EXP, d, "repro.sh")), f"experiments/{d}/repro.sh is missing"


def test_every_experiment_module_is_reachable_from_a_repro_script():
    """Named by a repro.sh, or imported (transitively) by a module that is."""
    all_mods = {_module_name(p): p for p in _py_files(EXP) if _module_name(p) != "__init__"}
    reached = set()
    for d in _figure_dirs():
        sh = open(os.path.join(EXP, d, "repro.sh"), encoding="utf-8").read()
        for m in re.findall(r"([\w]+)\.py\b", sh):
            if m in all_mods:
                reached.add(m)
    frontier = list(reached)
    while frontier:
        m = frontier.pop()
        for dep in _imports(all_mods[m]):
            if dep in all_mods and dep not in reached:
                reached.add(dep); frontier.append(dep)
    unreached = sorted(set(all_mods) - reached)
    assert not unreached, (
        "modules under experiments/ that no figure uses (delete them, or wire them into a "
        f"repro.sh): {unreached}")


FORBIDDEN = [
    (r"/workspace/|/home/|/Users/", "machine-specific path"),
    (r"\bhf-cache\b", "machine-specific HF cache"),
]


@pytest.mark.parametrize("path", sorted(list(_py_files(EXP)) + list(_py_files(os.path.join(REPO, "src")))
                                        + [os.path.join(EXP, d, "repro.sh") for d in _figure_dirs()
                                           if os.path.exists(os.path.join(EXP, d, "repro.sh"))]))
def test_no_stale_references(path):
    src = open(path, encoding="utf-8").read()
    hits = [why for pat, why in FORBIDDEN if re.search(pat, src)]
    assert not hits, f"{os.path.relpath(path, REPO)}: {hits}"


def test_every_library_module_is_used_by_a_figure():
    """Every src/weekday_manifold module is imported by an experiment or library module."""
    src = os.path.join(REPO, "src", "weekday_manifold")
    mods = {}
    for d, _, fs in os.walk(src):
        for f in fs:
            if f.endswith(".py") and f != "__init__.py":
                rel = os.path.relpath(os.path.join(d, f), src)[:-3].replace(os.sep, ".")
                mods[rel] = os.path.join(d, f)
    pat = re.compile(r"weekday_manifold\.([\w.]+)")
    texts = [open(p, encoding="utf-8").read() for p in _py_files(EXP)]
    texts += [open(p, encoding="utf-8").read() for p in _py_files(src)]
    used = set()
    for t in texts:
        for m in pat.finditer(t):
            name = m.group(1)
            for k in mods:
                if name == k or name.startswith(k + "."):
                    used.add(k)
    unused = sorted(k for k in mods if k not in used)
    assert not unused, f"library modules no figure reaches: {unused}"
