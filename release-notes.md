# Release Notes -- v0.16.0

> Released: 2026-09-21

The turtle moves out. `TurtleND`, `Turtle3D` and `Vector3D` now come from the `turtlend` package, and WaveRider releases reach PyPI on their own.

## What changed

**One source for the turtles.** WaveRider and proteusPy each carried an identical copy of `turtleND.py`, `turtle3D.py` and `vector3D.py`. Both now depend on [`turtlend`](https://pypi.org/project/turtlend/) 0.1.0, a BSD-3-Clause package whose only dependency is NumPy. The three modules in this repo are re-exports, so `waverider.turtleND`, `waverider.turtle3D`, `waverider.vector3D` and `waverider.TurtleND` keep resolving and no caller changes. Before and after the switch, the full suite passes the same 359 tests, and a `ManifoldObserver` field over 80 points plus a 300-step `TurtleND` walk produce identical numbers to eight decimal places. The `ty` override that silenced five rules for `turtle3D.py` is gone; `turtlend` fixed the annotations it was hiding. The move also brings three `Turtle3D` fixes that nothing here called: `orient` left `Position` stale, `ResetTape` raised `TypeError`, and `orient_at_residue` raised `AttributeError`.

**Releases publish to PyPI.** `release.yml` used to build the wheel and create the GitHub Release and stop there, so every PyPI version was uploaded by hand and 0.15.0 never was. It now carries the fleet's Trusted Publishing job, which uploads the same files the GitHub Release holds, and a `workflow_dispatch` path that publishes an existing tag's assets for a release that missed PyPI.

**Dependencies current.** The `doc-kg` and `pycode-kg` tools left the dependency list under the fleet's tools-are-global rule; the `kg` extra is now `bench` and holds only `proteusPy`, which the canonical benchmarks import. `quiltwright` is floored at 0.15.0 and `ruff` at 0.15. Two CIFAR-10 claim-verification documents move to the private companion repository; they are audit notes, not documentation, and the +8.5 pp result they verify is unchanged.

## Upgrading

`pip install --upgrade waverider` pulls in `turtlend` automatically. Nothing changes for code that imports the turtles through `waverider`. Code that wants the turtles without the rest of the stack can now `pip install turtlend` and import from it directly.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
