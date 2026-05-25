# Release Notes — v0.8.1

> Released: 2026-05-25

Citation patch. The Zenodo DOI is now minted and wired into `CITATION.cff`, the README header badge, the README Citation section, and the BibTeX entry. No code or benchmark changes.

### Changed
- **`CITATION.cff`** — `doi` field activated with the minted Zenodo identifier `10.5281/zenodo.20383651`.
- **`README.md`** — DOI badge (header + Citation section) wired to the live Zenodo concept-DOI badge (`zenodo.org/badge/1234120398.svg`); prose citation now resolves to `https://doi.org/10.5281/zenodo.20383651`; BibTeX entry replaces the placeholder `note` with a proper `doi` field.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
