# AGENTS.md

## design/

Design documents and exploratory work. Ownership rules differ per subdirectory:

- `design/specs/` — specifications. **Human-only**: agents may read these but must not create, edit, or delete them.
- `design/adr/` — architecture decision records. Agents may **propose** new ADRs (with status `Proposed`); only a human changes the status to `Accepted` or `Rejected`. Agents must not edit ADRs once they are accepted or rejected.
- `design/experiments/` — exploratory notebooks and scratch data. **Human-only**: agents may read these for context but must not create, edit, or delete them.
