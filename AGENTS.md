# AGENTS.md

## design/

Design documents and exploratory work. Ownership rules differ per subdirectory:

- `design/specs/` — specifications. **Human-only**: agents may read these but must not create, edit, or delete them.
- `design/adr/` — architecture decision records. Agents may **propose** new ADRs (with status `Proposed`); only a human changes the status to `Accepted` or `Rejected`. Agents must not edit ADRs once they are accepted or rejected.
- `design/experiments/` — exploratory notebooks and scratch data. **Human-only**: agents may read these for context but must not create, edit, or delete them.

## Sandbox

If a tool, package, or other dependency is missing from the sandbox, don't work around it silently: add an entry to `AGENT_REQUESTS.md` (what is missing, why it is needed, and any workaround used) for the human to install.

## CHANGELOG.md

Start every changelog entry with one of: Added, Changed, Deprecated, Removed, Fixed, Security (adapted from [Keep a Changelog](https://keepachangelog.com/)), e.g. ``- Added `fixname` to set ...``.
