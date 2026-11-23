# AGENTS.md

## design/

Design documents and exploratory work. Ownership rules differ per subdirectory:

- `design/specs/` — specifications. **Human-only**: agents may read these but must not create, edit, or delete them.
- `design/adr/` — architecture decision records. Agents may **propose** new ADRs (with status `Proposed`); only a human changes the status to `Accepted` or `Rejected`. Agents must not edit ADRs once they are accepted or rejected.
- `design/experiments/` — exploratory notebooks and scratch data. **Human-only**: agents may read these for context but must not create, edit, or delete them.

## Sandbox

If a tool, package, or other dependency is missing from the sandbox, don't work around it silently: add an entry to `AGENT_REQUESTS.md` (what is missing, why it is needed, and any workaround used) for the human to install.

## Plots

Use the defaults of the plotting library (e.g. matplotlib's colours, fonts, styles and layout engines); don't add custom styling such as colour palettes, themes, `rcParams` or hand-tuned margins and spines.

Format numbers on plots (values, axis and tick labels, annotations, titles) in English (US/UK) style: a comma as the thousands separator (digit grouping) and a period as the decimal separator, e.g. 1,234.5. Tick labels have no thousands separator by default in matplotlib; set one with e.g. `ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,g}'))`.

## CHANGELOG.md

Start every changelog entry with one of: Added, Changed, Deprecated, Removed, Fixed, Security (adapted from [Keep a Changelog](https://keepachangelog.com/)), e.g. ``- Added `fixname` to set ...``.
