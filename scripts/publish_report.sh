#!/usr/bin/env bash
# Publish a benchmark report directory to GitHub Pages (gh-pages branch).
# Usage: bash scripts/publish_report.sh runs/<report-dir> [commit-message]
#
# Publishes report.html (as index.html) and results.json. results-detailed.json
# (~1 GB) is not published. Runs entirely in a temporary worktree, so the main
# working tree (and any live sweep writing into it) is never touched.
set -euo pipefail

SRC=${1:?usage: publish_report.sh runs/<report-dir> [commit-message]}
MSG=${2:-"Publish benchmark report from ${SRC}"}

# Optional analytics injection: if ./analytics.html exists (git-ignored), its
# contents are inserted into <head> of every published page. Clones without
# this file publish beacon-free HTML.
INJECT=""
if [ -f analytics.html ]; then
  INJECT=$(cat analytics.html)
  echo "injecting analytics snippet from ./analytics.html"
fi

for f in report.html results.json; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f" >&2; exit 1; }
done

ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'git worktree remove --force "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

git worktree add --orphan -b gh-pages "$WORK" 2>/dev/null \
  || { git fetch origin gh-pages -q \
       && git worktree add "$WORK" gh-pages \
       && (cd "$WORK" && git pull --ff-only origin gh-pages -q) \
       || git worktree add "$WORK" gh-pages; }
cp "$SRC/report.html" "$WORK/index.html"
cp "$SRC/results.json" "$WORK/results.json"
[ -f "$SRC/report-details.html" ] && cp "$SRC/report-details.html" "$WORK/"
if [ -n "$INJECT" ]; then
  for page in "$WORK"/index.html "$WORK"/report-details.html; do
    [ -f "$page" ] || continue
    python3 - "$page" "$INJECT" <<'PYEOF'
import sys
path, snippet = sys.argv[1], sys.argv[2]
html = open(path).read()
marker = "<head>"
assert marker in html, f"no <head> in {path}"
open(path, "w").write(html.replace(marker, marker + snippet, 1))
PYEOF
  done
fi
cd "$WORK"
git add -A
if git diff --cached --quiet gh-pages 2>/dev/null; then
  echo "gh-pages already up to date"
else
  git commit -q -m "$MSG"
  git push origin gh-pages
  echo "published: https://robottwo.github.io/enigmaforge/"
fi

# Dispatch the analytics-injection workflow on main (it cannot trigger from
# gh-pages pushes because that orphan branch has no workflow files).
if command -v gh >/dev/null 2>&1; then
  gh workflow run inject-analytics.yml \
    && echo "analytics injection dispatched" \
    || echo "WARNING: could not dispatch inject-analytics.yml (beacon missing until re-run)"
else
  echo "WARNING: gh CLI not found; run 'gh workflow run inject-analytics.yml' to inject the analytics beacon"
fi
