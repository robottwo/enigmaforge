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

for f in report.html results.json; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f" >&2; exit 1; }
done

ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'git worktree remove --force "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

git worktree add --orphan -b gh-pages "$WORK" 2>/dev/null \
  || git worktree add "$WORK" gh-pages
cp "$SRC/report.html" "$WORK/index.html"
cp "$SRC/results.json" "$WORK/results.json"
cd "$WORK"
git add -A
if git diff --cached --quiet gh-pages 2>/dev/null; then
  echo "gh-pages already up to date"
else
  git commit -q -m "$MSG"
  git push origin gh-pages
  echo "published: https://robottwo.github.io/enigmaforge/"
fi
