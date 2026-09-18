# EnigmaForge benchmark reports

Published leaderboard: https://robottwo.github.io/enigmaforge/

Regenerated from the current v3 response corpus after each sweep:

    python3 -m enigmaforge.harness --providers providers-v3-models.json \
      --out runs/<new-dir> --baselines --grade-only   # re-grade from cache
    bash scripts/publish_report.sh runs/<new-dir>     # publish to Pages

`results-detailed.json` (raw responses, ~1 GB) is kept locally only.
