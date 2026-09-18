# EnigmaForge — benchmark workflow
#
#   make test                  run the test suite
#   make grade OUT=runs/x      re-grade the cached corpus into a new report
#   make publish OUT=runs/x    publish a report directory to GitHub Pages
#   make publish-latest        publish the most recent complete report

PROVIDERS ?= providers-v3-models.json
OUT ?= runs/v3-models-7
CORPUS ?= benchmarks/v3-pilot-models

.PHONY: test grade publish publish-latest

test:
	python3 -m pytest tests/ -q

grade:
	set -a; . .env; set +a; \
	python3 -m enigmaforge.harness --providers $(PROVIDERS) \
	  --out $(OUT) --corpus $(CORPUS) --baselines --grade-only

publish:
	bash scripts/publish_report.sh $(OUT) "Publish benchmark report from $(OUT)"

publish-latest:
	@latest=""; \
	for d in $$(ls -dt runs/v3-models-* 2>/dev/null); do \
	  [ -f "$$d/report.html" ] && [ -f "$$d/results.json" ] && latest=$$d && break; \
	done; \
	[ -n "$$latest" ] \
	  && bash scripts/publish_report.sh $$latest "Publish benchmark report from $$latest" \
	  || { echo "no complete report directory under runs/"; exit 1; }
