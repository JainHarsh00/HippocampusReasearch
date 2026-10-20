# Stochastic-Transition Experiment Audit Checklist

## Before the full run
- [ ] Confirm Python 3.10+ and install `requirements.txt`.
- [ ] Run the 56-configuration smoke test.
- [ ] Optionally run the 14-configuration terminal/success sanity test.
- [ ] Confirm each map's computed shortest path is 22 or 40 moves.
- [ ] Choose a new output folder; leave all earlier experiments unchanged.
- [ ] Save the package ZIP and keep its `SOURCE_SHA256.txt`.

## After the full run
- [ ] Confirm `completed_runs == planned_runs == 840` in `run_complete.json`.
- [ ] Check the `source_sha256` in `config.json` and `run_complete.json` matches the script's SHA-256.
- [ ] Confirm 840 unique maze × slip × bias × algorithm × seed conditions.
- [ ] Confirm each run has 200 training episodes and 10 evaluation episodes.
- [ ] Confirm all slip and corruption levels are present for all algorithms and maps.
- [ ] Check success is in [0,1] and steps are between 1 and 250.
- [ ] Check real-transition Q-updates use the actual observed transition and are not weighted.
- [ ] Check the reliability audit only scores predictions made before their outcomes are observed.
- [ ] Check the zero-probability forecast count and mean prequential NLL by pair.
- [ ] Inspect each maze separately before interpreting pooled summaries.
- [ ] Inspect actual planning updates for prioritized sweeping; the configured number is a maximum, not a guarantee.
- [ ] Report unfavorable results and limitations. Do not infer broad stochastic robustness from two fixed layouts.
