# Cooperative Highway Merging for Connected Autonomous Vehicles — CARLA Study

Compares five decision conditions for a main-lane vehicle deciding how to
respond to a ramp vehicle's merge attempt: **egoistic** (no cooperation),
**rule-based** (fixed TTC-threshold yielding), **negotiation** (a formal
request/response utility model, risk-aware under uncertain HDV intent —
the study's core contribution), **negotiation_no_uncertainty** (the same
negotiation model with the deterministic point-estimate risk term — an
ablation isolating what uncertainty-awareness buys you), and an optional
**learned** (PPO) high-level policy.

### Reasoning under uncertain human-driven-vehicle (HDV) intent

Recent cooperative-merging literature (e.g. the 2025 CARLA study on
optimization-based merging for CAVs interacting with HDVs of uncertain
intentions) frames mixed-traffic merging around not knowing exactly how a
human-driven vehicle will behave. This study's negotiation model addresses
that directly without requiring a full joint/multi-vehicle optimization
controller: `negotiation_protocol.HDVUncertainty` lets collision risk be
computed as an expectation over plausible follower-vehicle behavior (a
Gaussian speed-noise term plus an explicit non-compliance probability —
the human driver closing the gap instead of holding it) rather than a
single deterministic point estimate. It reduces bit-for-bit to the
original deterministic model when uncertainty is zero, so the
`negotiation` vs `negotiation_no_uncertainty` conditions in
`experiment_grid.default_strategy_factories` are a clean ablation, and
`scenario.py`'s background traffic spawns a fraction of "impatient" HDVs
(shorter following distance, higher speed, occasional right-of-way
violations via CARLA's Traffic Manager) so the simulated environment
actually contains the kind of behavior the belief model is reasoning
about.

Every strategy shares the same low-level controller (`controller.py`) so
comparisons measure decision quality, not driving skill. See
`C:\Users\gowth\.claude\plans\breezy-puzzling-clock.md` for the full design
rationale and the earlier chat discussion for why each of these design
choices (negotiation formalization, string-effect metric, failure-mode
taxonomy, density x strategy grid, bootstrap CIs) was chosen to be
competitive against existing published work in this space.

## Repository layout

```
src/merge_sim/
  negotiation_protocol.py   # formal utility-based negotiation model + HDV-uncertainty-aware risk (Monte Carlo) — NO CARLA dependency, unit tested
  metrics.py                 # TTC, jerk, hard-braking, string-effect metrics — NO CARLA dependency
  failure_modes.py           # failure-mode taxonomy classifier — NO CARLA dependency
  carla_utils.py              # CARLA connection/spawn/query helpers — requires CARLA
  controller.py               # shared IDM + PID low-level controller — requires CARLA
  scenario.py                  # MergeScenario: spawns traffic, runs one episode — requires CARLA
  strategies/                  # egoistic, rule_based, negotiation, learned — requires CARLA (via controller.py)
  episode_runner.py            # runs one episode, computes metrics — requires CARLA
  experiment_grid.py           # resumable strategy x density x seed grid runner — requires CARLA
  rl_env.py                    # Gymnasium env for training the learned strategy — requires CARLA + gymnasium
src/analysis/
  stats.py    # bootstrap CIs, Kruskal-Wallis, pairwise Mann-Whitney — NO CARLA dependency
  plots.py    # violin plots, success-rate heatmap, failure-mode bars — NO CARLA dependency
  report.py   # generates results.md from raw_episodes.csv — NO CARLA dependency
notebooks/
  00_colab_setup.ipynb       # installs + launches CARLA headless on Colab, smoke test, merge-point calibration
  01_run_experiments.ipynb   # runs the experiment grid (resumable)
  02_analysis.ipynb          # stats + figures + report.md (no CARLA needed, can run anywhere)
tests/   # unit tests for every NO-CARLA-dependency module — run with `pytest tests/`
results/ # raw_episodes.csv (append-only), figures/, report.md land here
```

## Why some modules need CARLA and some don't

This machine has no GPU and no CARLA install, so `negotiation_protocol.py`,
`metrics.py`, `failure_modes.py`, and the entire `analysis/` package were
built to have **zero CARLA dependency** and are fully unit tested here
(`pytest tests/` — 33 tests, all passing). The CARLA-dependent modules
(`carla_utils.py`, `controller.py`, `scenario.py`, `strategies/*`,
`episode_runner.py`, `experiment_grid.py`, `rl_env.py`) are written against
the documented CARLA 0.9.15 Python API but have **not been executed
against a live server** — their first real test happens on Colab. They are
kept small and single-purpose specifically so any bug found there is easy
to localize.

## Running on Google Colab

1. Push this repo to GitHub (or upload it directly), then in Colab:
   ```
   !git clone <your-repo-url> /content/fag-project
   ```
2. Set the Colab runtime to a GPU instance (*Runtime > Change runtime type > GPU*).
3. Run `notebooks/00_colab_setup.ipynb` top to bottom. This downloads and
   launches the CARLA server headless, then smoke-tests the connection.
   **If the smoke test times out**, try the `-opengl` fallback cell —
   documented inline — before assuming something else is broken.
4. Run the "calibrate merge point" cell at the bottom of `00_...` and
   confirm the auto-detected on-ramp junction is the right one for your
   CARLA version's Town04. If not, hardcode the correct `carla.Transform`
   into `MANUAL_MERGE_POINT` in `src/merge_sim/scenario.py`.
5. Run `notebooks/01_run_experiments.ipynb`: a 1-seed smoke test first,
   then the full 20-seed grid. Safe to re-run after a disconnect — it
   skips any `(strategy, density, seed)` already in `results/raw_episodes.csv`.
6. Run `notebooks/02_analysis.ipynb` (on Colab or downloaded locally — no
   CARLA needed) to get the summary table, significance tests, figures,
   and `results/report.md`.

## Running the CARLA-independent parts right now, on this machine

```
pip install -r requirements.txt   # only the non-CARLA subset will actually be used here
pytest tests/ -q
```

## Known things you'll likely need to tune once on Colab

- `scenario.py`'s `MANUAL_MERGE_POINT` (see step 4 above) — auto-detected
  junction may not be the intended highway on-ramp for your CARLA version.
- `scenario.py`'s `RAMP_LATERAL_OFFSET_M` inside `_spawn_ramp_vehicle` —
  the lateral offset from the main lane to the on-ramp lane is
  town-geometry-dependent; verify the ramp vehicle actually spawns on the
  ramp (not in the grass or another lane) and adjust if not.
- `experiment_grid.DEFAULT_SEEDS` — start with the 20-seed default; extend
  to 30+ if `analysis/stats.py`'s bootstrap CIs are still wide after the
  first full run.
