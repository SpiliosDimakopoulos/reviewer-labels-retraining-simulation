# Simulation code and results: reviewer labels, retraining and group disparity

Code and stored results for a study of public-sector risk models that are retrained on the decisions of the human reviewers who handle escalated cases. All data are synthetic. No personal data are used.

## Contents
- `simulation/sim/`: the simulator (population, model, reviewer, routing, one retraining run).
- `simulation/experiments.py`: experiments E1 to E14. Raw per-cycle output is in `simulation/results/raw/`.
- `simulation/analyze.py` and `simulation/analyze_extra.py`: aggregate the raw output into `simulation/results/results.json`.
- `simulation/figures.py`: the figures, written to `figures/`.
- `simulation/tests/test_sim.py`: automated tests of the simulator and of the first-step formula.
- `simulation/run_all.py`: runs experiments, analysis, figures and tests in order.

## Experiments
| Experiment | Question |
|---|---|
| E1, E12 | False alarms of monitoring rules in null deployments, by cohort size, proxy strength and capacity |
| E2, E2L | Step in the escalation-rate ratio over a grid of reviewer penalty and proxy strength; long horizon |
| E3 | Groups with different base rates |
| E4, E4C, E4D | Reviewer who follows the model: linear and mixture forms |
| E5 | Reviewer who cannot see the group and uses a visible correlate |
| E6 | Two attributes and their intersection |
| E7 | Training window and escalation capacity |
| E8 | First model trained on historic reviewer labels |
| E9 | Several proxies and correlated proxy noise |
| E10, E10b, E10c | Repairs applied at cycle 6, and label correction under a base-rate gap |
| E11 | The first-step formula when its assumptions are varied or broken |
| E13 | Logistic and boosted models with different proxy structures |
| E14 | Hold-out test of the formula on 24 random configurations with unused seeds |

## Reproduce
Requires Python 3.12 and the packages in `requirements.txt` (`pip install -r requirements.txt`).
`cd simulation && python run_all.py` reruns everything. Use `--skip-experiments` to rebuild the results file, the figures and the tests from the stored raw output. The experiments use four worker processes and take several minutes each.

`simulation/results/results.json` also holds blocks from earlier analysis passes (for example `E2_step_prediction` and `R2form_all`). They are superseded by `First_step_formula_variants`.

## Author and citation
Spilios Dimakopoulos (ORCID 0009-0001-4227-3627). This repository accompanies the manuscript "When Human Review Becomes the Vector: Reviewer Labels, Retraining and Group Disparity in Public-Sector AI" (under review). Until the article is published, please cite the repository: Dimakopoulos, S. (2026). Simulation code and results: reviewer labels, retraining and group disparity. https://github.com/SpiliosDimakopoulos/reviewer-labels-retraining-simulation

Licence: MIT (see `LICENSE`).
