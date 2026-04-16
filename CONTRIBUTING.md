# Contributing to GreenRouting

Thanks for your interest in contributing to GreenRouting. Every contribution helps reduce the environmental impact of AI.

## Getting Started

### Setup

```bash
git clone https://github.com/spectrallogic/GreenRouting.git
cd GreenRouting
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,train]"
```

### Run Tests

```bash
python -m pytest tests/ -v
```

All tests must pass before submitting a PR.

## How to Contribute

### Adding a New Model Profile (Most Common)

This is the easiest and most impactful contribution. When a new model is released, adding its benchmark scores lets GreenRouting route to it immediately — no retraining needed.

1. Open `greenrouting/energy/profiles.py`
2. Add a new `ModelProfile` entry with:
   - `name`: Short identifier (e.g. `"gpt-4.1"`)
   - `provider`: Provider name (e.g. `"openai"`, `"anthropic"`, `"google"`, `"meta"`)
   - `model_id`: The API model identifier
   - `estimated_params_b`: Parameter count in billions (if known)
   - `cost_per_1k_input` / `cost_per_1k_output`: Pricing in USD
   - `energy_per_query_wh`: Estimated energy per query in Watt-hours
   - `benchmark_scores`: Dict mapping capability names to 0-1 scores

Benchmark score keys: `reasoning`, `math`, `code`, `knowledge`, `creative`, `instruction`, `multilingual`, `simple`

Sources for benchmark scores:
- [LMSYS Chatbot Arena](https://chat.lmsys.org/?leaderboard)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- Model cards and published papers

3. Add a test in `tests/test_energy.py` to verify the profile loads correctly
4. Run `python -m pytest tests/ -v` to confirm

### Bug Fixes

1. Open an issue describing the bug (or find an existing one)
2. Fork the repo and create a branch: `git checkout -b fix/description`
3. Write a failing test that reproduces the bug
4. Fix the bug
5. Verify all tests pass
6. Submit a PR

### New Features

1. Open an issue to discuss the feature first — this saves everyone time
2. Fork the repo and create a branch: `git checkout -b feature/description`
3. Implement with tests
4. Update the README if it's user-facing
5. Submit a PR

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check .
ruff format .
```

- Target: Python 3.10+
- Line length: 100 characters
- Follow existing patterns in the codebase

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with tests
4. Run `python -m pytest tests/ -v` — all tests must pass
5. Run `ruff check .` — no lint errors
6. Submit a PR with a clear description of what and why
7. Respond to review feedback

## Architecture Overview

If you're contributing beyond model profiles, here's how the pieces fit:

```
User Query
  -> ClassifierRouter.classify_query()    # Sentence embedding + MLP head
  -> QueryProfile                         # Capability weights + difficulty
  -> BenchmarkMatcher.match()             # Scores models using benchmark data
  -> GreenScorer.select()                 # Picks greenest model above threshold
  -> RoutingDecision                      # Selected model + energy savings
```

- **Classifier** (`routers/classifier_router.py`): Frozen sentence-transformers encoder + trainable MLP head
- **Taxonomy** (`core/taxonomy.py`): 8 capability dimensions, weighted profiles
- **Matcher** (`core/matcher.py`): Maps query profiles to model fitness scores
- **Scorer** (`energy/green_score.py`): Composite metric balancing quality, energy, cost
- **Profiles** (`energy/profiles.py`): Model benchmark scores and energy data
- **Training** (`training/`): Synthetic data generation and training pipeline

## Questions?

Open an issue or reach out at [savetokens.ai](https://www.savetokens.ai).
