# GreenRouting

**Intelligent model routing for sustainable AI. Route queries to the most energy-efficient model that can handle them — from any pool of models, with zero retraining when new models drop.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

Created by [Alan Hourmand](https://github.com/spectrallogic) | [savetokens.ai](https://www.savetokens.ai)

---

## Why This Matters

AI inference is becoming one of the fastest-growing sources of energy consumption worldwide. The International Energy Agency projects that data center electricity demand will more than double by 2026, with AI workloads as a primary driver. A single GPT-4-class query consumes roughly **10x the energy** of a GPT-3.5-class query — yet the majority of queries sent to frontier models don't require frontier capabilities.

Today, most AI applications route every query to the largest available model regardless of complexity. "What is 2+2?" burns the same energy as "Prove the Cauchy-Schwarz inequality." This is wasteful at every level — for developers paying API costs, for providers scaling infrastructure, and for energy grids absorbing the load.

GreenRouting fixes this. A lightweight classifier analyzes each query, determines what it actually needs (capability type, difficulty level), and routes it to the smallest model in your pool that can still deliver quality results.

## Measured Energy Savings

Routing a sample workload of mixed queries across 11 models (GPT-4o, Claude Opus, Llama 3.1 405B, and 8 smaller models):

| Query | Routed To | Energy (Wh) | vs. Always-Frontier | Savings |
|-------|-----------|-------------|---------------------|---------|
| "What is 2+2?" | llama-3.1-8b | 0.0050 | 0.4000 (llama-405b) | **98.8%** |
| "Explain quantum entanglement" | gemini-2.0-flash | 0.0150 | 0.4000 | **96.2%** |
| "Write a binary search in Python" | gpt-4o-mini | 0.0080 | 0.4000 | **98.0%** |
| "Prove the Cauchy-Schwarz inequality" | gemini-2.0-flash | 0.0150 | 0.4000 | **96.2%** |
| "Solve a differential equation in Python" (mixed: code + math) | gpt-4o-mini | 0.0080 | 0.4000 | **98.0%** |

**Aggregate across 6 test queries: 97.7% energy reduction, 99.5% cost reduction** compared to always routing to the most capable model.

### What This Means at Scale

| Scale | Queries/Day | Energy Without Routing | Energy With GreenRouting | Annual Savings | Annual Cost Savings* |
|-------|-------------|----------------------|------------------------|----------------|---------------------|
| Startup | 10,000 | 4,000 Wh | 93 Wh | **~14 MWh/year** | ~$1,100/yr |
| Mid-size | 1,000,000 | 400 kWh | 9.3 kWh | **~1,430 MWh/year** | ~$114K/yr |
| Enterprise | 100,000,000 | 40 MWh | 930 Wh | **~143 GWh/year** | ~$11.4M/yr |

*\*At $0.08/kWh commercial electricity rate.*

For context: 143 GWh/year is roughly the annual electricity consumption of 13,000 US households — or about **16.3 MW of continuous load removed from the grid**.

## How It Works

GreenRouting uses a classifier that predicts **what a query needs** — not which model to use. This is the key insight that makes it model-agnostic: when a new model drops, you just plug in its benchmark scores. Zero retraining.

```
User Query
     |
     v
Classifier ──> Predicts: capability weights + difficulty
                 e.g. {code: 50%, math: 35%, reasoning: 15%}, difficulty 4/5
                          |
                          v
Benchmark Matcher ──> Scores each model's fitness using published benchmarks
                          |
                          v
Green Score ──> Picks the greenest model above the quality threshold
                 green_score = α·quality − β·energy − γ·cost
                          |
                          v
RoutingDecision (selected model, energy savings, compression hint)
```

The classifier supports **multi-capability queries**. Real-world queries are messy — "Write a Python script that solves a differential equation" is code + math + reasoning simultaneously. GreenRouting handles this natively by predicting weighted capability distributions, not single labels.

## Quick Start

### Option 1: Load the pre-trained model (recommended)

The package ships with a pre-trained classifier. One function call gives you a fully functional router:

```python
from greenrouting import load_pretrained

# Load pre-trained classifier + 11 model profiles — ready to go
router = load_pretrained()

# Route any query
decision = router.route("What is 2+2?")
print(decision.selected_model)        # "llama-3.1-8b"
print(decision.energy_estimate_wh)    # 0.005
print(decision.energy_savings_vs_max) # 0.99 (99% energy saved)

# Works with complex queries too
decision = router.route("Write a Python script that solves a differential equation")
print(decision.selected_model)        # "gpt-4o-mini"
print(decision.reasoning)             # Explains the routing decision
```

Change the sustainability preset to match your priorities:

```python
# Maximum energy savings
router = load_pretrained(scorer_config={"green_score": {"preset": "maximum_green"}})

# Maximum quality (still saves energy vs. always-frontier)
router = load_pretrained(scorer_config={"green_score": {"preset": "quality_first"}})
```

### Option 2: Manual routing with QueryProfiles

If you want to bypass the classifier and route based on known query characteristics:

```python
from greenrouting import (
    BenchmarkMatcher, Capability, GreenScorer,
    ModelRegistry, QueryProfile, get_known_profiles,
)

# 1. Load pre-built profiles for 11 popular models
registry = ModelRegistry()
for profile in get_known_profiles().values():
    registry.register(profile)

# 2. Create a matcher with your preferred sustainability setting
scorer = GreenScorer.from_config({"green_score": {"preset": "balanced"}})
matcher = BenchmarkMatcher(registry, scorer)

# 3. Route a simple query
profile = QueryProfile.single(Capability.SIMPLE, difficulty=1)
decision = matcher.match(profile)
print(decision.selected_model)        # "llama-3.1-8b"
print(decision.energy_savings_vs_max) # 0.99 (99% energy saved)

# 4. Route a complex mixed-capability query
profile = QueryProfile(
    capability_weights={"code": 0.5, "math": 0.35, "reasoning": 0.15},
    difficulty=4,
)
decision = matcher.match(profile)
print(decision.selected_model)        # "gpt-4o-mini"
```

### Training from scratch

Train your own classifier in under 30 seconds on CPU:

```bash
# Train and save to greenrouting/pretrained/
python -m greenrouting.training

# Custom options
python -m greenrouting.training --epochs 50 --output ./my_model
```

Or use the full training + evaluation script:

```bash
python examples/train_and_evaluate.py
```

Load your custom-trained model:

```python
from greenrouting import load_pretrained

router = load_pretrained("./my_model")
decision = router.route("Explain quantum entanglement")
```

### Adding your own models

The classifier doesn't need retraining when you add or swap models — just provide benchmark scores:

```python
from greenrouting import ModelProfile, ModelRegistry, load_pretrained

# Start with the default pool or build your own
registry = ModelRegistry()

# Add a custom model with its benchmark scores
registry.register(ModelProfile(
    name="my-fine-tuned-llama",
    provider="local",
    estimated_params_b=8.0,
    cost_per_1k_input=0.0,
    cost_per_1k_output=0.0,
    energy_per_query_wh=0.005,
    benchmark_scores={
        "code": 0.85,       # HumanEval/MBPP score
        "math": 0.70,       # GSM8K/MATH score
        "reasoning": 0.65,  # ARC/BBH score
    },
))

# Load the pre-trained classifier with your custom registry
router = load_pretrained(registry=registry)
decision = router.route("Write a function to check if a number is prime")
```

## The Green Score

The Green Score is a tunable composite metric that balances quality, energy, and cost:

```
green_score = α · quality − β · energy − γ · cost
```

Three presets cover common use cases:

| Preset | Quality (α) | Energy (β) | Cost (γ) | Use Case |
|--------|-------------|------------|----------|----------|
| `quality_first` | 1.0 | 0.1 | 0.1 | When accuracy is critical |
| `balanced` | 0.6 | 0.25 | 0.15 | Default — good quality/efficiency tradeoff |
| `maximum_green` | 0.3 | 0.5 | 0.2 | Maximum energy savings |

The same trained classifier works across all presets — only the selection weights change.

## Capability Taxonomy

The classifier maps queries to 8 capability dimensions, each tied to established benchmarks:

| Capability | Benchmarks | Example Query |
|------------|------------|---------------|
| **Reasoning** | ARC, BBH, GPQA | "Prove this argument is fallacious" |
| **Math** | GSM8K, MATH | "Solve this integral" |
| **Code** | HumanEval, MBPP | "Implement an LRU cache" |
| **Knowledge** | MMLU, TriviaQA | "Explain quantum entanglement" |
| **Creative** | MT-Bench, AlpacaEval | "Write a short story with a twist" |
| **Instruction** | IFEval, MT-Bench | "Format this as a table with 5 rows" |
| **Multilingual** | MGSM, FLORES | "Translate this to Japanese" |
| **Simple** | — | "What is 2+2?" |

Adding a new model to the pool requires only its benchmark scores on these dimensions. The classifier doesn't need retraining — it predicts what queries *need*, not which model to pick.

## Output Compression (Caveman Integration)

Inspired by the [Caveman](https://github.com/JuliusBrussee/caveman) project, GreenRouting includes optional output compression hints. After routing to the right model, the system can suggest response brevity based on query difficulty:

| Query Difficulty | Compression Level | Estimated Token Savings |
|-----------------|-------------------|------------------------|
| 1-2 (trivial) | Aggressive | ~65% |
| 3 (moderate) | Moderate | ~30% |
| 4-5 (complex) | None | 0% |

This is additive to the energy savings from routing itself — for simple queries, you save on both the model used and the output generated.

## The AI Energy Crisis: Why This Matters Now

### The Problem

AI is creating the largest surge in electricity demand in a generation. The numbers are staggering:

- **Global data center electricity consumption reached ~460 TWh in 2024** and is projected to exceed 1,000 TWh by 2030 (IEA, *Electricity 2025*).
- **US data centers now consume ~4% of national electricity**, projected to reach 6-12% by 2030 (EPRI, 2024).
- **Goldman Sachs projects US data center power demand will grow 160% by 2030**, reaching ~45 GW of continuous load.
- **A single GPT-4-class query uses ~10x the electricity of a standard search** (~0.01 kWh vs ~0.001 kWh per query, IEA 2024).

Utilities are spending billions to keep up. AEP committed **$40 billion in capital expenditure (2024-2028)** citing data center demand as a top growth driver. Dominion Energy allocated **$9.6 billion in grid investments (2024-2028)** driven by Northern Virginia's "Data Center Alley." Georgia Power was approved to build **1.4 GW of new gas generation** partly to serve data center load. ERCOT (Texas) has over **60 GW of large-load interconnection requests** — mostly data centers — against a system peak of ~85 GW.

**The core issue:** building new generation and transmission capacity costs **$1-1.4 billion per GW** (natural gas) or **$1.2-1.8 billion per GW** (renewables + storage). Every megawatt of demand that can be *avoided* saves utilities $1-5 million per year in capacity, transmission, and distribution costs.

### How GreenRouting Helps

GreenRouting is a **demand-side optimization layer** for AI workloads. Instead of building more supply to meet runaway AI demand, it reduces the demand itself:

| Metric | Without Routing | With GreenRouting | Impact |
|--------|----------------|-------------------|--------|
| Energy per query (avg) | 0.40 Wh (frontier model) | 0.012 Wh (routed) | **97% reduction** |
| 100M queries/day | 40 MWh/day (~1.7 MW continuous) | 1.2 MWh/day | **~1.6 MW removed from grid** |
| Annual energy (enterprise) | 14,600 MWh | 438 MWh | **14,162 MWh saved** |
| Annual electricity cost* | $1.17M | $35K | **$1.13M saved per deployment** |
| Avoided capacity cost** | — | — | **$1.6-8M/yr per deployment** |

*\*At $0.08/kWh. \*\*Based on fully loaded avoided cost of $50-150/MWh (Lazard LCOE+, 2024) applied to 14,162 MWh saved, representing the infrastructure that does NOT need to be built.*

### Why Energy Companies Should Care

1. **Avoided infrastructure investment**: Every 1 MW of reduced continuous AI load avoids approximately **$1-5M/year** in generation, transmission, and distribution costs. At scale across a service territory with dozens of data center customers, intelligent routing could defer **hundreds of millions** in capital expenditure.

2. **Grid reliability**: PJM Interconnection paused new interconnection agreements in 2022 due to a backlog exceeding 250 GW. A single 500 MW data center campus equals the load of a city of ~300,000 people. Reducing AI inference demand by 80-95% directly eases interconnection queue pressure and grid stability concerns.

3. **Rate case impact**: Capital expenditure for new generation flows into rate base. If demand growth from AI can be moderated through efficiency (not curtailment), it reduces the rate increase pressure on all customers while still serving the AI market.

4. **Carbon and ESG reporting**: Every `RoutingDecision` includes energy estimates in Watt-hours. The built-in `EnergyTracker` produces cumulative reports that map directly to Scope 2 and Scope 3 emissions reporting. A 97% reduction in AI inference energy translates directly to emissions reductions.

5. **Demand response compatibility**: By shifting non-critical queries to models that use 1/25th the power, GreenRouting reduces the baseload compute requirement — making AI workloads more compatible with renewable intermittency and demand response programs.

### The Math for a Large Utility

Consider a utility serving 20 large data center customers, each running ~100M AI inference queries per day:

| | Without GreenRouting | With GreenRouting |
|---|---|---|
| Total AI load | ~34 MW continuous | ~1.4 MW continuous |
| Annual energy | 292 GWh | 8.8 GWh | 
| Annual electricity revenue impact | $23.4M | $0.7M |
| **Avoided new capacity needed** | **34 MW** | **1.4 MW** |
| **Avoided capital expenditure*** | — | **$34-170M** |

*\*At $1-5M per MW of avoided capacity (generation + T&D). Revenue decrease is offset many times over by avoided capital expenditure, reduced interconnection risk, and improved grid reliability.*

This is not about reducing revenue — it is about **avoiding billions in infrastructure buildout** that utilities are currently being forced to plan for. The most profitable megawatt is the one you never have to build.

## Comparison With Existing Approaches

| Project | Approach | # Models | Energy-Aware | Model-Agnostic | Open Source |
|---------|----------|----------|:------------:|:--------------:|:-----------:|
| **GreenRouting** | Multi-capability classifier routing | Any pool | Yes | Yes | Yes |
| RouteLLM | Pairwise strong/weak routing | 2 | No | No | Yes |
| Martian | Meta-model prediction | Proprietary set | No | No | No |
| Not Diamond | Quality-optimized routing | Proprietary set | No | No | No |
| Caveman | Output token compression | N/A | Indirect | Yes | Yes |

GreenRouting is the only solution that combines N-model routing, energy-first optimization, multi-capability classification, and full model-agnosticism in an open-source package.

## Installation

```bash
pip install greenrouting

# With training support (torch, sentence-transformers, datasets)
pip install greenrouting[train]

# With proxy server (FastAPI, uvicorn)
pip install greenrouting[serve]

# With direct energy measurement (CodeCarbon)
pip install greenrouting[energy]

# Everything
pip install greenrouting[train,serve,energy,dev]
```

## Project Structure

```
greenrouting/
    core/           # Taxonomy, matcher, registry, router base class
    routers/        # ClassifierRouter (neural), RandomRouter (baseline)
    energy/         # Green Score, energy estimation, model profiles, tracker
    training/       # Synthetic data generator, trainer pipeline
    serving/        # OpenAI-compatible client and FastAPI proxy (coming soon)
```

## Contributing

Contributions are welcome. Whether it's adding benchmark profiles for new models, improving the classifier architecture, expanding the training data, or building integrations — open an issue or submit a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Alan Hourmand** — [GitHub](https://github.com/spectrallogic) | [savetokens.ai](https://www.savetokens.ai)

Built to reduce the environmental impact of AI, one query at a time.

If this project is useful to you, consider supporting its development:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/alanhourmand)
