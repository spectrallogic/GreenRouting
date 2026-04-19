---
title: GreenRouting Demo
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Route AI queries to the most energy-efficient model
---

# GreenRouting Demo

Live routing demo for [GreenRouting](https://github.com/spectrallogic/GreenRouting) — intelligent model routing for sustainable AI.

Type a query; the classifier predicts what capabilities it needs (code, math, reasoning, etc.) and routes it to the smallest model in an 11-model pool that can handle it. Adjust the quality dial to trade energy savings for capability.

No API keys required — all routing decisions are made by a local 22M-parameter classifier.

## Install locally

```bash
pip install greenrouting
python -m greenrouting.repl
```

Or in Python:

```python
from greenrouting import load_pretrained
router = load_pretrained(quality=0.5)
decision = router.route("What is 2+2?")
print(decision.selected_model, decision.energy_savings_vs_max)
```
