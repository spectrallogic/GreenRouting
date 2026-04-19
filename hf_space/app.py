"""GreenRouting live demo — Hugging Face Space.

Routes queries to the most energy-efficient model in a pool of 11 LLMs
based on predicted capability needs. No API keys required — the classifier
runs locally, and routing decisions are returned without calling any model.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402

from greenrouting import (  # noqa: E402
    BenchmarkMatcher,
    GreenScorer,
    get_compression_hint,
    load_pretrained,
)

print("Loading pretrained classifier (one-time, ~20s)...")
ROUTER = load_pretrained()
print(f"Ready. {len(ROUTER.registry)} models in pool.")


def route(query: str, quality: float):
    if not query.strip():
        return "—", "—", "—", "—", "—", "—"

    ROUTER.scorer = GreenScorer.from_quality(quality)
    ROUTER.matcher = BenchmarkMatcher(ROUTER.registry, ROUTER.scorer)

    decision = ROUTER.route(query)
    profile = ROUTER.classify_query(query)
    hint = get_compression_hint(profile)

    caps = sorted(profile.capability_weights.items(), key=lambda x: x[1], reverse=True)
    cap_str = ", ".join(f"{c} {w:.0%}" for c, w in caps if w >= 0.05)

    return (
        decision.selected_model,
        f"{decision.energy_estimate_wh:.4f} Wh",
        f"{decision.energy_savings_vs_max * 100:.1f}%",
        f"${decision.cost_estimate:.5f}",
        f"{cap_str}  |  difficulty {profile.difficulty}/5",
        hint.level,
    )


with gr.Blocks(title="GreenRouting Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GreenRouting — Live Routing Demo

        Type any query. The classifier predicts what capabilities it needs (code, math, reasoning, etc.)
        and routes it to the smallest model in an 11-model pool that can handle it.

        Adjust the **quality dial** to trade efficiency for capability:
        - `0.0` → maximum energy savings (smallest capable model)
        - `0.5` → balanced (default)
        - `1.0` → maximum quality (smartest model available)
        """
    )

    with gr.Row():
        query = gr.Textbox(
            label="Query",
            placeholder="e.g. 'Write a Python function to sort a list'",
            lines=2,
            scale=4,
        )
        quality = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.1,
            label="Quality dial",
            scale=1,
        )

    btn = gr.Button("Route", variant="primary")

    with gr.Row():
        model_out = gr.Textbox(label="Selected model", interactive=False)
        energy_out = gr.Textbox(label="Energy per query", interactive=False)
        savings_out = gr.Textbox(label="Energy saved vs largest", interactive=False)

    with gr.Row():
        cost_out = gr.Textbox(label="Estimated cost", interactive=False)
        caps_out = gr.Textbox(label="Detected capabilities", interactive=False)
        compress_out = gr.Textbox(label="Compression hint", interactive=False)

    gr.Examples(
        examples=[
            ["What is 2+2?", 0.5],
            ["Write a Python quicksort function", 0.5],
            ["Prove the Cauchy-Schwarz inequality", 0.5],
            ["Translate 'good morning' to Japanese", 0.5],
            ["Analyze the trade-offs between monolithic and microservices architecture", 0.5],
            ["Write a haiku about autumn leaves", 0.0],
            ["What is 2+2?", 1.0],
            ["Explain quantum entanglement in one sentence", 0.5],
        ],
        inputs=[query, quality],
    )

    outputs = [model_out, energy_out, savings_out, cost_out, caps_out, compress_out]
    btn.click(route, inputs=[query, quality], outputs=outputs)
    query.submit(route, inputs=[query, quality], outputs=outputs)

    gr.Markdown(
        """
        ---
        **How it works**: a 22M-parameter classifier on top of `all-MiniLM-L6-v2` predicts capability
        weights + difficulty for each query, then a benchmark-aware matcher picks the model with the
        best Green Score: `alpha * quality - beta * energy - gamma * cost`.

        **Note on numbers**: energy values come from model-profile estimates in the package, not live
        measurements. The *ratio* between models is usually more meaningful than the absolute Wh.
        See the [methodology](https://github.com/spectrallogic/GreenRouting#how-it-works) on GitHub.

        **Install**: `pip install greenrouting` —
        [PyPI](https://pypi.org/project/greenrouting/) |
        [GitHub](https://github.com/spectrallogic/GreenRouting) |
        Built by [Alan Hourmand](https://github.com/spectrallogic) ([savetokens.ai](https://www.savetokens.ai))
        """
    )


if __name__ == "__main__":
    demo.launch()
