"""World Models Eval demo Space: pre-computed sim-vs-dream videos side by side + trust-region chart.

Local scaffold (Type 1): runs with placeholder videos from space/precomputed/.
Type 2 fills precomputed/ with real (task, tier, checkpoint) videos and the final chart.

  python space/app.py
"""

import json
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
PRECOMPUTED = HERE / "precomputed"
MANIFEST = PRECOMPUTED / "manifest.json"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"tasks": ["placeholder_task"], "tiers": ["tiny"], "checkpoints": ["step_000200"]}


def lookup(task: str, tier: str, checkpoint: str) -> tuple[str | None, str | None]:
    sim = PRECOMPUTED / f"sim_{task}_{checkpoint}.mp4"
    dream = PRECOMPUTED / f"dream_{task}_{tier}_{checkpoint}.mp4"
    return (str(sim) if sim.exists() else None, str(dream) if dream.exists() else None)


def build() -> gr.Blocks:
    m = load_manifest()
    with gr.Blocks(title="World Models Eval") as demo:
        gr.Markdown(
            "# World Models Eval\n"
            "**How good must a world model be before you can trust it to evaluate robot policies?**\n\n"
            "Compare a policy rollout in the simulator (left) with the same policy dreaming inside "
            "a learned world model (right). Builds on WorldEval, WPE, Ctrl-World, and SIMPLER."
        )
        with gr.Row():
            task = gr.Dropdown(m["tasks"], label="Task", value=m["tasks"][0])
            tier = gr.Dropdown(m["tiers"], label="World-model tier", value=m["tiers"][0])
            ckpt = gr.Dropdown(m["checkpoints"], label="Policy checkpoint", value=m["checkpoints"][0])
        with gr.Row():
            sim_video = gr.Video(label="Simulator (ground truth)")
            dream_video = gr.Video(label="World-model dream")
        chart = PRECOMPUTED / "trust_region.png"
        if chart.exists():
            gr.Image(str(chart), label="Trust region: fidelity vs. ranking reliability")
        else:
            gr.Markdown("*Trust-region chart appears here after Type 2.*")
        for widget in (task, tier, ckpt):
            widget.change(lookup, [task, tier, ckpt], [sim_video, dream_video])
        demo.load(lookup, [task, tier, ckpt], [sim_video, dream_video])
    return demo


if __name__ == "__main__":
    build().launch()
