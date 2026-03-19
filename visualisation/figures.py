import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────

GUIDANCE_MAP   = {"Yes": 1.0, "To some extent": 0.5, "No": 0.0}
GUIDANCE_DIMS  = ["providing_guidance", "actionability"]
DIM_LABELS     = {"providing_guidance": "Providing Guidance",
                  "actionability":      "Actionability"}
GROUP_ORDER    = ["No", "To some extent", "Yes"]
GROUP_LABELS   = ["No", "To some\nextent", "Yes"]   # wrapped for x-axis

# Clean display names — edit here to change how models appear in figures
MODEL_DISPLAY = {
    "google_gemma-3-12b-it":                    "Gemma-3\n12B",
    "meta-llama_Llama-3.2-3B-Instruct":         "LLaMA-3.2\n3B",
    "microsoft_Phi-4-reasoning-plus":           "Phi-4\nReasoning+",
    "openai_gpt-oss-20b":                       "GPT-OSS\n20B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507-FP8":     "Qwen3\n30B",
}

# Palette: three muted colours for No / To some extent / Yes
BOX_PALETTE = ["#c0392b", "#e67e22", "#27ae60"]   # red, amber, green

# Ablation bar colours
CLR_BASELINE = "#7f8c8d"   # muted gray  — no-last-response
CLR_REGULAR  = "#2980b9"   # blue        — regular
CLR_MISMATCH = "#c0392b"   # red         — mismatch


# ── Data loading helpers ────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_boxplot_data(results_dir: Path) -> dict:
    """
    Returns nested dict:
        data[model_key][dimension] = {"No": [...], "To some extent": [...], "Yes": [...]}
    log-prob values (avg_log_prob) grouped by guidance label.
    """
    data = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        regular_path = model_dir / "regular.json"
        if not regular_path.exists():
            continue

        regular_data = load_json(regular_path)
        model_key    = model_dir.name
        data[model_key] = {}

        for dim in GUIDANCE_DIMS:
            groups = {"No": [], "To some extent": [], "Yes": []}
            for conv in regular_data:
                for model_name, metrics in conv.get("models", {}).items():
                    label = metrics.get(dim)
                    lp    = metrics.get("avg_log_prob")
                    if label in groups and lp is not None:
                        groups[label].append(lp)
            data[model_key][dim] = groups

    return data


def collect_ablation_data(results_dir: Path) -> dict:
    """
    Returns dict keyed by model_key:
        {
            "mean_regular":  float,
            "mean_baseline": float,
            "mean_mismatch": float,
        }
    Averages are taken over all conversations (dimension-agnostic,
    since ablation scores are the same across dimensions).
    """
    ablation = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        reg_path  = model_dir / "regular.json"
        base_path = model_dir / "no_last_response.json"
        mis_path  = model_dir / "mismatch.json"
        if not (reg_path.exists() and base_path.exists() and mis_path.exists()):
            continue

        regular_data  = load_json(reg_path)
        baseline_data = load_json(base_path)
        mismatch_data = load_json(mis_path)

        # Regular: mean avg_log_prob across all (conv, model) pairs
        reg_scores = [
            metrics.get("avg_log_prob")
            for conv in regular_data
            for metrics in conv.get("models", {}).values()
            if metrics.get("avg_log_prob") is not None
        ]

        # Baseline: one score per conversation (all models identical)
        base_scores = [
            next(iter(conv["models"].values())).get("avg_log_prob")
            for conv in baseline_data
            if conv.get("models")
        ]
        base_scores = [s for s in base_scores if s is not None]

        # Mismatch: mean avg_log_prob across all (conv, model) pairs
        mis_scores = [
            metrics.get("avg_log_prob")
            for conv in mismatch_data
            for metrics in conv.get("models", {}).values()
            if metrics.get("avg_log_prob") is not None
        ]

        ablation[model_dir.name] = {
            "mean_regular":  float(np.mean(reg_scores))  if reg_scores  else np.nan,
            "mean_baseline": float(np.mean(base_scores)) if base_scores else np.nan,
            "mean_mismatch": float(np.mean(mis_scores))  if mis_scores  else np.nan,
        }

    return ablation


# ── Figure 1: Box plots ─────────────────────────────────────────────────────

def plot_boxplots(data: dict, output_path: Path):
    """
    5-row × 2-column grid.
    Rows = models, columns = guidance dimensions.
    Each cell: three side-by-side box plots (No / To some extent / Yes).
    """
    model_keys = [k for k in MODEL_DISPLAY if k in data]
    n_models   = len(model_keys)
    n_dims     = len(GUIDANCE_DIMS)

    fig, axes = plt.subplots(
        n_models, n_dims,
        figsize=(7, 2.6 * n_models),
        sharey="row",
        constrained_layout=True,
    )

    # Ensure axes is always 2-D
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        for col, dim in enumerate(GUIDANCE_DIMS):
            ax      = axes[row, col]
            groups  = data[model_key][dim]

            plot_data   = [groups[g] for g in GROUP_ORDER]
            positions   = np.arange(len(GROUP_ORDER))

            bp = ax.boxplot(
                plot_data,
                positions=positions,
                patch_artist=True,
                widths=0.55,
                showfliers=True,
                flierprops=dict(marker="o", markersize=2.5,
                                linestyle="none", alpha=0.35),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                boxprops=dict(linewidth=0.8),
            )

            for patch, colour in zip(bp["boxes"], BOX_PALETTE):
                patch.set_facecolor(colour)
                patch.set_alpha(0.75)

            # Overlay individual data points (strip)
            for i, (group, colour) in enumerate(zip(GROUP_ORDER, BOX_PALETTE)):
                vals = groups[group]
                jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter, vals,
                    color=colour, alpha=0.18, s=5, zorder=2, linewidths=0,
                )

            # Labels
            ax.set_xticks(positions)
            ax.set_xticklabels(GROUP_LABELS, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            # Column header (top row only)
            if row == 0:
                ax.set_title(DIM_LABELS[dim], fontsize=10, fontweight="bold", pad=6)

            # Row label (left column only)
            if col == 0:
                display = MODEL_DISPLAY.get(model_key, model_key)
                ax.set_ylabel(display.replace("\n", " "), fontsize=8, labelpad=6)

    fig.supylabel("Avg. log-probability", fontsize=9, x=0.01)
    fig.supxlabel("Expert guidance quality", fontsize=9, y=0.01)

    sns.despine(fig=fig, top=True, right=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved → {output_path}")


# ── Figure 2: Ablation (broken y-axis) ─────────────────────────────────────

def plot_ablation(ablation: dict, output_path: Path):
    """
    Broken y-axis figure.
      Top panel:    regular vs baseline (zoomed in, small differences)
      Bottom panel: regular vs mismatch (full range)

    One group of 2-bar clusters per model. The break is indicated by
    diagonal tick marks on the shared spine.
    """
    model_keys   = [k for k in MODEL_DISPLAY if k in ablation]
    display_names = [MODEL_DISPLAY[k] for k in model_keys]

    reg_vals  = [ablation[k]["mean_regular"]  for k in model_keys]
    base_vals = [ablation[k]["mean_baseline"] for k in model_keys]
    mis_vals  = [ablation[k]["mean_mismatch"] for k in model_keys]

    n      = len(model_keys)
    x      = np.arange(n)
    width  = 0.26

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 1.6]},
        constrained_layout=True,
    )

    # ── Top panel: regular vs baseline ──
    for ax in (ax_top, ax_bot):
        ax.bar(x - width / 2, reg_vals,  width, label="Regular",     color=CLR_REGULAR,  alpha=0.82, zorder=3)
        ax.bar(x + width / 2, base_vals, width, label="Baseline\n(no last response)", color=CLR_BASELINE, alpha=0.82, zorder=3)

    # Mismatch bars only in bottom panel
    ax_bot.bar(x, mis_vals, width * 1.1,
               label="Mismatch", color=CLR_MISMATCH, alpha=0.75,
               zorder=2, bottom=0)

    # ── Y-axis limits ──
    # Determine a tight range for the top panel (reg vs baseline)
    all_rb   = reg_vals + base_vals
    rb_min   = min(all_rb)
    rb_max   = max(all_rb)
    rb_pad   = max(abs(rb_max - rb_min) * 0.5, 0.05)
    ax_top.set_ylim(rb_min - rb_pad, rb_max + rb_pad)

    # Bottom panel: cover mismatch down to its min
    bot_min  = min(mis_vals) * 1.08
    bot_max  = rb_max + rb_pad   # same top as ax_top so bars align visually
    ax_bot.set_ylim(bot_min, bot_max)

    # Hide the inner spines to create the break effect
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)

    # ── Diagonal break markers ──
    d       = 0.012   # size of diagonal tick
    kwargs  = dict(transform=fig.transFigure, color="k",
                   linewidth=0.8, clip_on=False)
    # Get the y positions of the break in figure coordinates
    pos_top = ax_top.get_position()
    pos_bot = ax_bot.get_position()
    y_break_top = pos_top.y0
    y_break_bot = pos_bot.y1

    for x_frac in [pos_top.x0 + 0.01, pos_top.x1 - 0.01]:
        fig.add_artist(
            plt.Line2D([x_frac - d, x_frac + d],
                       [y_break_top - d * 0.6, y_break_top + d * 0.6], **kwargs)
        )
        fig.add_artist(
            plt.Line2D([x_frac - d, x_frac + d],
                       [y_break_bot - d * 0.6, y_break_bot + d * 0.6], **kwargs)
        )

    # ── Axes formatting ──
    for ax in (ax_top, ax_bot):
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    ax_top.spines["top"].set_visible(False)

    # Shared y-label
    fig.supylabel("Mean avg. log-probability", fontsize=9, x=0.01)

    # Panel annotations
    ax_top.set_title("Regular vs. baseline (no-last-response)", fontsize=9,
                     loc="left", pad=4)
    ax_bot.set_title("Regular vs. mismatch", fontsize=9,
                     loc="left", pad=4)

    # Legend (top panel only)
    ax_top.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax_bot.legend(fontsize=8, frameon=False, ncol=1, loc="lower right")

    sns.despine(fig=fig, top=True, right=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved → {output_path}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument(
        "--results-dir", required=True,
        help="Root directory containing one subdirectory per model."
    )
    parser.add_argument(
        "--output-dir", default="./figures",
        help="Directory where figures will be saved (default: ./figures)."
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    boxplot_data  = collect_boxplot_data(results_dir)
    ablation_data = collect_ablation_data(results_dir)

    print(f"Found {len(boxplot_data)} model(s): {list(boxplot_data.keys())}")

    print("\nGenerating Figure 1 (box plots)...")
    plot_boxplots(boxplot_data, output_dir / "figure1_boxplots.pdf")

    print("\nGenerating Figure 2 (ablation)...")
    plot_ablation(ablation_data, output_dir / "figure2_ablation.pdf")

    print("\nDone. Both figures saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()