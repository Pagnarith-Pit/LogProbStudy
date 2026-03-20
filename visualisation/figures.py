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

MODEL_DISPLAY = {
    "google_gemma-3-12b-it":                    "Gemma-3B",
    "meta-llama_Llama-3.2-3B-Instruct":         "LLaMA-3.2",
    "microsoft_Phi-4-reasoning-plus":           "Phi-4\nReasoning",
    "openai_gpt-oss-20b":                       "GPT-OSS\n20B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507-FP8":     "Qwen3\n30B",
}


BOX_PALETTE = ["#c0392b", "#e67e22", "#27ae60"]  

# Ablation bar colours
CLR_BASELINE = "#7f8c8d"   
CLR_REGULAR  = "#2980b9"  
CLR_MISMATCH = "#c0392b"   

# Loading data as step 1
def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# def collect_boxplot_data(results_dir: Path) -> dict:
#    PLOT TOO BIG. REQUIRES NEW VISUAL
#    Maybe violin? Or strip plot with jitter? Or boxplot with fewer models/dimensions?
#     data = {}
#     for model_dir in sorted(results_dir.iterdir()):
#         if not model_dir.is_dir():
#             continue
#         regular_path = model_dir / "regular.json"
#         if not regular_path.exists():
#             continue

#         regular_data = load_json(regular_path)
#         model_key    = model_dir.name
#         data[model_key] = {}

#         for dim in GUIDANCE_DIMS:
#             groups = {"No": [], "To some extent": [], "Yes": []}
#             for conv in regular_data:
#                 for model_name, metrics in conv.get("models", {}).items():
#                     label = metrics.get(dim)
#                     lp    = metrics.get("avg_log_prob")
#                     if label in groups and lp is not None:
#                         groups[label].append(lp)
#             data[model_key][dim] = groups

#     return data

def normalize_within_conversation(records: list) -> list:
    scores = np.array([r["logprob"] for r in records])
    mean   = scores.mean()
    for r in records:
        r["norm_logprob"] = r["logprob"] - mean
    return records


def filter_outliers_iqr(values: list[float], whisker_width: float = 1.5) -> list[float]:
    if len(values) < 4:
        return values

    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return values

    lower = q1 - whisker_width * iqr
    upper = q3 + whisker_width * iqr
    filtered = arr[(arr >= lower) & (arr <= upper)]

    if len(filtered) < max(3, int(0.4 * len(arr))):
        return values

    return filtered.tolist()


def collect_violin_data(results_dir: Path) -> dict:
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
                records = []
                for metrics in conv.get("models", {}).values():
                    label = metrics.get(dim)
                    lp    = metrics.get("avg_log_prob")
                    if label in GUIDANCE_MAP and lp is not None:
                        records.append({"label": label, "logprob": lp})

                if not records:
                    continue

                normalize_within_conversation(records)

                for r in records:
                    groups[r["label"]].append(r["norm_logprob"])

            data[model_key][dim] = groups

    return data

def collect_ablation_data(results_dir: Path) -> dict:
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


def plot_violins(data: dict, output_path: Path):
    # Much better looking
    model_keys = [k for k in MODEL_DISPLAY if k in data]
    n_models   = len(model_keys)
    n_dims     = len(GUIDANCE_DIMS)

    fig, axes = plt.subplots(
        n_dims, n_models,
        figsize=(2.4 * n_models, 3.6 * n_dims),
        sharey="row",
        constrained_layout=False,
    )

    if n_dims == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]

    rng = np.random.default_rng(42)

    for row, dim in enumerate(GUIDANCE_DIMS):
        for col, model_key in enumerate(model_keys):
            ax        = axes[row, col]
            groups    = data[model_key][dim]
            plot_data = [filter_outliers_iqr(groups[g]) for g in GROUP_ORDER]
            positions = np.arange(len(GROUP_ORDER))

            # Violin
            parts = ax.violinplot(
                plot_data,
                positions=positions,
                widths=0.65,
                showmedians=True,
                showextrema=False,
            )
            for body, colour in zip(parts["bodies"], BOX_PALETTE):
                body.set_facecolor(colour)
                body.set_edgecolor("none")
                body.set_alpha(0.50)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.8)

            # Jittered strip
            for i, (vals, colour) in enumerate(zip(plot_data, BOX_PALETTE)):
                vals = np.array(vals)
                jitter = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter, vals,
                    color=colour, alpha=0.20, s=5,
                    zorder=3, linewidths=0,
                )

            # Mean marker (white dot with dark edge)
            for i, vals in enumerate(plot_data):
                if vals:
                    ax.scatter(
                        i, np.mean(vals),
                        color="white", s=30, zorder=4,
                        linewidths=0.9, edgecolors="0.3",
                    )

            # Reference line at zero
            ax.axhline(0, color="0.6", linewidth=0.6,
                       linestyle="--", zorder=1)

            ax.set_xticks(positions)
            ax.set_xticklabels(GROUP_LABELS, fontsize=7.5)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

            # Column header: model name (top row only)
            if row == 0:
                ax.set_title(
                    MODEL_DISPLAY.get(model_key, model_key),
                    fontsize=8.5, fontweight="bold", pad=5,
                )

            # Row label: dimension name (leftmost column only)
            if col == 0:
                ax.set_ylabel(DIM_LABELS[dim], fontsize=8.5, labelpad=6)

    # Reserve fixed margins so shared labels do not collide with ticks.
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.15, top=0.88,
                        wspace=0.18, hspace=0.22)

    fig.suptitle("Guidance Quality vs Normalised Log-Probability",
                 fontsize=11, y=0.97)

    fig.supylabel("Normalised log-probability", fontsize=12, x=0.03)
    fig.supxlabel("Expert guidance quality",    fontsize=12, y=0.05)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Figure 1 saved → {output_path}")

# Figure 2: Ablation bar chart
def plot_ablation(ablation: dict, output_path: Path):
    model_keys   = [k for k in MODEL_DISPLAY if k in ablation]
    # Keep wrapped display names to reduce x-label crowding.
    display_names = [MODEL_DISPLAY[k] for k in model_keys]

    reg_vals  = [ablation[k]["mean_regular"]  for k in model_keys]
    base_vals = [ablation[k]["mean_baseline"] for k in model_keys]
    mis_vals  = [ablation[k]["mean_mismatch"] for k in model_keys]

    n = len(model_keys)
    if n == 0:
        raise ValueError("No ablation data available for Figure 2.")

    x = np.arange(n)
    width = 0.34

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(7.2, 3.2),
        sharex=True,
        constrained_layout=False,
    )

    # Left: Regular vs baseline
    ax_left.bar(x - width / 2, reg_vals,  width,
                label="Regular", color=CLR_REGULAR, alpha=0.84, zorder=3)
    ax_left.bar(x + width / 2, base_vals, width,
                label="Baseline (no last response)",
                color=CLR_BASELINE, alpha=0.84, zorder=3)

    # Right: Regular vs mismatch
    ax_right.bar(x - width / 2, reg_vals, width,
                 label="Regular", color=CLR_REGULAR, alpha=0.84, zorder=3)
    ax_right.bar(x + width / 2, mis_vals, width,
                 label="Mismatch", color=CLR_MISMATCH, alpha=0.78, zorder=3)

    # Axis limits tuned per panel for readability
    all_left = reg_vals + base_vals
    left_min = min(all_left)
    left_max = max(all_left)
    left_pad = max((left_max - left_min) * 0.35, 0.04)
    ax_left.set_ylim(left_min - left_pad, left_max + left_pad)

    all_right = reg_vals + mis_vals
    right_min = min(all_right)
    right_max = max(all_right)
    right_pad = max((right_max - right_min) * 0.08, 0.06)
    ax_right.set_ylim(right_min - right_pad, right_max + right_pad)

    # Shared formatting
    for ax in (ax_left, ax_right):
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=8)
        ax.tick_params(axis="x", pad=3)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(18)
            lbl.set_rotation_mode("anchor")
            lbl.set_ha("right")
            lbl.set_linespacing(1.05)

    ax_left.set_title("Regular vs Baseline", fontsize=9.5, pad=8)
    ax_right.set_title("Regular vs Mismatch", fontsize=9.5, pad=8)

    fig.suptitle("Ablation Comparison by Model", fontsize=11, y=0.99)
    fig.supxlabel("Model", fontsize=9, y=0.05)
    fig.supylabel("Mean avg. log-probability", fontsize=9, x=0.02)

    # Single global legend keeps panel area uncluttered.
    legend_handles = [
        plt.Line2D([], [], color=CLR_REGULAR,  lw=8, alpha=0.84),
        plt.Line2D([], [], color=CLR_BASELINE, lw=8, alpha=0.84),
        plt.Line2D([], [], color=CLR_MISMATCH, lw=8, alpha=0.78),
    ]
    legend_labels = ["Regular", "Baseline (no last response)", "Mismatch"]
    fig.legend(legend_handles, legend_labels, ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.93), fontsize=8)

    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.31, top=0.77, wspace=0.25)

    sns.despine(fig=fig, top=True, right=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved → {output_path}")

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

    violin_data   = collect_violin_data(results_dir)
    ablation_data = collect_ablation_data(results_dir)

    plot_violins(violin_data, output_dir / "figure1_violins.pdf")
    plot_ablation(ablation_data, output_dir / "figure2_ablation.pdf")

    print("\nDone. Both figures saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()