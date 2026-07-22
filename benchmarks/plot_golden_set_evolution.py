"""
Gráfico de evolução do golden set taxa de acerto por iteração de optimização.
Uso: python benchmarks/plot_golden_set_evolution.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

iterations = [
    {
        "label": "Baseline\n7B ReAct",
        "score": 13,
        "changes": "ReAct puro\nsem views",
        "color": "#d9534f",
    },
    {
        "label": "14B\n(referência)",
        "score": 37,
        "changes": "Modelo 14B\n≈150s/query",
        "color": "#f0ad4e",
    },
    {
        "label": "7B + Pipeline\n+ Views",
        "score": 37,
        "changes": "Pipeline controlado\n+ 3 views PostgreSQL",
        "color": "#5bc0de",
    },
    {
        "label": "7B + Materialized\nViews + BM25",
        "score": 63,
        "changes": "4 MVs + BM25+MMR\n+ view-first P2",
        "color": "#5cb85c",
    },
    {
        "label": "7B Final\n(30 perguntas)",
        "score": 77,
        "changes": "Few-shot dinâmico\nSchema Pruning",
        "color": "#2e7d32",
    },
    {
        "label": "7B Generalização\n(50 perguntas)",
        "score": 72,
        "changes": "+20 perguntas Cellar\nTeste de stress",
        "color": "#1565c0",
    },
]

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(iterations))
bars = ax.bar(
    x,
    [it["score"] for it in iterations],
    color=[it["color"] for it in iterations],
    width=0.55,
    edgecolor="white",
    linewidth=1.5,
    zorder=3,
)

for bar, it in zip(bars, iterations):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{it['score']}%",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold", color="#222222",
    )

for bar, it in zip(bars, iterations):
    h = bar.get_height()
    if h >= 30:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h / 2,
            it["changes"],
            ha="center", va="center",
            fontsize=7.5, color="white",
            linespacing=1.5,
        )
    elif h >= 15:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h / 2,
            it["changes"],
            ha="center", va="center",
            fontsize=6.5, color="white",
            linespacing=1.4,
        )

ax.set_xticks(x)
ax.set_xticklabels([it["label"] for it in iterations], fontsize=10)
ax.set_ylabel("Taxa de Acerto (%)", fontsize=11)
ax.set_title(
    "Evolução da Taxa de Acerto no Golden Set da AgriTech",
    fontsize=13, fontweight="bold", pad=15,
)
ax.set_ylim(0, 95)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

hw_patch = mpatches.Patch(color="none",
    label="Hardware: NVIDIA RTX 3070 · 8 GB VRAM · Modelo: Qwen2.5 7B")
ax.legend(handles=[hw_patch], loc="upper left", fontsize=9, framealpha=0.7)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "results", "golden_set_evolution.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Grafico guardado em {out}")
plt.show()
