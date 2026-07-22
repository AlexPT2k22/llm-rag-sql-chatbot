import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ORDER = ["baseline", "hyde", "hybrid_30_70", "hybrid", "hybrid_70_30", "rerank"]
LABELS = {
    "baseline":     "MMR (baseline)",
    "hyde":         "HyDE",
    "hybrid":       "BM25+MMR 50/50",
    "hybrid_30_70": "BM25+MMR 30/70",
    "hybrid_70_30": "BM25+MMR 70/30",
    "rerank":       "MMR + reranker",
}
COLORS = {
    "baseline":     "#888888",
    "hyde":         "#1f77b4",
    "hybrid":       "#2ca02c",
    "hybrid_30_70": "#98df8a",
    "hybrid_70_30": "#006400",
    "rerank":       "#d62728",
}


def load_all():
    data = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "rag_compare_*.json")):
        name = os.path.basename(f).replace("rag_compare_", "").replace(".json", "")
        if name == "summary":
            continue
        with open(f, encoding="utf-8") as fp:
            data[name] = json.load(fp)
    return data


def categorize(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["cellar", "vinifica", "lote", "cuba", "ferment", "mosto",
                             "barric", "vindima", "engarrafa", "trasfeg", "rotul",
                             "granel", "enolog", "uva"]):
        return "Cellar"
    if any(k in q for k in ["parcela", "propriedade", "visita", "colheita", "acidente",
                             "exploraç", "rota", "exposiç", "solo", "coorden", "prag",
                             "cast", "área", "georref", "monitoriz"]):
        return "Field Monitoring"
    if any(k in q for k in ["pratic", "fitofarmac", "stock", "compra", "relator",
                             "equipament", "human", "custo", "orcament", "empreiteir",
                             "rega", "adubo", "tratament"]):
        return "Field Ops"
    return "Outro"


def plot_accuracy(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [n for n in ORDER if n in data]
    vals = [data[n]["accuracy_pct"] for n in names]
    bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS[n] for n in names], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Acerto (%)")
    ax.set_title("Acerto por estratégia de retrieval")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "1_accuracy.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def plot_no_info(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [n for n in ORDER if n in data]
    vals = [data[n]["no_info_pct"] for n in names]
    bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS[n] for n in names], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("NO_INFO (%)")
    ax.set_title("Taxa de \"Não tenho informação suficiente\"")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "2_no_info.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def plot_latency(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [n for n in ORDER if n in data]
    box_data = [[it["elapsed"] for it in data[n]["items"] if "elapsed" in it] for n in names]
    bp = ax.boxplot(box_data, labels=[LABELS[n] for n in names], patch_artist=True)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[n])
        patch.set_alpha(0.6)
    ax.set_ylabel("Latência (segundos)")
    ax.set_title("Latência por pergunta")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "3_latency.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def plot_per_category(data):
    cats = ["Cellar", "Field Monitoring", "Field Ops"]
    names = [n for n in ORDER if n in data]
    matrix = np.zeros((len(cats), len(names)))
    totals = np.zeros((len(cats), len(names)))

    for j, n in enumerate(names):
        for it in data[n]["items"]:
            cat = categorize(it["q"])
            if cat not in cats:
                continue
            i = cats.index(cat)
            totals[i][j] += 1
            if it.get("status") == "OK":
                matrix[i][j] += 1

    pct = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cats))
    w = 0.2
    for j, n in enumerate(names):
        ax.bar(x + (j - len(names) / 2 + 0.5) * w, pct[:, j], w,
               label=LABELS[n], color=COLORS[n])
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Acerto (%)")
    ax.set_title("Acerto por categoria de pergunta")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "4_per_category.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def plot_pareto(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    points = []
    for n in ORDER:
        if n not in data:
            continue
        x = data[n]["avg_time"]
        y = data[n]["accuracy_pct"]
        ax.scatter(x, y, s=200, color=COLORS[n], edgecolors="black", zorder=5)
        points.append((n, x, y))
    for n, x, y in points:
        if x < 3 and y >= 90:
            offset = {"hybrid_70_30": (-60, 15), "hybrid": (-60, -20),
                      "hybrid_30_70": (-60, -18)}.get(n, (10, 5))
            ax.annotate(LABELS[n], (x, y), xytext=offset, textcoords="offset points",
                        fontsize=9, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        else:
            ax.annotate(LABELS[n], (x, y), xytext=(10, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Latência média (s)")
    ax.set_ylabel("Acerto (%)")
    ax.set_title("Trade-off latência vs acerto")
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "5_pareto.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def main():
    data = load_all()
    if not data:
        print("Sem resultados em", RESULTS_DIR)
        print("Corre primeiro: python benchmarks/eval_rag_compare.py")
        return
    print(f"Estratégias encontradas: {list(data.keys())}")
    print("A gerar gráficos...")
    plot_accuracy(data)
    plot_no_info(data)
    plot_latency(data)
    plot_per_category(data)
    plot_pareto(data)
    print(f"\nFeito. Plots em {PLOTS_DIR}")


if __name__ == "__main__":
    main()
