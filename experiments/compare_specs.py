"""Compara la especificación entregada en la Entrega 1 con la especificación final.

Las recompensas de los dos esquemas no son comparables entre sí, así que la comparación usa
indicadores operativos medidos sobre la misma demanda (mismas semillas de prueba): órdenes
vencidas, órdenes pendientes al cierre, visitas por línea y WIP medio.

Requiere haber corrido antes:
    python experiments/train.py
    python experiments/train.py --config configs/env_entrega1.yaml --out results/entrega1
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA", "monte_carlo": "Monte Carlo",
          "lote_mas_urgente": "Regla: lote más urgente", "umbral_consolidacion": "Regla: umbral de consolidación",
          "lote_mas_consolidado": "Regla: lote más consolidado", "aleatoria": "Política aleatoria"}
KPIS = [("expired", "Órdenes vencidas por turno", "{:.1f}"),
        ("pending", "Órdenes pendientes al cierre", "{:.1f}"),
        ("visits_per_line", "Visitas por línea (menos es mejor)", "{:.3f}"),
        ("wip_mean", "WIP medio (líneas sin asignar)", "{:.1f}")]
SPECS = [("entrega1", "Especificación entregada (Entrega 1)", "#9CA3AF"),
         ("final", "Especificación final", "#2a78d6")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", type=Path, default=ROOT / "results/logs/summary.json")
    parser.add_argument("--entrega1", type=Path, default=ROOT / "results/entrega1/logs/summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    data = {"final": json.loads(args.final.read_text(encoding="utf-8")),
            "entrega1": json.loads(args.entrega1.read_text(encoding="utf-8"))}
    policies = [p for p in LABELS if all(p in d["results"] for d in data.values())]

    lines = ["# Especificación entregada frente a especificación final", "",
             f"Mismo escenario de demanda y mismas {data['final']['meta']['eval_episodes']} semillas de prueba. "
             "Promedios por turno; ± es el intervalo de confianza del 95 %.", "",
             "| Política | Especificación | Vencidas | A tiempo (%) | Pendientes al cierre | Visitas/línea | WIP medio | Viajes |",
             "|---|---|---|---|---|---|---|---|"]
    for policy in policies:
        for key, name, _ in SPECS:
            s = data[key]["results"][policy]
            lines.append(f"| {LABELS[policy]} | {name} | {s['expired']:.1f} ± {s['expired_ci95']:.1f} | "
                         f"{100 * s['on_time_rate']:.1f} | {s['pending']:.1f} | {s['visits_per_line']:.3f} | "
                         f"{s['wip_mean']:.1f} | {s['trips']:.1f} |")
    out_md = args.out / "logs/comparacion_especificaciones.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    plot(data, policies, args.out / "figures/comparacion_especificaciones.png")


def plot(data, policies, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, muted, grid = "#1F2933", "#6B7280", "#E5E7EB"
    shown = [p for p in policies if p != "aleatoria"]
    fig, axes = plt.subplots(1, len(KPIS), figsize=(14, 4.6), dpi=150, sharey=True)
    y = np.arange(len(shown))
    height = 0.36
    for ax, (kpi, title, fmt) in zip(axes, KPIS):
        top = 0
        for j, (key, name, color) in enumerate(SPECS):
            values = [data[key]["results"][p][kpi] for p in shown]
            top = max(top, max(values))
            offset = (j - 0.5) * (height + 0.04)
            ax.barh(y + offset, values, height=height, color=color, label=name)
            for yi, v in zip(y + offset, values):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7.5, color=ink)
        ax.set_xlim(0, top * 1.3)
        ax.set_title(title, loc="left", fontsize=10, color=ink, weight="bold")
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=muted, labelsize=8)
    axes[0].set_yticks(y, [LABELS[p] for p in shown], fontsize=9, color=ink)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
