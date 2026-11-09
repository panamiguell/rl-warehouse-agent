"""Entrena y compara agentes tabulares contra reglas fijas.

Ejemplo:
    python experiments/train.py --zones AB --episodes 2400 --eval-every 300

Salidas (todas en results/):
    models/<algoritmo>.npz          Q-tabla + hiperparámetros (TabularAgent.load la recupera)
    models/<algoritmo>_qtable.csv   Q-tabla legible: estado desglosado y Q por acción
    logs/training_<algoritmo>.csv   métricas por episodio de entrenamiento
    logs/learning_curve.csv         evaluación greedy periódica (mismas semillas para todos)
    logs/comparison.csv             evaluación final por episodio: agentes y reglas fijas
    logs/summary.json, summary.md   promedios finales
    figures/curvas_aprendizaje.png, figures/comparacion.png
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_warehouse.agents import ALGORITHMS, POLICIES, TabularAgent, run_policy  # noqa: E402
from rl_warehouse.env import WarehouseEnv  # noqa: E402

EVAL_SEED = 1_000_000          # semillas de evaluación disjuntas de las de entrenamiento
METRICS = ["reward", "arrived", "on_time", "expired", "pending", "saved_visits", "trips", "visits", "picked_lines"]


def load_config(path, zones):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if zones:
        config.setdefault("centro_distribucion", {})["zonas"] = list(zones)
    return config


def evaluate(agent, env, n):
    return [agent.episode(env, seed=EVAL_SEED + i, train=False) for i in range(n)]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    out = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    out["reward_std"] = float(np.std([r["reward"] for r in rows]))
    out["visits_per_line"] = out["visits"] / max(out["picked_lines"], 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/env_default.yaml"))
    parser.add_argument("--zones", default="", help="Sobrescribe las zonas, p. ej. AB para la versión de 2 zonas")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--episodes", type=int, default=2400)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon inicial")
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--time-unit", type=float, default=None,
                        help="Minutos por factor gamma (descuento semi-Markov). Por defecto, delta_t del config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--plot-only", action="store_true", help="Regenera las gráficas desde results/logs")
    args = parser.parse_args()
    if args.plot_only:
        logs = args.out / "logs"
        with (logs / "learning_curve.csv").open(encoding="utf-8") as stream:
            curve = [dict(r, episode=int(r["episode"]), reward=float(r["reward"])) for r in csv.DictReader(stream)]
        summary = json.loads((logs / "summary.json").read_text(encoding="utf-8"))["results"]
        plot(curve, summary, args.out / "figures")
        return
    if min(args.episodes, args.eval_episodes, args.eval_every) < 1:
        parser.error("Los números de episodios deben ser positivos")

    config = load_config(args.config, args.zones)
    gamma = config.get("episodio", {}).get("gamma", 0.99)
    time_unit = args.time_unit or config.get("episodio", {}).get("delta_t_min", 2)
    # epsilon llega a su mínimo al 80 % del entrenamiento.
    decay = (args.epsilon_min / args.epsilon) ** (1 / max(1, int(0.8 * args.episodes)))
    models, logs, figs = args.out / "models", args.out / "logs", args.out / "figures"
    curve, comparison, summary = [], [], {}

    for algorithm in args.algorithms:
        env = WarehouseEnv(config)
        agent = TabularAgent(env.action_space.n, algorithm, alpha=args.alpha, gamma=gamma, epsilon=args.epsilon,
                             epsilon_min=args.epsilon_min, epsilon_decay=decay, seed=args.seed, time_unit=time_unit)
        history, start = [], time.time()
        for i in range(args.episodes):
            result = agent.episode(env, seed=args.seed * 100_000 + i)
            history.append(dict(episode=i + 1, epsilon=round(agent.epsilon, 4), **result))
            if (i + 1) % args.eval_every == 0 or i + 1 == args.episodes:
                stats = summarize(evaluate(agent, env, args.eval_episodes))
                curve.append(dict(algorithm=algorithm, episode=i + 1, states=len(agent.q), **stats))
                print(json.dumps(dict(algorithm=algorithm, episode=i + 1, reward=round(stats["reward"], 1),
                                      expired=round(stats["expired"], 1), states=len(agent.q),
                                      minutes=round((time.time() - start) / 60, 1))), flush=True)
        final = evaluate(agent, env, args.eval_episodes)
        comparison += [dict(policy=algorithm, kind="rl", episode=i, **r) for i, r in enumerate(final)]
        summary[algorithm] = dict(kind="rl", states=len(agent.q), **summarize(final))
        agent.save(models / f"{algorithm}.npz")
        write_csv(models / f"{algorithm}_qtable.csv", agent.to_table(env))
        write_csv(logs / f"training_{algorithm}.csv", history)

    env = WarehouseEnv(config)
    for name, policy in POLICIES.items():
        rows = [run_policy(env, policy, EVAL_SEED + i) for i in range(args.eval_episodes)]
        comparison += [dict(policy=name, kind="regla", episode=i, steps=0, **r) for i, r in enumerate(rows)]
        summary[name] = dict(kind="regla", **summarize(rows))

    write_csv(logs / "learning_curve.csv", curve)
    write_csv(logs / "comparison.csv", comparison)
    meta = dict(config=str(args.config), zones=env.zones, episodes=args.episodes, eval_episodes=args.eval_episodes,
                alpha=args.alpha, gamma=gamma, time_unit=time_unit, epsilon=[args.epsilon, args.epsilon_min])
    (logs / "summary.json").write_text(json.dumps(dict(meta=meta, results=summary), indent=2), encoding="utf-8")
    write_markdown(logs / "summary.md", meta, summary)
    plot(curve, summary, figs)
    print((logs / "summary.md").read_text(encoding="utf-8"))


def write_markdown(path, meta, summary):
    lines = [f"# Resultados ({len(meta['zones'])} zonas, {meta['episodes']} episodios, "
             f"{meta['eval_episodes']} episodios de evaluación)", "",
             "| Política | Tipo | Recompensa | Desv. | A tiempo | Vencidas | Pendientes | Visitas ahorradas | Viajes | Visitas/línea |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["reward"]):
        lines.append(f"| {name} | {s['kind']} | {s['reward']:.1f} | {s['reward_std']:.1f} | {s['on_time']:.1f} | "
                     f"{s['expired']:.1f} | {s['pending']:.1f} | {s['saved_visits']:.1f} | {s['trips']:.1f} | "
                     f"{s['visits_per_line']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(curve, summary, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    figs.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#1F2933", "#6B7280", "#E5E7EB"
    colors = {"q_learning": "#2a78d6", "sarsa": "#eb6834", "monte_carlo": "#1baf7a"}
    labels = {"q_learning": "Q-learning", "sarsa": "SARSA", "monte_carlo": "Monte Carlo"}
    rules = {k: v for k, v in summary.items() if v["kind"] == "regla" and k != "esperar_siempre"}
    best_rule = max(rules, key=lambda k: rules[k]["reward"])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for algorithm in colors:
        pts = [c for c in curve if c["algorithm"] == algorithm]
        if not pts:
            continue
        x, y = [p["episode"] for p in pts], [p["reward"] for p in pts]
        ax.plot(x, y, color=colors[algorithm], lw=2, marker="o", ms=4, label=labels[algorithm])
        ax.annotate(labels[algorithm], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=ink)
    ax.axhline(rules[best_rule]["reward"], color=muted, lw=1.5, ls="--",
               label=f"Mejor regla fija ({best_rule})")
    ax.set_xlabel("Episodios de entrenamiento", color=ink)
    ax.set_ylabel("Recompensa media por turno (evaluación greedy)", color=ink)
    ax.set_title("Curvas de aprendizaje", loc="left", fontsize=13, color=ink, weight="bold")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "curvas_aprendizaje.png")
    plt.close(fig)

    # Las reglas degeneradas (no despachar nunca, o despachar orden por orden) siguen en summary.md,
    # pero aplastarían la escala de la gráfica.
    hidden = {"esperar_siempre", "urgente_siempre"}
    items = sorted(((k, v) for k, v in summary.items() if k not in hidden), key=lambda kv: kv[1]["reward"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharey=True)
    names = [labels.get(k, k.replace("_", " ")) for k, _ in items]
    for ax, key, title in [(axes[0], "reward", "Recompensa media por turno"),
                           (axes[1], "expired", "Órdenes vencidas por turno")]:
        values = [v[key] for _, v in items]
        bar_colors = [colors.get(k, "#9CA3AF") for k, _ in items]
        ax.barh(names, values, color=bar_colors, height=0.6)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:,.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=ink)
        ax.axvline(0, color=muted, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        lo, hi = min(values + [0]), max(values + [0])
        pad = 0.18 * (hi - lo or 1)
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        ax.set_title(title, loc="left", fontsize=11, color=ink, weight="bold")
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Agentes de RL (color) frente a reglas fijas (gris), mismas semillas de evaluación. "
                 "Sin las reglas degeneradas (ver summary.md)",
                 x=0.01, ha="left", fontsize=10, color=muted)
    fig.tight_layout()
    fig.savefig(figs / "comparacion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
