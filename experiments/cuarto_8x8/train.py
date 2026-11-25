"""Entrena Q-learning y SARSA en el cuarto 8x8 y los compara con reglas fijas.

    python experiments/cuarto_8x8/train.py

Salidas en results/cuarto_8x8/:
    models/<algoritmo>.npz          Q-tabla + hiperparámetros (TabularAgent.load la recupera)
    models/<algoritmo>_qtable.csv   Q-tabla legible: estado desglosado y Q de cada acción
    logs/training_<algoritmo>.csv   promedios cada 1 000 episodios de entrenamiento
    logs/learning_curve.csv         evaluación greedy periódica en semillas de validación
    logs/comparison.csv             evaluación final por episodio (agentes y reglas, mismas semillas)
    logs/summary.json, summary.md   resultados finales, tiempos de ejecución y equipo usado
    figures/curvas_aprendizaje.png, figures/comparacion.png
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_gridworld import ACTION_NAMES, ALGORITHMS, ObstacleGridEnv, TabularAgent  # noqa: E402
from rl_gridworld.baselines import POLICIES, run_policy  # noqa: E402

# Semillas disjuntas: entrenamiento (0..N), validación y prueba.
VALIDATION_SEED = 500_000
TEST_SEED = 1_000_000
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA", "borde": "Regla: por el borde",
          "escalera": "Regla: escalera por el centro",
          "aleatoria": "Aleatoria (sin entrenar)"}


def summarize(rows):
    n = len(rows)
    out = {}
    for key in ("success", "collision", "truncated"):
        p = float(np.mean([r[key] for r in rows]))
        out[key] = p
        out[f"{key}_ci95"] = 1.96 * np.sqrt(p * (1 - p) / n)
    out["reward"] = float(np.mean([r["reward"] for r in rows]))
    out["reward_ci95"] = 1.96 * float(np.std([r["reward"] for r in rows], ddof=1)) / np.sqrt(n)
    ok = [r["steps"] for r in rows if r["success"]]
    out["steps_success"] = float(np.mean(ok)) if ok else float("nan")
    out["steps"] = float(np.mean([r["steps"] for r in rows]))
    out["episodes"] = n
    return out


def system_info():
    """Equipo y software en el que se corre el entrenamiento (para el reporte)."""
    import gymnasium
    import matplotlib

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            cpu = next(line.split(":", 1)[1].strip() for line in stream if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        ram = None
    system = f"{platform.system()} {platform.release()}"
    try:
        with open("/etc/os-release", encoding="utf-8") as stream:
            system = next(line.split("=", 1)[1].strip().strip('"') for line in stream if line.startswith("PRETTY_NAME"))
    except (OSError, StopIteration):
        pass
    return dict(cpu=cpu, cores=os.cpu_count(), ram_gb=ram, gpu="no se usa", os=system,
                python=platform.python_version(), numpy=np.__version__, gymnasium=gymnasium.__version__,
                matplotlib=matplotlib.__version__, pyyaml=yaml.__version__)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/cuarto_8x8.yaml"))
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--episodes", type=int, default=None, help="Por defecto, el del config")
    parser.add_argument("--eval-every", type=int, default=None, help="Por defecto, episodios / 10")
    parser.add_argument("--validation-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=2_000)
    parser.add_argument("--out", type=Path, default=None, help="Por defecto, la carpeta del config (salida)")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    args.out = args.out or ROOT / config.get("salida", "results/cuarto_8x8")
    logs, models, figs = args.out / "logs", args.out / "models", args.out / "figures"
    if args.plot_only:
        summary = json.loads((logs / "summary.json").read_text(encoding="utf-8"))
        with (logs / "learning_curve.csv").open(encoding="utf-8") as stream:
            curve = [{k: (v if k == "algorithm" else float(v)) for k, v in r.items()} for r in csv.DictReader(stream)]
        plot(curve, summary["results"], figs)
        return

    hp = config.get("entrenamiento", {})
    args.episodes = args.episodes or hp.get("episodios", 100_000)
    args.eval_every = args.eval_every or max(1, args.episodes // 10)
    env = ObstacleGridEnv(config.get("ambiente", {}))
    eps0, eps_min = hp.get("epsilon_inicial", 1.0), hp.get("epsilon_minimo", 0.05)
    decay = (eps_min / eps0) ** (1 / max(1, int(0.8 * args.episodes)))
    curve, comparison, summary = [], [], {}
    run_start = time.perf_counter()

    for algorithm in args.algorithms:
        agent = TabularAgent(3, algorithm, alpha=hp.get("alpha_minima", 0.01), gamma=hp.get("gamma", 0.99),
                             epsilon=eps0, epsilon_min=eps_min, epsilon_decay=decay, seed=hp.get("semilla", 0),
                             alpha_power=hp.get("alpha_potencia", 0.7))
        best = dict(success=-1.0, episode=0, q=None)
        history, window, start = [], [], time.perf_counter()
        train_time = 0.0                                # solo los episodios de entrenamiento

        def checkpoint(episode):
            stats = summarize([agent.episode(env, seed=VALIDATION_SEED + i, train=False)
                               for i in range(args.validation_episodes)])
            curve.append(dict(algorithm=algorithm, episode=episode, states=len(agent.q), **stats))
            if episode > 0 and stats["success"] > best["success"]:
                best.update(success=stats["success"], episode=episode, q={k: v.copy() for k, v in agent.q.items()})
            print(json.dumps(dict(algorithm=algorithm, episode=episode, success=round(stats["success"], 3),
                                  collision=round(stats["collision"], 3), states=len(agent.q),
                                  minutes=round((time.perf_counter() - start) / 60, 1))), flush=True)

        checkpoint(0)
        for i in range(args.episodes):
            tic = time.perf_counter()
            window.append(agent.episode(env, seed=i))
            train_time += time.perf_counter() - tic
            if (i + 1) % 1000 == 0:
                history.append(dict(episode=i + 1, epsilon=round(agent.epsilon, 4),
                                    **{k: round(v, 4) for k, v in summarize(window).items()
                                       if not k.endswith("ci95")}))
                window = []
            if (i + 1) % args.eval_every == 0:
                checkpoint(i + 1)
        total_time = time.perf_counter() - start         # entrenamiento + validaciones periódicas
        agent.q.clear()
        agent.q.update(best["q"])
        tic = time.perf_counter()
        rows = [agent.episode(env, seed=TEST_SEED + i, train=False) for i in range(args.test_episodes)]
        test_time = time.perf_counter() - tic
        comparison += [dict(policy=algorithm, episode=i, **r) for i, r in enumerate(rows)]
        summary[algorithm] = dict(kind="rl", checkpoint=best["episode"], states=len(agent.q),
                                  train_seconds=round(train_time, 1), train_validation_seconds=round(total_time, 1),
                                  test_seconds=round(test_time, 1), **summarize(rows))
        agent.save(models / f"{algorithm}.npz")
        write_csv(models / f"{algorithm}_qtable.csv", agent.to_rows(ACTION_NAMES))
        write_csv(logs / f"training_{algorithm}.csv", history)

    for name, policy in POLICIES.items():
        rows = [run_policy(env, policy, TEST_SEED + i) for i in range(args.test_episodes)]
        comparison += [dict(policy=name, episode=i, **r) for i, r in enumerate(rows)]
        validation = [run_policy(env, policy, VALIDATION_SEED + i) for i in range(args.validation_episodes)]
        summary[name] = dict(kind="regla", validation_success=summarize(validation)["success"], **summarize(rows))

    write_csv(logs / "learning_curve.csv", curve)
    write_csv(logs / "comparison.csv", comparison)
    meta = dict(config=str(Path(args.config).resolve().relative_to(ROOT)), episodes=args.episodes, validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes, epsilon=[eps0, eps_min], **hp, ambiente=config.get("ambiente", {}),
                total_seconds=round(time.perf_counter() - run_start, 1), system=system_info())
    (logs / "summary.json").write_text(json.dumps(dict(meta=meta, results=summary), indent=2), encoding="utf-8")
    lines = [f"# Resultados en el cuarto 8x8 ({args.test_episodes} episodios de prueba)", "",
             "± es el intervalo de confianza del 95 %.", "",
             "| Política | Llega a la meta | Choca | Límite de pasos | Pasos (si llega) | Recompensa |",
             "|---|---|---|---|---|---|"]
    for key, s in sorted(summary.items(), key=lambda kv: -kv[1]["success"]):
        lines.append(f"| {LABELS[key]} | {100 * s['success']:.1f} % ± {100 * s['success_ci95']:.1f} | "
                     f"{100 * s['collision']:.1f} % | {100 * s['truncated']:.1f} % | {s['steps_success']:.1f} | "
                     f"{s['reward']:.1f} ± {s['reward_ci95']:.1f} |")
    (logs / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(curve, summary, figs)
    print("\n".join(lines))


def plot(curve, summary, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#1F2933", "#6B7280", "#E5E7EB"
    colors = {"q_learning": "#2a78d6", "sarsa": "#eb6834"}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ends = {}
    for algorithm, color in colors.items():
        pts = [c for c in curve if c["algorithm"] == algorithm]
        if not pts:
            continue
        x = [p["episode"] / 1000 for p in pts]
        y = [100 * p["success"] for p in pts]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=4, label=LABELS[algorithm])
        ends[algorithm] = (x[-1], y[-1])
    if len(ends) == 2:                                   # separa las etiquetas finales si se enciman
        (xa, ya), (xb, yb) = ends["q_learning"], ends["sarsa"]
        if abs(ya - yb) < 5:
            mid = (ya + yb) / 2
            ends["q_learning"], ends["sarsa"] = (xa, mid + 3 if ya >= yb else mid - 3), (xb, mid - 3 if ya >= yb else mid + 3)
    for algorithm, (x_end, y_end) in ends.items():
        ax.annotate(LABELS[algorithm], (x_end, y_end), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=ink)
    for rule, style in (("borde", "--"), ("escalera", ":")):
        if rule in summary:
            ax.axhline(100 * summary[rule]["validation_success"], color=muted, lw=1.5, ls=style, label=LABELS[rule])
    ax.set_xlabel("Episodios de entrenamiento (miles)", color=ink)
    ax.set_ylabel("Episodios que llegan a la meta (%)", color=ink)
    ax.set_title("Curvas de aprendizaje (validación, política greedy)", loc="left", fontsize=12, color=ink,
                 weight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "curvas_aprendizaje.png")
    plt.close(fig)

    order = [k for k in ("q_learning", "sarsa", "borde", "escalera", "aleatoria") if k in summary]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    y = np.arange(len(order))
    parts = [("success", "Llega a la meta", "#1baf7a"), ("collision", "Choca", "#C0392B"),
             ("truncated", "Sin terminar (corte de seguridad)", "#9CA3AF")]
    left = np.zeros(len(order))
    for key, label, color in parts:
        values = np.array([100 * summary[k][key] for k in order])
        ax.barh(y, values, left=left, color=color, height=0.6, label=label, edgecolor="white", linewidth=2)
        for yi, (l, v) in enumerate(zip(left, values)):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f} %", ha="center", va="center", fontsize=8.5, color="white",
                        weight="bold")
        left += values
    ax.set_yticks(y, [LABELS[k] for k in order], fontsize=8.5, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de los episodios de prueba", color=ink)
    ax.set_title("Resultado de cada episodio de prueba", loc="left", fontsize=12, color=ink, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(figs / "comparacion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
