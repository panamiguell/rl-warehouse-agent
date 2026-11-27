"""Búsqueda de hiperparámetros para el cuarto 8x8 (se elige solo con semillas de validación, nunca de prueba).

    python experiments/cuarto_8x8/grid_search.py            # las dos etapas + confirmación (~1 h con 2 núcleos)
    python experiments/cuarto_8x8/grid_search.py --plot-only

Etapa 1, barrido de gamma (6 combinaciones): algoritmo x gamma {0,90  0,95  0,99}, con alfa por visitas (w = 0,7)
y epsilon mínimo 0,05. Con gamma bajo el agente gira en la salida hasta el corte de seguridad, y cada episodio
cuesta 1 000 pasos; para que quepa en el cómputo, en esta etapa el corte de entrenamiento es de 200 pasos. No
cambia lo que se aprende: en el corte no se trata el estado como terminal y se sigue estimando su valor.

Etapa 2, grid factorial (2 x 2 x 4 x 3 = 48 combinaciones), con el ambiente oficial (corte de 1 000 pasos):
    algoritmo      Q-learning, SARSA
    gamma          0,99  0,999
    tasa alfa      fija 0,1 | por visitas max(0,01, N^-w) con w = 0,5  0,7  0,9
    epsilon mínimo 0,01  0,05  0,10   (desde 1,0, decaimiento geométrico hasta el 80 % del entrenamiento)
Cada combinación se entrena 50 000 episodios.

Hipótesis, escritas antes de correr la búsqueda:
    H1 (gamma bajo) Con gamma 0,90 o 0,95 el agente se queda girando en la salida, que es segura: girar sin fin
       vale -1/(1-gamma) = -10 o -20, más que intentar llegar con cerca de 65 % de choques. Solo con gamma 0,99
       aprende a salir.
    H2 (gamma alto) Con gamma 0,999 también sale (girar sin fin vale -1 000), pero los valores son más grandes y
       ruidosos, así que aprende igual o un poco peor que con 0,99.
    H3 (alfa) Con alfa fija 0,1 los valores oscilan por el ruido de los choques y la política es menos estable
       entre puntos de control; la tasa por visitas converge mejor.
    H4 (epsilon) El epsilon mínimo casi no cambia la ruta final aprendida, porque la evaluación es greedy.
    H5 (algoritmo) Q-learning y SARSA llegan a resultados parecidos: el agente no ve los obstáculos y lo único que
       puede aprender es la ruta.

Selección:
    1. Cada combinación se evalúa cada 5 000 episodios en 500 episodios de validación y se guarda el mejor punto
       de control (como en train.py).
    2. Ese punto de control se vuelve a evaluar en 2 000 episodios de validación nuevos (semillas 600 000+), para
       no elegir por la suerte de los 500 primeros.
    3. Las 5 mejores combinaciones de la etapa 2 se repiten con 2 semillas de entrenamiento más (1 y 2). Gana la
       de mayor promedio en validación; en empate, la de mayor estabilidad (promedio de los 5 últimos puntos de
       control).
En la evaluación greedy, no llegar en 100 pasos cuenta como no llegar (la ruta aprendida tiene 16 pasos); así
las políticas que dan vueltas no cuestan 1 000 pasos por episodio.

Salidas en results/cuarto_8x8/grid/: etapa1_gamma.csv, etapa2_grid.csv, confirmacion.csv, seleccion.json y
figures/grid.png.
"""
import argparse
import csv
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_gridworld import ObstacleGridEnv, TabularAgent  # noqa: E402

VALIDATION_SEED = 500_000      # puntos de control (los mismos de train.py)
REVALIDATION_SEED = 600_000    # reevaluación del mejor punto de control
EVAL_STEPS = 100               # evaluación greedy: no llegar en 100 pasos es dar vueltas
ALGORITHMS = ("q_learning", "sarsa")
STAGE1_GAMMAS = (0.90, 0.95, 0.99)
STAGE1_CAP = 200
GAMMAS = (0.99, 0.999)
ALPHAS = ("fija", 0.5, 0.7, 0.9)        # "fija": alfa = 0,1; número: potencia w de la tasa por visitas
EPS_MINS = (0.01, 0.05, 0.10)
OUT = ROOT / "results/cuarto_8x8/grid"


def success_rate(agent, env, seed0, n):
    return float(np.mean([agent.episode(env, seed=seed0 + i, train=False)["success"] for i in range(n)]))


def greedy_route(agent):
    """Acciones de la política greedy sin obstáculos (I, D, A); termina en '...' si no llega en 64 pasos."""
    env = ObstacleGridEnv({"obstaculos": 0, "salvaguarda_pasos": 64})
    obs, info = env.reset(seed=0)
    letters = []
    while True:
        action = agent.act(obs, info["action_mask"], explore=False)
        letters.append("IDA"[action])
        obs, _, terminated, truncated, info = env.step(action)
        if terminated:
            return "".join(letters)
        if truncated:
            return "".join(letters[:12]) + "..."


def run(cfg):
    """Entrena una combinación y devuelve sus métricas de validación."""
    algorithm, gamma, alpha, eps_min, seed, episodes, train_cap = cfg
    ambiente = yaml.safe_load((ROOT / "configs/cuarto_8x8.yaml").read_text(encoding="utf-8")).get("ambiente", {})
    env = ObstacleGridEnv({**ambiente, **({"salvaguarda_pasos": train_cap} if train_cap else {})})
    eval_env = ObstacleGridEnv({**ambiente, "salvaguarda_pasos": EVAL_STEPS})
    decay = (eps_min / 1.0) ** (1 / max(1, int(0.8 * episodes)))
    fixed = alpha == "fija"
    agent = TabularAgent(3, algorithm, alpha=0.1 if fixed else 0.01, gamma=gamma, epsilon=1.0, epsilon_min=eps_min,
                         epsilon_decay=decay, seed=seed, alpha_power=None if fixed else float(alpha))
    every, checkpoints, best = episodes // 10, [], dict(success=-1.0, episode=0, q=None)
    steps, start = 0, time.perf_counter()
    for i in range(episodes):
        steps += agent.episode(env, seed=i)["steps"]
        if (i + 1) % every == 0:
            s = success_rate(agent, eval_env, VALIDATION_SEED, 500)
            checkpoints.append(s)
            if s > best["success"]:
                best.update(success=s, episode=i + 1, q={k: v.copy() for k, v in agent.q.items()})
    seconds = time.perf_counter() - start
    agent.q.clear()
    agent.q.update(best["q"])
    route = greedy_route(agent)
    loops = route.endswith("...")
    return dict(algorithm=algorithm, gamma=gamma, alpha=alpha, eps_min=eps_min, seed=seed,
                validation=round(success_rate(agent, eval_env, REVALIDATION_SEED, 2000), 4),
                best_checkpoint=round(best["success"], 4), best_episode=best["episode"],
                stability=round(float(np.mean(checkpoints[-5:])), 4),
                zero_checkpoints=int(sum(c == 0 for c in checkpoints)),
                route=route, route_steps="" if loops else len(route),
                turns="" if loops else sum(c != "A" for c in route),
                train_steps_per_episode=round(steps / episodes, 1), seconds=round(seconds, 1),
                checkpoints=" ".join(f"{c:.3f}" for c in checkpoints))


def key(r):
    return (r["algorithm"], float(r["gamma"]), str(r["alpha"]), float(r["eps_min"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_all(pool, configs, label):
    rows = []
    for row in pool.imap_unordered(run, configs):
        rows.append(row)
        print(f"[{label} {len(rows)}/{len(configs)}] {row['algorithm']} g={row['gamma']} a={row['alpha']} "
              f"e={row['eps_min']} s={row['seed']}: val={row['validation']:.3f} ruta={row['route']} "
              f"({row['seconds']:.0f} s)", flush=True)
    return sorted(rows, key=lambda r: (key(r), r["seed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(read_csv(OUT / "etapa2_grid.csv"))
        return

    start = time.perf_counter()
    stage1 = [(a, g, 0.7, 0.05, 0, args.episodes, STAGE1_CAP) for a, g in itertools.product(ALGORITHMS, STAGE1_GAMMAS)]
    stage2 = [(a, g, al, e, 0, args.episodes, None)
              for a, g, al, e in itertools.product(ALGORITHMS, GAMMAS, ALPHAS, EPS_MINS)]
    with Pool(args.workers) as pool:
        rows2 = run_all(pool, stage2, "etapa 2")
        write_csv(OUT / "etapa2_grid.csv", rows2)
        ranked = sorted(rows2, key=lambda r: (-r["validation"], -r["stability"]))[:args.top]
        extra = [(r["algorithm"], r["gamma"], r["alpha"], r["eps_min"], seed, args.episodes, None)
                 for r in ranked for seed in (1, 2)]
        confirm = run_all(pool, extra, "confirmación")
        rows1 = run_all(pool, stage1, "etapa 1")
        write_csv(OUT / "etapa1_gamma.csv", rows1)

    top_keys = {key(r) for r in ranked}
    confirm = sorted(confirm + [r for r in rows2 if key(r) in top_keys], key=lambda r: (key(r), r["seed"]))
    write_csv(OUT / "confirmacion.csv", confirm)
    groups = {}
    for r in confirm:
        groups.setdefault(key(r), []).append(r)
    scored = sorted(groups.items(), key=lambda kv: (-np.mean([r["validation"] for r in kv[1]]),
                                                    -np.mean([r["stability"] for r in kv[1]])))
    best_key, best_rows = scored[0]
    selection = dict(
        algorithm=best_key[0], gamma=best_key[1],
        alpha=best_key[2] if best_key[2] == "fija" else float(best_key[2]), eps_min=best_key[3],
        validation_mean=round(float(np.mean([r["validation"] for r in best_rows])), 4),
        validation_by_seed=[r["validation"] for r in best_rows],
        ranking=[dict(algorithm=k[0], gamma=k[1], alpha=k[2], eps_min=k[3],
                      validation_mean=round(float(np.mean([r["validation"] for r in v])), 4),
                      validation_by_seed=[r["validation"] for r in v],
                      stability_mean=round(float(np.mean([r["stability"] for r in v])), 4),
                      routes=sorted({r["route"] for r in v})) for k, v in scored],
        episodes=args.episodes, combinations=len(stage1) + len(stage2), confirmation_runs=len(extra),
        minutes=round((time.perf_counter() - start) / 60, 1))
    (OUT / "seleccion.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    plot([{k: str(v) for k, v in r.items()} for r in rows2])
    print(json.dumps(selection, indent=2, ensure_ascii=False))


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink = "#1F2933"
    maia = LinearSegmentedColormap.from_list("maia", ["#FFFFFF", "#5BB1A5", "#5A6ACE", "#2E3A8C"])
    comma = lambda x: str(x).replace(".", ",")
    labels = [f"α {'fija 0,1' if a == 'fija' else 'N^-' + comma(a)}  ·  ε mín {comma(e)}" for a in ALPHAS for e in EPS_MINS]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.6), dpi=150, sharey=True)
    for ax, algorithm, title in zip(axes, ALGORITHMS, ("Q-learning", "SARSA")):
        grid = np.zeros((len(labels), len(GAMMAS)))
        for r in rows:
            if r["algorithm"] != algorithm:
                continue
            a = r["alpha"] if r["alpha"] == "fija" else float(r["alpha"])
            i = ALPHAS.index(a) * len(EPS_MINS) + EPS_MINS.index(float(r["eps_min"]))
            grid[i, GAMMAS.index(float(r["gamma"]))] = 100 * float(r["validation"])
        ax.imshow(grid, cmap=maia, vmin=0, vmax=45, aspect="auto")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                    color="white" if v > 28 else ink)
        ax.set_xticks(range(len(GAMMAS)), [f"γ = {comma(g)}" for g in GAMMAS], fontsize=9, color=ink)
        ax.set_yticks(range(len(labels)), labels, fontsize=8.5, color=ink)
        for k in range(1, len(ALPHAS)):
            ax.axhline(k * len(EPS_MINS) - 0.5, color="white", lw=3)
        ax.set_title(title, fontsize=11, color=ink, weight="bold")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("Etapa 2: llegadas a la meta en validación (%) por combinación", x=0.02, ha="left",
                 fontsize=12, color=ink, weight="bold")
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures/grid.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
