"""Genera las cifras y tablas del reporte del cuarto 8x8 desde los resultados guardados.

    python experiments/cuarto_8x8/report_numbers.py

Escribe docs/proyecto_8x8/generado/*.tex (el reporte no tiene números copiados a mano).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_gridworld import AVANZAR, ObstacleGridEnv, TabularAgent  # noqa: E402
from rl_gridworld.baselines import POLICIES  # noqa: E402

RESULTS = ROOT / "results/cuarto_8x8"
OUT = ROOT / "docs/proyecto_8x8/generado"
TEST_SEED = 1_000_000
NAMES = {"q_learning": "Q-learning", "sarsa": "SARSA", "cautelosa": "Regla cautelosa", "directa": "Regla directa",
         "aleatoria": "Aleatoria (sin entrenar)"}
SHORT = {"q_learning": "QL", "sarsa": "SARSA", "cautelosa": "Caut", "directa": "Dir", "aleatoria": "Rand"}
FRONT = {0: "Libre", 1: "En riesgo", 2: "Ocupada"}


def num(x, d=1):
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", "{,}").replace("X", "\\,")
    return ("$-$" if x < 0 and round(abs(x), d) > 0 else "") + s


def rollout(env, decide, episodes):
    """Juega episodios de prueba y registra decisiones por percepción y causas de colisión."""
    by_front, causes = Counter(), Counter()
    for i in range(episodes):
        obs, info = env.reset(seed=TEST_SEED + i)
        rng = np.random.default_rng(i)
        while True:
            action = decide(env, obs, info["action_mask"], rng)
            if obs[3] in FRONT:
                by_front[(int(obs[3]), int(obs[4]), action == AVANZAR)] += 1
            obs, _, terminated, truncated, info = env.step(action)
            if env.outcome == "colision":
                causes["cruce" if info["swap"] else "avanzando" if action == AVANZAR else "girando"] += 1
            if terminated or truncated:
                break
    return by_front, causes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((RESULTS / "logs/summary.json").read_text(encoding="utf-8"))
    res, meta = data["results"], data["meta"]
    lines = []

    def macro(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for key, short in SHORT.items():
        s = res[key]
        macro(f"{short}Exito", num(100 * s["success"], 1))
        macro(f"{short}Exitoci", num(100 * s["success_ci95"], 1))
        macro(f"{short}Choque", num(100 * s["collision"], 1))
        macro(f"{short}Pasos", num(s["steps_success"], 1) if s["success"] > 0 else "--")
        macro(f"{short}R", num(s["reward"], 1))
        if "checkpoint" in s:
            macro(f"{short}Ckpt", f"{s['checkpoint']:,}".replace(",", "\\,"))
            macro(f"{short}Estados", f"{s['states']:,}".replace(",", "\\,"))
    macro("Episodios", f"{meta['episodes']:,}".replace(",", "\\,"))
    macro("TurnosPrueba", f"{meta['test_episodes']:,}".replace(",", "\\,"))
    macro("TurnosValidacion", str(meta["validation_episodes"]))
    macro("Alpha", num(meta["alpha"], 2))
    macro("GammaF", num(meta["gamma"], 2))
    macro("EpsIni", num(meta["epsilon"][0], 1))
    macro("EpsMin", num(meta["epsilon"][1], 2))

    # Diferencias pareadas (mismos episodios de prueba) frente a la regla cautelosa
    with (RESULTS / "logs/comparison.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    col = lambda p, k: np.array([float(r[k]) for r in rows if r["policy"] == p])
    for key in ("q_learning", "sarsa"):
        d = col(key, "success") - col("cautelosa", "success")
        macro(f"Dif{SHORT[key]}", num(100 * d.mean(), 1))
        macro(f"Dif{SHORT[key]}ci", num(100 * 1.96 * d.std(ddof=1) / np.sqrt(len(d)), 1))
        both = int(np.sum((col(key, "success") == 1) & (col("cautelosa", "success") == 0)))
        other = int(np.sum((col(key, "success") == 0) & (col("cautelosa", "success") == 1)))
        macro(f"Solo{SHORT[key]}", str(both))
        macro(f"SoloCaut{SHORT[key]}", str(other))

    # Tabla de resultados
    table = []
    for key in ("q_learning", "sarsa", "cautelosa", "directa", "aleatoria"):
        s = res[key]
        name = f"\\textbf{{{NAMES[key]}}}" if key == "q_learning" else NAMES[key]
        steps = num(s["steps_success"], 1) if s["success"] > 0 else "--"
        table.append(f"{name} & {num(100 * s['success'], 1)} $\\pm$ {num(100 * s['success_ci95'], 1)} & "
                     f"{num(100 * s['collision'], 1)} & {num(100 * s['truncated'], 1)} & {steps} & "
                     f"{num(s['reward'], 1)} $\\pm$ {num(s['reward_ci95'], 1)} \\\\")
        if key == "sarsa":
            table.append("\\midrule")
    (OUT / "tabla_resultados.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    # Política por percepción y causas de colisión
    config = yaml.safe_load((ROOT / "configs/cuarto_8x8.yaml").read_text(encoding="utf-8"))
    env = ObstacleGridEnv(config.get("ambiente", {}))
    agent = TabularAgent.load(RESULTS / "models/q_learning.npz")
    episodes = meta["test_episodes"]
    ql_front, ql_causes = rollout(env, lambda e, o, m, r: agent.act(o, m, explore=False), episodes)
    ca_front, ca_causes = rollout(env, POLICIES["cautelosa"], episodes)
    table = []
    for front, label in FRONT.items():
        for aq in (0, 1):
            cells = []
            for counts in (ql_front, ca_front):
                adv, turn = counts[(front, aq, True)], counts[(front, aq, False)]
                total = adv + turn
                cells += [f"{total:,}".replace(",", "\\,"), num(100 * adv / total, 0) if total else "--"]
            here = "Sí" if aq else "No"
            if cells[0] == "0" and cells[2] == "0":
                continue                                   # combinación imposible (p. ej. ocupada sin vecino)
            table.append(f"{label} & {here} & " + " & ".join(cells) + " \\\\")
    (OUT / "tabla_politica.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    for tag, counts in (("QL", ql_causes), ("Caut", ca_causes)):
        total = sum(counts.values())
        for cause in ("avanzando", "cruce", "girando"):
            macro(f"Causa{tag}{cause.capitalize()}", num(100 * counts[cause] / total, 0))
    adv = lambda counts, f: (sum(counts[(f, a, True)] for a in (0, 1)),
                             sum(counts[(f, a, True)] + counts[(f, a, False)] for a in (0, 1)))
    for f, name in ((1, "Riesgo"), (2, "Ocupada"), (0, "Libre")):
        a, t = adv(ql_front, f)
        macro(f"QLAvanza{name}", num(100 * a / t, 0))

    # Extracto de la Q-tabla
    heading = ["N", "E", "S", "O"]
    front_name = ["libre", "riesgo", "ocupada", "pared"]
    picks = [(0, 0, 0, 3, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0), (0, 0, 2, 0, 0, 0, 0),
             (3, 3, 1, 0, 0, 0, 0), (3, 3, 1, 2, 1, 0, 0), (7, 6, 1, 0, 0, 0, 0), (6, 7, 2, 0, 0, 0, 0)]
    table = []
    for key in picks:
        if key not in agent.q:
            continue
        q = agent.q[key]
        mask = [True, True, key[3] != 3]
        best = max((v, i) for i, v in enumerate(q) if mask[i])[1]
        vals = [("\\textbf{" + num(v, 1) + "}" if i == best else num(v, 1)) if mask[i] else "n/a"
                for i, v in enumerate(q)]
        table.append(f"$({key[0]},{key[1]})$ & {heading[key[2]]} & {front_name[key[3]]} & {key[4]} & {key[5]} & "
                     f"{key[6]} & " + " & ".join(vals) + " \\\\")
    (OUT / "tabla_qtabla.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    (OUT / "cifras.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado en {OUT}")
    print("Causas QL:", dict(ql_causes), "Causas cautelosa:", dict(ca_causes))


if __name__ == "__main__":
    main()
