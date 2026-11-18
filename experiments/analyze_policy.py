"""Analiza la política aprendida: qué acción toma el agente en cada clase de estado.

Juega la política greedy de una Q-tabla (o una regla fija) en turnos de prueba y cuenta las
decisiones por situación, para contrastarlas con los comportamientos deseados (D1-D5) y no
deseados (N1-N5) de la Entrega 1.

Ejemplo:
    python experiments/analyze_policy.py --zones AB --model results/models/q_learning.npz \\
        --compare lote_mas_urgente --out results/logs/politica.md
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_warehouse.agents import POLICIES, TabularAgent  # noqa: E402
from rl_warehouse.env import WarehouseEnv  # noqa: E402

EVAL_SEED = 1_000_000

# Situaciones en las que hay una decisión real (o >= 1 y hay trabajo), en el orden de la Entrega 1.
SITUATIONS = [
    ("S3", "Hay una orden crítica (u_max = 3)"),
    ("S4a", "Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4)"),
    ("S4b", "Sin urgencia, alguna zona con consolidación alta (c = 2)"),
    ("S4c", "Sin urgencia, resto de casos"),
]
KINDS = ["lote_critica", "lote_otra", "lote_consolidada", "lote", "urgente", "esperar"]


def situation(env, s):
    n = env.n
    w, c, u = s[:n], s[n:2 * n], s[2 * n:3 * n]
    if s[-2] == 0 or not any(w):
        return None                                     # S1 o S2: solo puede esperar
    if max(u) == 3:
        return "S3"
    if s[-1] == 2 and max(c) <= 1:
        return "S4a"
    if max(c) == 2:
        return "S4b"
    return "S4c"


def kind(env, s, a):
    n = env.n
    c, u = s[n:2 * n], s[2 * n:3 * n]
    if a == env.wait:
        return "esperar"
    if a == env.urgent:
        return "urgente"
    if max(u) == 3:
        return "lote_critica" if u[a] == 3 else "lote_otra"
    return "lote_consolidada" if c[a] == max(c) and max(c) > 0 else "lote"


def rollout(env, decide, episodes):
    counts = {name: Counter() for name, _ in SITUATIONS}
    for i in range(episodes):
        state, info = env.reset(seed=EVAL_SEED + i)
        rng = np.random.default_rng(i)
        while True:
            action = decide(env, state, info["action_mask"], rng)
            name = situation(env, state)
            if name is not None:
                counts[name][kind(env, state, action)] += 1
            state, _, done, _, info = env.step(action)
            if done:
                break
    return counts


def table(title, counts):
    lines = [f"### {title}", "",
             "| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | "
             "Otro lote | Urgente | Esperar |",
             "|---|---|---|---|---|---|---|---|"]
    for name, description in SITUATIONS:
        total = sum(counts[name].values())
        cells = [f"{100 * counts[name][k] / total:.0f} %" if total else "—" for k in KINDS]
        lines.append(f"| {name}: {description} | {total} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/env_default.yaml"))
    parser.add_argument("--zones", default="")
    parser.add_argument("--model", type=Path, action="append", default=[], help="Q-tabla .npz (repetible)")
    parser.add_argument("--compare", nargs="*", default=[], choices=list(POLICIES), help="Reglas fijas a incluir")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None, help="Archivo .md de salida")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.zones:
        config.setdefault("centro_distribucion", {})["zonas"] = list(args.zones)
    env = WarehouseEnv(config)

    sections = [f"# Política aprendida por situación ({args.episodes} turnos de prueba, % de las decisiones)", ""]
    for path in args.model:
        agent = TabularAgent.load(path)
        counts = rollout(env, lambda e, s, m, r: agent.act(s, m, explore=False), args.episodes)
        sections += [table(f"Agente {path.stem} ({agent.algorithm})", counts), ""]
    for name in args.compare:
        counts = rollout(env, POLICIES[name], args.episodes)
        sections += [table(f"Regla fija {name}", counts), ""]
    text = "\n".join(sections)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
