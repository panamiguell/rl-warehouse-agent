"""Genera las tablas y cifras del reporte final a partir de los resultados guardados.

Escribe docs/entrega_final/generado/*.tex para que el reporte nunca tenga números copiados a mano.

    python experiments/report_numbers.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_policy import KINDS, SITUATIONS, rollout  # noqa: E402
from rl_warehouse.agents import POLICIES, TabularAgent  # noqa: E402
from rl_warehouse.env import WarehouseEnv  # noqa: E402

NAMES = {"q_learning": "Q-learning", "sarsa": "SARSA", "monte_carlo": "Monte Carlo",
         "lote_mas_urgente": "Lote más urgente", "umbral_consolidacion": "Umbral de consolidación",
         "lote_mas_consolidado": "Lote más consolidado", "aleatoria": "Aleatoria",
         "urgente_siempre": "Urgente siempre", "esperar_siempre": "Esperar siempre"}
MACRO = {"q_learning": "QL", "sarsa": "SARSA", "monte_carlo": "MC", "lote_mas_urgente": "Urg",
         "umbral_consolidacion": "Umbral", "lote_mas_consolidado": "Cons", "aleatoria": "Rand",
         "urgente_siempre": "UrgSiempre", "esperar_siempre": "Esperar"}


def num(x, d=1):
    """Número en formato español para LaTeX (coma decimal, espacio fino en miles, signo menos)."""
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", "{,}").replace("X", "\\,")
    return ("$-$" if x < 0 and round(abs(x), d) > 0 else "") + s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/entrega_final/generado")
    parser.add_argument("--policy-episodes", type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    final = json.loads((args.results / "logs/summary.json").read_text(encoding="utf-8"))
    e1 = json.loads((args.results / "entrega1/logs/summary.json").read_text(encoding="utf-8"))
    fr, er, meta = final["results"], e1["results"], final["meta"]

    # ---------- macros con cifras clave ----------
    m = []
    def macro(name, value):
        m.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    for key, short in MACRO.items():
        for tag, res in (("", fr), ("E", er)):
            if key not in res:
                continue
            s = res[key]
            macro(f"{tag}{short}R", num(s["reward"], 0))
            macro(f"{tag}{short}Rci", num(s["reward_ci95"], 0))
            macro(f"{tag}{short}V", num(s["expired"], 1))
            macro(f"{tag}{short}Vci", num(s["expired_ci95"], 1))
            macro(f"{tag}{short}P", num(s["pending"], 1))
            macro(f"{tag}{short}T", num(100 * s["on_time_rate"], 1))
            macro(f"{tag}{short}VL", num(s["visits_per_line"], 3))
            macro(f"{tag}{short}W", num(s["wip_mean"], 1))
            macro(f"{tag}{short}Viajes", num(s["trips"], 1))
            macro(f"{tag}{short}Llegadas", num(s["arrived"], 1))
            if "checkpoint" in s:
                macro(f"{tag}{short}Ckpt", f"{s['checkpoint']:,}".replace(",", "\\,"))
                macro(f"{tag}{short}Estados", f"{s['states']:,}".replace(",", "\\,"))
    rules = {k: v for k, v in fr.items() if v["kind"] == "regla"}
    best = max(rules, key=lambda k: rules[k]["reward"])

    # Diferencias pareadas agente - mejor regla (mismas semillas de prueba), IC del 95 %.
    for tag, folder in (("", args.results / "logs"), ("E", args.results / "entrega1/logs")):
        with (folder / "comparison.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        def col(policy, key):
            return np.array([float(r[key]) for r in rows if r["policy"] == policy])
        for agent_key in ("q_learning", "sarsa"):
            for key, label, d in (("reward", "R", 0), ("expired", "V", 1), ("trips", "Viajes", 1),
                                  ("pending", "P", 1), ("wip_line_min", "W", 1)):
                diff = col(agent_key, key) - col(best, key)
                if key == "wip_line_min":
                    diff = diff / 480
                macro(f"Dif{tag}{MACRO[agent_key]}{label}", num(diff.mean(), d))
                macro(f"Dif{tag}{MACRO[agent_key]}{label}ci", num(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)), d))
    macro("MejorRegla", NAMES[best])
    macro("Episodios", f"{meta['episodes']:,}".replace(",", "\\,"))
    macro("TurnosPrueba", str(meta["eval_episodes"]))
    macro("TurnosValidacion", str(meta["curve_episodes"]))
    macro("Alpha", num(meta["alpha"], 2))
    macro("GammaF", num(meta["gamma"], 2))
    macro("EpsIni", num(meta["epsilon"][0], 2))
    macro("EpsMin", num(meta["epsilon"][1], 2))
    macro("EGamma", num(e1["meta"]["gamma"], 2))
    (args.out / "cifras.tex").write_text("\n".join(m) + "\n", encoding="utf-8")

    # ---------- tabla de resultados en prueba (especificación final) ----------
    order = ["q_learning", "sarsa", "monte_carlo", "lote_mas_urgente", "umbral_consolidacion",
             "lote_mas_consolidado", "aleatoria", "urgente_siempre", "esperar_siempre"]
    rows = []
    for key in order:
        s = fr[key]
        name = NAMES[key] if s["kind"] == "rl" else f"Regla: {NAMES[key].lower()}"
        if key == "q_learning":
            name = f"\\textbf{{{name}}}"
        rows.append(f"{name} & {num(s['reward'], 0)} $\\pm$ {num(s['reward_ci95'], 0)} & "
                    f"{num(s['expired'], 1)} $\\pm$ {num(s['expired_ci95'], 1)} & {num(100 * s['on_time_rate'], 1)} & "
                    f"{num(s['pending'], 1)} & {num(s['visits_per_line'], 3)} & {num(s['trips'], 1)} & "
                    f"{num(s['wip_mean'], 1)} \\\\")
        if key == "monte_carlo":
            rows.append("\\midrule")
    (args.out / "tabla_resultados.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------- tabla especificación entregada vs final ----------
    rows = []
    for key in ["q_learning", "sarsa", "monte_carlo", "lote_mas_urgente", "umbral_consolidacion", "aleatoria"]:
        a, b = er[key], fr[key]
        rows.append(f"{NAMES[key]} & {num(a['expired'], 1)} & {num(b['expired'], 1)} & "
                    f"{num(a['pending'], 1)} & {num(b['pending'], 1)} & "
                    f"{num(a['visits_per_line'], 3)} & {num(b['visits_per_line'], 3)} & "
                    f"{num(a['wip_mean'], 1)} & {num(b['wip_mean'], 1)} \\\\")
        if key == "monte_carlo":
            rows.append("\\midrule")
    (args.out / "tabla_especificaciones.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------- política aprendida por situación ----------
    config = yaml.safe_load((ROOT / "configs/env_default.yaml").read_text(encoding="utf-8"))
    env = WarehouseEnv(config)
    agent = TabularAgent.load(args.results / "models/q_learning.npz")
    learned = rollout(env, lambda e, s, mk, r: agent.act(s, mk, explore=False), args.policy_episodes)
    rule = rollout(env, POLICIES[best], args.policy_episodes)
    short = {"S3": "Orden crítica ($u_{\\max}=3$)", "S4a": "Alta demanda, consolidación baja",
             "S4b": "Alguna zona con $c_z=2$", "S4c": "Resto sin urgencia"}
    rows = []
    for name, _ in SITUATIONS:
        for label, counts in (("Q-learning", learned), (NAMES[best], rule)):
            total = sum(counts[name].values())
            cells = [num(100 * counts[name][k] / total, 0) if total else "--" for k in KINDS]
            first = short[name] if label == "Q-learning" else ""
            rows.append(f"{first} & {label} & {total} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace[2pt]")
    (args.out / "tabla_politica.tex").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    s3 = learned["S3"]
    total = sum(s3.values())
    extra = [f"\\newcommand{{\\PolS{k.replace('_', '')}}}{{{num(100 * s3[k] / total, 0)}}}" for k in KINDS]
    s4a = learned["S4a"]
    t4a = sum(s4a.values())
    extra.append(f"\\newcommand{{\\PolDcuatroEsperar}}{{{num(100 * s4a['esperar'] / t4a, 0) if t4a else '--'}}}")
    extra.append(f"\\newcommand{{\\PolTurnos}}{{{args.policy_episodes}}}")
    extra.append(f"\\newcommand{{\\PolRestoEsperar}}{{{num(100 * learned['S4c']['esperar'] / max(sum(learned['S4c'].values()), 1), 0)}}}")
    # Agente entrenado con la especificación entregada: qué hace con una orden crítica.
    config_e1 = yaml.safe_load((ROOT / "configs/env_entrega1.yaml").read_text(encoding="utf-8"))
    env_e1 = WarehouseEnv(config_e1)
    agent_e1 = TabularAgent.load(args.results / "entrega1/models/q_learning.npz")
    learned_e1 = rollout(env_e1, lambda e, s, mk, r: agent_e1.act(s, mk, explore=False), args.policy_episodes)
    t3 = sum(learned_e1["S3"].values())
    extra += [f"\\newcommand{{\\PolES{k.replace('_', '')}}}{{{num(100 * learned_e1['S3'][k] / t3, 0)}}}" for k in KINDS]
    with (args.out / "cifras.tex").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(extra) + "\n")
    print(f"Generado en {args.out}")


if __name__ == "__main__":
    main()
