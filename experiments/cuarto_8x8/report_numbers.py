"""Genera las cifras, tablas y la figura de la ruta del reporte del cuarto 8x8.

    python experiments/cuarto_8x8/report_numbers.py

Lee results/cuarto_8x8 y escribe docs/proyecto_8x8/generado/*.tex y
docs/proyecto_8x8/figuras/ruta_aprendida.png. El reporte no tiene números copiados a mano.
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
from rl_gridworld.baselines import escalera  # noqa: E402

OFFICIAL = ROOT / "results/cuarto_8x8"
OUT, FIGS = ROOT / "docs/proyecto_8x8/generado", ROOT / "docs/proyecto_8x8/figuras"
TEST_SEED = 1_000_000
NAMES = {"q_learning": "Q-learning", "sarsa": "SARSA", "borde": "Regla: por el borde",
         "escalera": "Regla: escalera por el centro", "aleatoria": "Aleatoria (sin entrenar)"}
SHORT = {"q_learning": "QL", "sarsa": "SARSA", "borde": "Borde", "escalera": "Esc", "aleatoria": "Rand"}
HEAD = "NESO"


def num(x, d=1):
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", "{,}").replace("X", "\\,")
    return ("$-$" if x < 0 and round(abs(x), d) > 0 else "") + s


def greedy_route(agent):
    """Ruta de la política greedy: no depende de los obstáculos porque el agente no los ve."""
    env = ObstacleGridEnv({"obstaculos": 0})
    obs, info = env.reset(seed=0)
    cells, actions = [env.agent], []
    while True:
        action = agent.act(obs, info["action_mask"], explore=False)
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        cells.append(env.agent)
        if terminated or truncated:
            return cells, actions


def rule_route(policy):
    env = ObstacleGridEnv({"obstaculos": 0})
    obs, info = env.reset(seed=0)
    cells = [env.agent]
    while env.outcome is None:
        obs, *_ = env.step(policy(env, obs, info["action_mask"], None))
        cells.append(env.agent)
    return cells


def draw_routes(routes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(5.6, 5.9), dpi=150)
    ax.set_xlim(-0.05, 8.05)
    ax.set_ylim(8.05, -0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((7, 7), 1, 1, color="#7ED957", lw=0))
    for k in range(9):
        ax.plot([0, 8], [k, k], color="#9CA3AF", lw=0.8)
        ax.plot([k, k], [0, 8], color="#9CA3AF", lw=0.8)
    for cells, color, style, width, label, offset in routes:
        xs = [c + 0.5 + offset for _, c in cells]
        ys = [r + 0.5 + offset for r, _ in cells]
        ax.plot(xs, ys, color=color, ls=style, lw=width, label=label, solid_capstyle="round")
    pts = np.array([[0.5, 0.2], [0.3, 0.62], [0.7, 0.62]])
    ax.add_patch(Polygon(pts, closed=True, color="#111111"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    official = json.loads((OFFICIAL / "logs/summary.json").read_text(encoding="utf-8"))
    res, meta = official["results"], official["meta"]
    lines = []

    def macro(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for key, s in res.items():
        short = SHORT[key]
        macro(f"{short}Exito", num(100 * s["success"], 1))
        macro(f"{short}Exitoci", num(100 * s["success_ci95"], 1))
        macro(f"{short}Choque", num(100 * s["collision"], 1))
        macro(f"{short}Pasos", num(s["steps_success"], 1) if s["success"] > 0 else "--")
        if "checkpoint" in s:
            macro(f"{short}Ckpt", f"{s['checkpoint']:,}".replace(",", "\\,"))
            macro(f"{short}Estados", f"{s['states']:,}".replace(",", "\\,"))
    hp = official["meta"]
    macro("Episodios", f"{meta['episodes']:,}".replace(",", "\\,"))
    macro("TurnosPrueba", f"{meta['test_episodes']:,}".replace(",", "\\,"))
    macro("TurnosValidacion", str(meta["validation_episodes"]))
    macro("GammaF", num(hp["gamma"], 2))
    macro("AlphaPot", num(hp["alpha_potencia"], 1))
    macro("AlphaMin", num(hp["alpha_minima"], 2))
    macro("EpsIni", num(meta["epsilon"][0], 1))
    macro("EpsMin", num(meta["epsilon"][1], 2))

    # Tiempo de ejecución y equipo (los registra train.py)
    def dur(seconds):
        seconds = round(seconds)
        return f"{seconds // 60}\\,min {seconds % 60}\\,s" if seconds >= 60 else f"{seconds}\\,s"

    def tex(text):
        text = str(text).replace("(R)", "").replace("(TM)", "").replace(" Processor", "").replace("@", "a")
        return " ".join(text.split()).replace("_", "\\_")

    system = meta["system"]
    macro("CPU", tex(system["cpu"]))
    macro("Nucleos", str(system["cores"]))
    macro("RAM", num(system["ram_gb"], 1))
    macro("SO", tex(system["os"]))
    for key, name in (("python", "Py"), ("numpy", "Numpy"), ("gymnasium", "Gym"), ("matplotlib", "Mpl"),
                      ("pyyaml", "Yaml")):
        macro(f"Ver{name}", tex(system[key]))
    for key in ("q_learning", "sarsa"):
        s = res[key]
        macro(f"T{SHORT[key]}Ent", dur(s["train_seconds"]))
        macro(f"T{SHORT[key]}EntVal", dur(s["train_validation_seconds"]))
        macro(f"T{SHORT[key]}Prueba", dur(s["test_seconds"]))
        macro(f"T{SHORT[key]}MsEp", num(1000 * s["train_seconds"] / meta["episodes"], 1))
    macro("TTotal", dur(meta["total_seconds"]))

    # Video de la entrega (resumen que escribe demo.py)
    demo = json.loads((OFFICIAL / "logs/demo.json").read_text(encoding="utf-8"))
    macro("VideoEpisodios", str(demo["episodes"]))
    macro("VideoMeta", str(demo["successes"]))
    macro("VideoExito", num(100 * demo["successes"] / demo["episodes"], 0))
    minutes, seconds = divmod(round(demo["seconds"]), 60)
    macro("VideoDuracion", f"{minutes}:{seconds:02d}")
    macro("TVideo", dur(demo["generation_seconds"]))
    # Las mismas semillas del video, extendidas a tantos episodios como la prueba: muestra que la
    # diferencia con la prueba es por el tamaño de la muestra y no por las semillas elegidas.
    video_env = ObstacleGridEnv(yaml.safe_load((ROOT / "configs/cuarto_8x8.yaml").read_text(encoding="utf-8"))
                                .get("ambiente", {}))
    video_agent = TabularAgent.load(OFFICIAL / "models" / demo["model"])
    long_run = [video_agent.episode(video_env, seed=demo["seeds"][0] + i, train=False)["success"]
                for i in range(meta["test_episodes"])]
    macro("VideoExitoLargo", num(100 * np.mean(long_run), 1))

    # Comparaciones pareadas (mismos episodios de prueba)
    with (OFFICIAL / "logs/comparison.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    col = lambda p, k: np.array([float(r[k]) for r in rows if r["policy"] == p])
    same = int(np.sum(col("q_learning", "success") == col("borde", "success")))
    macro("IgualBorde", f"{same:,}".replace(",", "\\,"))
    for key in ("escalera", "aleatoria"):
        d = col("q_learning", "success") - col(key, "success")
        macro(f"DifQL{SHORT[key]}", num(100 * d.mean(), 1))
        macro(f"DifQL{SHORT[key]}ci", num(100 * 1.96 * d.std(ddof=1) / np.sqrt(len(d)), 1))
    # Tabla de resultados oficiales
    table = []
    for key in ("q_learning", "sarsa", "borde", "escalera", "aleatoria"):
        s = res[key]
        name = f"\\textbf{{{NAMES[key]}}}" if key == "q_learning" else NAMES[key]
        steps = num(s["steps_success"], 1) if s["success"] > 0 else "--"
        table.append(f"{name} & {num(100 * s['success'], 1)} $\\pm$ {num(100 * s['success_ci95'], 1)} & "
                     f"{num(100 * s['collision'], 1)} & {num(100 * s['truncated'], 1)} & {steps} & "
                     f"{num(s['reward'], 1)} $\\pm$ {num(s['reward_ci95'], 1)} \\\\")
        if key == "sarsa":
            table.append("\\midrule")
    (OUT / "tabla_resultados.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    # Búsqueda de hiperparámetros (grid_search.py)
    grid_dir = OFFICIAL / "grid"
    read = lambda name: list(csv.DictReader((grid_dir / name).open(encoding="utf-8")))
    stage1, stage2, confirm = read("etapa1_gamma.csv"), read("etapa2_grid.csv"), read("confirmacion.csv")
    selection = json.loads((grid_dir / "seleccion.json").read_text(encoding="utf-8"))

    def route_name(route):
        if route.endswith("..."):
            return "gira sin salir" if set(route[:-3]) <= {"I", "D"} else "da vueltas"
        r, c, h = 0, 0, 0
        cells = []
        for a in route:
            if a == "I":
                h = (h - 1) % 4
            elif a == "D":
                h = (h + 1) % 4
            else:
                r, c = r + [-1, 0, 1, 0][h], c + [0, 1, 0, -1][h]
                cells.append((r, c))
        turns = sum(a != "A" for a in route)
        side = "fila 0 y columna 7" if (0, 7) in cells else "columna 0 y fila 7" if (7, 0) in cells else "otra"
        return f"borde por {side} ({len(route)} pasos, {turns} giros)"

    def alpha_name(a):
        return "fija 0{,}1" if a == "fija" else f"$N^{{-{num(float(a), 1)}}}$"

    algo = {"q_learning": "Q-learning", "sarsa": "SARSA"}
    table = []
    for r in stage1:
        table.append(f"{algo[r['algorithm']]} & {num(float(r['gamma']), 2)} & {num(100 * float(r['validation']), 1)} & "
                     f"{route_name(r['route'])} & {num(float(r['train_steps_per_episode']), 0)} \\\\")
    (OUT / "tabla_gamma.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    groups = {}
    for r in confirm:
        groups.setdefault((r["algorithm"], r["gamma"], r["alpha"], r["eps_min"]), []).append(r)
    table = []
    for entry in selection["ranking"]:
        k = (entry["algorithm"], str(entry["gamma"]), str(entry["alpha"]), str(entry["eps_min"]))
        rows = sorted(groups[k], key=lambda r: int(r["seed"]))
        seeds = " & ".join(num(100 * float(r["validation"]), 1) for r in rows)
        routes = "; ".join(sorted({route_name(r["route"]).split(" (")[0].replace("borde por ", "") for r in rows}))
        name = f"{algo[k[0]]} & {num(float(k[1]), 3 if float(k[1]) > 0.99 else 2)} & {alpha_name(k[2])} & {num(float(k[3]), 2)}"
        table.append(f"{name} & {seeds} & \\textbf{{{num(100 * entry['validation_mean'], 1)}}} & {routes} \\\\")
    (OUT / "tabla_confirmacion.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    val2 = [float(r["validation"]) for r in stage2]
    border = [r for r in stage2 if route_name(r["route"]).startswith("borde")]
    macro("GridTotal", str(selection["combinations"]))
    macro("GridEtapaDos", str(len(stage2)))
    macro("GridConfirmacion", str(selection["confirmation_runs"]))
    macro("GridMinutos", str(round(selection["minutes"])))
    macro("GridMin", num(100 * min(val2), 1))
    macro("GridMax", num(100 * max(val2), 1))
    macro("GridBorde", str(len(border)))
    macro("GridFilaCero", str(sum("fila 0 y" in route_name(r["route"]) for r in stage2)))
    macro("GridColumnaCero", str(sum("columna 0 y" in route_name(r["route"]) for r in stage2)))
    for name, rows in (("Fija", [r for r in stage2 if r["alpha"] == "fija"]),
                       ("Visitas", [r for r in stage2 if r["alpha"] != "fija"])):
        macro(f"GridCeros{name}", num(np.mean([int(r["zero_checkpoints"]) for r in rows]), 1))
        macro(f"GridVal{name}", num(100 * np.mean([float(r["validation"]) for r in rows]), 1))
    for key in ("q_learning", "sarsa"):
        macro(f"GridVal{SHORT[key]}", num(100 * np.mean([float(r["validation"]) for r in stage2 if r["algorithm"] == key]), 1))
    for e, name in (("0.01", "Uno"), ("0.05", "Cinco"), ("0.1", "Diez")):
        macro(f"GridValEps{name}", num(100 * np.mean([float(r["validation"]) for r in stage2 if r["eps_min"] == e]), 1))
    for g, name in (("0.99", "NN"), ("0.999", "NNN")):
        macro(f"GridValGamma{name}", num(100 * np.mean([float(r["validation"]) for r in stage2 if r["gamma"] == g]), 1))
    macro("SelAlg", algo[selection["algorithm"]])
    macro("SelGamma", num(float(selection["gamma"]), 3 if float(selection["gamma"]) > 0.99 else 2))
    macro("SelAlpha", alpha_name(str(selection["alpha"]) if selection["alpha"] == "fija" else selection["alpha"]))
    macro("SelEps", num(float(selection["eps_min"]), 2))
    macro("SelVal", num(100 * selection["validation_mean"], 1))

    # Ruta aprendida y causas de colisión
    config = yaml.safe_load((ROOT / "configs/cuarto_8x8.yaml").read_text(encoding="utf-8"))
    ql = TabularAgent.load(OFFICIAL / "models/q_learning.npz")
    sarsa = TabularAgent.load(OFFICIAL / "models/sarsa.npz")
    ql_cells, ql_actions = greedy_route(ql)
    sa_cells, sa_actions = greedy_route(sarsa)
    letters = {0: "I", 1: "D", 2: "A"}
    macro("RutaQL", "".join(letters[a] for a in ql_actions))
    macro("RutaSARSA", "".join(letters[a] for a in sa_actions))
    macro("GirosQL", str(sum(a != AVANZAR for a in ql_actions)))
    macro("GirosSARSA", str(sum(a != AVANZAR for a in sa_actions)))
    draw_routes([(rule_route(escalera), "#9CA3AF", ":", 2, "Regla: escalera por el centro (28 pasos, 14 giros)", 0),
                 (sa_cells, "#eb6834", "--", 2.2, f"SARSA ({len(sa_actions)} pasos)", 0.08),
                 (ql_cells, "#2a78d6", "-", 2.6, f"Q-learning ({len(ql_actions)} pasos)", -0.08)],
                FIGS / "ruta_aprendida.png")
    env = ObstacleGridEnv(config.get("ambiente", {}))
    causes, where = Counter(), Counter()
    for i in range(meta["test_episodes"]):
        obs, info = env.reset(seed=TEST_SEED + i)
        while True:
            action = ql.act(obs, info["action_mask"], explore=False)
            cell = env.agent
            obs, _, terminated, truncated, info = env.step(action)
            if env.outcome == "colision":
                causes["cruce" if info["swap"] else "avanzando" if action == AVANZAR else "girando"] += 1
                where["esquina" if cell == (0, 7) else "fila 0" if cell[0] == 0 else "columna 7"] += 1
            if terminated or truncated:
                break
    total = sum(causes.values())
    for cause in ("avanzando", "cruce", "girando"):
        macro(f"Causa{cause.capitalize()}", num(100 * causes[cause] / total, 0))
    for place, name in (("fila 0", "Fila"), ("esquina", "Esquina"), ("columna 7", "Columna")):
        macro(f"Lugar{name}", num(100 * where[place] / total, 0))

    # Extracto de la Q-tabla
    picks = [(0, 0, 0), (0, 0, 1), (0, 0, 3), (0, 3, 1), (0, 7, 1), (0, 7, 2), (3, 7, 2), (6, 7, 2), (3, 3, 1)]
    table = []
    for key in picks:
        if key not in ql.q:
            continue
        q = ql.q[key]
        env.agent, env.heading = key[:2], key[2]
        mask = [True, True, env.inside(env.ahead())]
        best = max((v, i) for i, v in enumerate(q) if mask[i])[1]
        vals = [("\\textbf{" + num(v, 1) + "}" if i == best else num(v, 1)) if mask[i] else "n/a"
                for i, v in enumerate(q)]
        visits = sum(ql.visits.get((key, a), 0) for a in range(3)) if ql.visits else None
        table.append(f"$({key[0]},{key[1]})$ & {HEAD[key[2]]} & " + " & ".join(vals) + " \\\\")
    (OUT / "tabla_qtabla.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    (OUT / "cifras.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generado en {OUT}")
    print("Ruta QL:", "".join(letters[a] for a in ql_actions), "| SARSA:", "".join(letters[a] for a in sa_actions))
    print("Causas:", dict(causes), "Lugares:", dict(where))


if __name__ == "__main__":
    main()
