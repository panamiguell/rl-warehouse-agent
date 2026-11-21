"""Carga la Q-tabla entrenada y graba episodios del agente en el cuarto 8x8 (video de la entrega).

    python experiments/cuarto_8x8/demo.py                                   # Q-learning, 30 episodios
    python experiments/cuarto_8x8/demo.py --model results/cuarto_8x8/models/sarsa.npz
    python experiments/cuarto_8x8/demo.py --human                           # ventana en vivo
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rl_gridworld import ObstacleGridEnv, TabularAgent  # noqa: E402
from rl_gridworld.render import draw  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/cuarto_8x8.yaml"))
    parser.add_argument("--model", type=Path, default=ROOT / "results/cuarto_8x8/models/q_learning.npz")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2_000_000, help="Semillas no usadas al entrenar")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cuarto_8x8/figures/demo.mp4")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    env = ObstacleGridEnv(config.get("ambiente", {}), render_mode="human" if args.human else "rgb_array")
    agent = TabularAgent.load(args.model)                 # la memoria del agente: la Q-tabla guardada
    name = {"q_learning": "Q-learning", "sarsa": "SARSA"}[agent.algorithm]
    frames, results = [], []
    for k in range(args.episodes):
        obs, info = env.reset(seed=args.seed + k)
        title = f"Agente {name} · episodio {k + 1} de {args.episodes}"
        frames += [draw(env, title=title)] * 2
        while True:
            action = agent.act(obs, info["action_mask"], explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(draw(env, title=title))
            if terminated or truncated:
                frames += [frames[-1]] * (2 * args.fps)    # pausa al final del episodio
                results.append(env.outcome or "límite")
                break
    print(f"{name}: " + ", ".join(results) + f" ({results.count('meta')}/{len(results)} llegan a la meta)")
    if args.human:
        return
    save_video(frames, args.output, args.fps)
    print(f"Video: {args.output}")
    # Resumen del video para el reporte (lo lee report_numbers.py)
    summary = dict(model=args.model.name, algorithm=agent.algorithm, episodes=len(results),
                   seeds=[args.seed, args.seed + args.episodes - 1], successes=results.count("meta"),
                   collisions=results.count("colision"), seconds=len(frames) / args.fps, outcomes=results)
    log = ROOT / "results/cuarto_8x8/logs/demo.json"
    log.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def save_video(frames, path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    image = ax.imshow(frames[0])

    def update(i):
        image.set_data(frames[i])
        return (image,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    if path.suffix.lower() == ".gif":
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, codec="libx264",
                                                      extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"]))
    plt.close(fig)


if __name__ == "__main__":
    main()
