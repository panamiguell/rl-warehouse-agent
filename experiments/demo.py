"""Carga una Q-tabla entrenada y graba un turno del agente como video (para la entrega final).

Ejemplos:
    python experiments/demo.py --model results/models/q_learning.npz
    python experiments/demo.py --policy lote_mas_urgente          # una regla fija, para comparar
    python experiments/demo.py --model results/models/q_learning.npz --human   # ventana en vivo

Requiere ffmpeg para .mp4; con --output *.gif usa Pillow.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_warehouse.agents import POLICIES, TabularAgent  # noqa: E402
from rl_warehouse.env import WarehouseEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/env_default.yaml"))
    parser.add_argument("--zones", default="", help="Debe coincidir con las zonas usadas al entrenar")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path, help="Q-tabla guardada con TabularAgent.save (.npz)")
    group.add_argument("--policy", choices=list(POLICIES), help="Regla fija en lugar de un agente")
    parser.add_argument("--seed", type=int, default=2_000_000, help="Semilla del turno (no usada al entrenar)")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/demo.mp4")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = turno completo")
    parser.add_argument("--human", action="store_true", help="Mostrar en una ventana en vez de grabar")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.zones:
        config.setdefault("centro_distribucion", {})["zonas"] = list(args.zones)
    env = WarehouseEnv(config, render_mode="human" if args.human else "rgb_array")
    agent = TabularAgent.load(args.model) if args.model else None
    if agent is not None and agent.n_actions != env.action_space.n:
        parser.error(f"La Q-tabla tiene {agent.n_actions} acciones y el ambiente {env.action_space.n}: revisa --zones")
    rng = np.random.default_rng(args.seed)

    state, info = env.reset(seed=args.seed)
    frames, total, steps, unseen = [env.render()], 0.0, 0, 0
    while True:
        mask = info["action_mask"]
        if agent is None:
            action = POLICIES[args.policy](env, state, mask, rng)
        else:
            action = agent.greedy(state, mask)
            if action is None:          # estado nunca visto en entrenamiento
                unseen += 1
                action = agent.act(state, mask, explore=False)
        state, reward, done, _, info = env.step(action)
        total += reward
        steps += 1
        if not args.human:
            frames.append(env.render())
        if done or (args.max_steps and steps >= args.max_steps):
            break

    m = info["metrics"]
    print(f"Recompensa {total:.0f} | órdenes {m['arrived']} | a tiempo {m['on_time']} | vencidas {m['expired']} | "
          f"visitas ahorradas {m['saved_visits']} | decisiones {steps} | estados no vistos {unseen}")
    if args.human:
        return
    save_video(frames, args.output, args.fps)
    env.close()
    print(f"Video: {args.output}")


def save_video(frames, path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    frames += [frames[-1]] * (2 * fps)          # mantener el cuadro final 2 s
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
