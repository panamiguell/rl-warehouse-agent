"""Dibujo del cuarto 8x8 con matplotlib (figura del documento y cuadros del video)."""
import numpy as np

INK, MUTED, GRID = "#1F2933", "#6B7280", "#9CA3AF"
GOAL, OBSTACLE, AGENT, RISK, TRAIL = "#7ED957", "#1F3FE0", "#111111", "#FDE2E1", "#D6E4F5"
LABELS = {0: "Girar izq.", 1: "Girar der.", 2: "Avanzar"}


def draw(env, title=None, show_risk=True, show_trail=True):
    import matplotlib
    if env.render_mode != "human":
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    n = env.size
    if env._fig is None:
        env._fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
    fig = env._fig
    fig.clf()
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.82])
    ax.set_xlim(-0.03, n + 0.03)
    ax.set_ylim(n + 0.03, -0.03)
    ax.set_aspect("equal")
    ax.axis("off")

    if show_risk and env.outcome is None:
        for r in range(n):
            for c in range(n):
                if (r, c) not in env.obstacles and env._threat((r, c)) and (r, c) != env.goal:
                    ax.add_patch(Rectangle((c, r), 1, 1, color=RISK, lw=0))
    if show_trail:
        for r, c in set(env.trail[:-1]):
            ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, color=TRAIL, lw=0))
    gr, gc = env.goal
    ax.add_patch(Rectangle((gc, gr), 1, 1, color=GOAL, lw=0))
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color=GRID, lw=0.8)
        ax.plot([k, k], [0, n], color=GRID, lw=0.8)
    for r, c in env.obstacles:
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.34, color=OBSTACLE))
    r, c = env.agent
    angle = {0: 90, 1: 0, 2: -90, 3: 180}[env.heading]
    pts = np.array([[0.36, 0], [-0.3, 0.3], [-0.3, -0.3]])
    rot = np.deg2rad(angle)
    rotm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    pts = pts @ rotm.T * np.array([1, -1]) + np.array([c + 0.5, r + 0.5])
    color = "#C0392B" if env.outcome == "colision" else AGENT
    ax.add_patch(Polygon(pts, closed=True, color=color))

    status = {"meta": "¡Llegó a la meta!", "colision": "Chocó con un obstáculo"}.get(env.outcome, "")
    action = LABELS.get(env.last_action, "Inicio")
    fig.text(0.04, 0.95, title or "Cuarto 8x8 con obstáculos móviles", fontsize=15, weight="bold", color=INK)
    fig.text(0.04, 0.905, f"Paso {env.steps}  ·  Acción: {action}  ·  Recompensa {env.last_reward:+.0f}  ·  "
             f"Acumulada {env.total_reward:+.0f}", fontsize=10.5, color=MUTED)
    if status:
        fig.text(0.96, 0.95, status, fontsize=13, weight="bold", ha="right",
                 color="#1B7F4B" if env.outcome == "meta" else "#C0392B")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if env.render_mode == "human":
        plt.pause(1 / env.metadata["render_fps"])
        return None
    return frame
