"""Visualización del centro de distribución con matplotlib (para el video de la entrega).

``draw(env)`` dibuja el estado actual y devuelve un arreglo RGB (modo ``rgb_array``) o
actualiza una ventana (modo ``human``).
"""
import numpy as np

URGENCY = {0: ("#D9DDE3", "sin líneas"), 1: ("#2E8B57", "holgura > 60 min"),
           2: ("#E0A100", "holgura 15–60 min"), 3: ("#C0392B", "crítica ≤ 15 min")}
INK, MUTED, BG = "#1F2933", "#6B7280", "#FFFFFF"
ACCENT = "#1F4E79"


def _clock(minutes, start_hour=6):
    total = int(minutes) + start_hour * 60
    return f"{total // 60:02d}:{total % 60:02d}"


def draw(env, start_hour=6):
    import matplotlib
    if env.render_mode != "human":
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    if env._fig is None:
        env._fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig = env._fig
    fig.clf()
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 72)
    ax.axis("off")

    state = env._zone_view() if env.terminated else env._state()
    n = env.n
    w, c, u = state[:n], state[n:2 * n], state[2 * n:3 * n]
    pending = {z: env._pending(z) for z in range(n)}

    # Encabezado
    ax.text(4, 67.5, "Consolidación de picking en un CD", fontsize=19, weight="bold", color=INK, va="center")
    ax.text(4, 63.5, f"Turno {_clock(0, start_hour)}–{_clock(env.shift, start_hour)}  ·  "
            f"{len(env.open)} órdenes abiertas  ·  demanda últimos 15 min: {len(env.arrivals)} órdenes",
            fontsize=10.5, color=MUTED, va="center")
    ax.text(124, 67.5, _clock(env.time, start_hour), fontsize=26, weight="bold", color=ACCENT, ha="right", va="center")
    progress = min(env.time / env.shift, 1)
    ax.add_patch(Rectangle((84, 62.6), 40, 1.2, color="#E5E7EB", lw=0))
    ax.add_patch(Rectangle((84, 62.6), 40 * progress, 1.2, color=ACCENT, lw=0))

    # Zonas de racks
    cols = min(n, 4)
    zh, gap, x0, y0, width = 40, 2.5, 4, 14, 80
    zw = (width - gap * (cols - 1)) / cols          # las zonas ocupan el mismo ancho con 2 o 4 zonas
    for z in range(n):
        x = x0 + (z % cols) * (zw + gap)
        color, label = URGENCY[int(u[z])]
        ax.add_patch(FancyBboxPatch((x, y0), zw, zh, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#F7F8FA", ec=color, lw=3))
        ax.text(x + zw / 2, y0 + zh - 3.5, f"Zona {env.zones[z]}", ha="center", fontsize=13, weight="bold", color=INK)
        ax.text(x + zw / 2, y0 + zh - 7, label, ha="center", fontsize=8.5,
                color=MUTED if u[z] == 0 else color, weight="bold")
        # Racks: 30 ubicaciones (SKU), se resaltan las que tienen líneas pendientes.
        demand = {}
        for o, line in pending[z]:
            demand.setdefault(line.sku, set()).add(o.id)
        size = 2.6
        per_row = max(1, int((zw - 3) // (size + 0.4)))
        left = x + (zw - per_row * (size + 0.4) + 0.4) / 2
        for k in range(env.skus):
            rx = left + (k % per_row) * (size + 0.4)
            ry = y0 + zh - 12 - (k // per_row) * (size + 0.5)
            if ry < y0 + 9:
                break
            orders = len(demand.get(k, ()))
            fc = "#E5E7EB" if not orders else ("#9DB7D5" if orders == 1 else ACCENT)
            ax.add_patch(Rectangle((rx, ry), size, size, fc=fc, ec="white", lw=0.6))
            if orders > 1:
                ax.text(rx + size / 2, ry + size / 2, str(orders), ha="center", va="center", fontsize=7,
                        color="white", weight="bold")
        ax.text(x + 1.5, y0 + 5.2, f"{len(pending[z])} líneas pendientes", fontsize=9, color=INK)
        top = max((len(v) for v in demand.values()), default=0)
        ax.text(x + 1.5, y0 + 2, f"SKU más pedido: {top} órdenes", fontsize=8, color=MUTED)

    # Operarios
    px = x0 + width + 2
    ax.text(px, 52, "Operarios", fontsize=13, weight="bold", color=INK)
    jobs = sorted(env.jobs, key=lambda j: j[0])
    for k in range(env.workers):
        y = 46 - k * 6
        busy = k < len(jobs)
        ax.add_patch(FancyBboxPatch((px, y - 2), 32, 4.6, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=ACCENT if busy else "#F3F4F6", ec="none"))
        if busy:
            finish, pairs = jobs[k]
            zones = "".join(sorted({env.zones[l.zone] for _, l in pairs}))
            locs = len({(l.zone, l.sku) for _, l in pairs})
            txt = f"Op {k + 1}: zona {zones} · {len(pairs)} líneas / {locs} visitas · libre {_clock(finish, start_hour)}"
            ax.text(px + 1.2, y + 0.3, txt, fontsize=8.6, color="white", va="center")
        else:
            ax.text(px + 1.2, y + 0.3, f"Op {k + 1}: libre", fontsize=9, color=MUTED, va="center")

    # Última decisión
    y = 46 - env.workers * 6 - 3
    action = "—" if env.last_action is None else env.action_names[env.last_action]
    ax.text(px, y, "Última decisión", fontsize=13, weight="bold", color=INK)
    ax.text(px, y - 4.5, action, fontsize=16, weight="bold", color=ACCENT)
    ax.text(px + 32, y - 4.5, f"R = {env.last_reward:+.0f}", fontsize=12, color=INK, ha="right")

    # Métricas
    s = env.stats
    lines = max(s["picked_lines"], 1)
    metrics = [("A tiempo", s["on_time"]), ("Vencidas", s["expired"]), ("Visitas ahorradas", s["saved_visits"]),
               ("Visitas / línea", f"{s['visits'] / lines:.2f}"), ("Viajes", s["trips"])]
    for i, (name, value) in enumerate(metrics):
        mx = 4 + i * 24.5
        ax.text(mx, 8, str(value), fontsize=17, weight="bold", color=INK)
        ax.text(mx, 4.5, name, fontsize=9, color=MUTED)

    # Leyenda de racks
    ax.add_patch(Rectangle((px, 56.1), 1.8, 1.8, fc="#9DB7D5", ec="none"))
    ax.text(px + 2.5, 57, "1 orden pide el SKU", fontsize=8, color=MUTED, va="center")
    ax.add_patch(Rectangle((px + 17, 56.1), 1.8, 1.8, fc=ACCENT, ec="none"))
    ax.text(px + 19.5, 57, "varias (se consolida)", fontsize=8, color=MUTED, va="center")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if env.render_mode == "human":
        plt.pause(1 / env.metadata["render_fps"])
        return None
    return frame
