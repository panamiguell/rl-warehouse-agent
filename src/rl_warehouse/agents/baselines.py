"""Políticas de referencia (reglas fijas) para comparar contra los agentes de RL.

Cada política recibe (env, estado, máscara, rng) y devuelve una acción aplicable.
"""
import numpy as np


def _zones(env, mask):
    return [z for z in range(env.n) if mask[z]]


def aleatoria(env, s, mask, rng):
    """Elige uniformemente entre las acciones aplicables."""
    return int(rng.choice(np.flatnonzero(mask)))


def esperar_siempre(env, s, mask, rng):
    """Nunca asigna trabajo: cota inferior."""
    return env.wait


def urgente_siempre(env, s, mask, rng):
    """Recoge orden por orden (sin consolidar), la más urgente primero."""
    return env.urgent if mask[env.urgent] else env.wait


def lote_mas_urgente(env, s, mask, rng):
    """Despacha de inmediato el lote de la zona más urgente (desempate: consolidación, WIP)."""
    zones = _zones(env, mask)
    if not zones:
        return env.wait
    u, c = s[2 * env.n:3 * env.n], s[env.n:2 * env.n]
    return max(zones, key=lambda z: (u[z], c[z], s[z]))


def lote_mas_consolidado(env, s, mask, rng):
    """Despacha de inmediato el lote de la zona con más consolidación (desempate: WIP)."""
    zones = _zones(env, mask)
    if not zones:
        return env.wait
    c = s[env.n:2 * env.n]
    return max(zones, key=lambda z: (c[z], s[z]))


def umbral_consolidacion(env, s, mask, rng):
    """Regla con espera: si nada es crítico y ninguna zona alcanza un lote mediano
    (w_z >= 2) ni consolidación alta (c_z = 2), espera a que se acumulen líneas;
    si no, despacha la zona más urgente."""
    zones = _zones(env, mask)
    if not zones:
        return env.wait
    w, c, u = s[:env.n], s[env.n:2 * env.n], s[2 * env.n:3 * env.n]
    if max(u) < 3 and all(w[z] < 2 and c[z] < 2 for z in zones):
        return env.wait
    return lote_mas_urgente(env, s, mask, rng)


POLICIES = {
    "aleatoria": aleatoria,
    "esperar_siempre": esperar_siempre,
    "urgente_siempre": urgente_siempre,
    "lote_mas_urgente": lote_mas_urgente,
    "lote_mas_consolidado": lote_mas_consolidado,
    "umbral_consolidacion": umbral_consolidacion,
}


def run_policy(env, policy, seed):
    """Ejecuta un episodio completo con una regla fija y devuelve sus métricas."""
    state, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    while True:
        state, reward, done, truncated, info = env.step(policy(env, state, info["action_mask"], rng))
        total += reward
        if done or truncated:
            return dict(reward=total, **info["metrics"])
