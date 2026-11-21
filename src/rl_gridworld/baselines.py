"""Políticas de referencia (reglas fijas) para el cuarto 8x8.

Cada política recibe (env, observación, máscara, rng) y devuelve una acción de A(s).
"""
import numpy as np

from .env import AVANZAR, GIRAR_DER, GIRAR_IZQ


def aleatoria(env, obs, mask, rng):
    """Acción al azar entre las aplicables (es lo que hace el agente antes de entrenar)."""
    return int(rng.choice(np.flatnonzero(mask)))


def _go(env, heading):
    """Avanza si ya mira hacia ``heading``; si no, gira por el lado más corto."""
    if env.heading == heading:
        return AVANZAR
    return GIRAR_DER if (heading - env.heading) % 4 in (1, 2) else GIRAR_IZQ


def borde(env, obs, mask, rng):
    """Camino más corto por el borde: hacia el este por la fila 0 y luego hacia el sur por la columna 7."""
    r, c = env.agent
    return _go(env, 1 if c < env.goal[1] else 2)


def escalera(env, obs, mask, rng):
    """Camino más corto en escalera por la diagonal: alterna este y sur (cruza el centro del cuarto)."""
    r, c = env.agent
    if c >= env.goal[1]:
        return _go(env, 2)
    if r >= env.goal[0]:
        return _go(env, 1)
    return _go(env, 1 if c <= r else 2)


POLICIES = {"aleatoria": aleatoria, "borde": borde, "escalera": escalera}


def run_policy(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs, info["action_mask"], rng))
        total += reward
        if terminated or truncated:
            return dict(reward=total, truncated=int(truncated), **info["metrics"])
