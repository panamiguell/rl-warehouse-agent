"""Políticas de referencia (reglas fijas) para el cuarto 8x8.

Cada política recibe (env, observación, máscara, rng) y devuelve una acción de A(s).
"""
import numpy as np

from .env import AVANZAR, DELTA, GIRAR_DER, GIRAR_IZQ, LIBRE


def aleatoria(env, obs, mask, rng):
    """Acción al azar entre las aplicables (es lo que hace el agente antes de entrenar)."""
    return int(rng.choice(np.flatnonzero(mask)))


def _toward_goal(env, heading):
    """¿La orientación acerca al agente a la meta?"""
    r, c = env.agent
    dr, dc = DELTA[heading]
    gr, gc = env.goal
    return (dr > 0 and r < gr) or (dc > 0 and c < gc) or (dr < 0 and r > gr) or (dc < 0 and c > gc)


def _turn_toward_goal(env):
    left, right = (env.heading - 1) % 4, (env.heading + 1) % 4
    if _toward_goal(env, right):
        return GIRAR_DER
    if _toward_goal(env, left):
        return GIRAR_IZQ
    return GIRAR_DER


def directa(env, obs, mask, rng):
    """Va a la meta por el camino más corto, sin mirar los obstáculos."""
    if mask[AVANZAR] and _toward_goal(env, env.heading):
        return AVANZAR
    return _turn_toward_goal(env)


def cautelosa(env, obs, mask, rng):
    """Como la directa, pero solo avanza si la casilla de enfrente está libre y sin riesgo;
    si no, gira (hacia la meta si puede)."""
    if mask[AVANZAR] and _toward_goal(env, env.heading) and obs[3] == LIBRE:
        return AVANZAR
    return _turn_toward_goal(env)


POLICIES = {"aleatoria": aleatoria, "directa": directa, "cautelosa": cautelosa}


def run_policy(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs, info["action_mask"], rng))
        total += reward
        if terminated or truncated:
            return dict(reward=total, truncated=int(truncated), **info["metrics"])
