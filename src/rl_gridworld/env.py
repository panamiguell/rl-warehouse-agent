"""Cuarto 8x8 con obstáculos móviles (ambiente del proyecto del curso).

El agente (triángulo) empieza en (0,0) mirando al norte y debe llegar a la meta (7,7) sin
chocar con 5 obstáculos que se mueven al azar en cada paso, al mismo tiempo que el agente
actúa. Acciones: girar 90° a la izquierda, girar 90° a la derecha y avanzar a la casilla de
enfrente (solo si existe). La definición formal está en docs/proyecto_8x8.

Coordenadas: (fila, columna), con (0,0) arriba a la izquierda y (7,7) abajo a la derecha.
Orientación: 0 = Norte (arriba), 1 = Este, 2 = Sur, 3 = Oeste.
"""
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

GIRAR_IZQ, GIRAR_DER, AVANZAR = 0, 1, 2
ACTION_NAMES = ["GIRAR_IZQ", "GIRAR_DER", "AVANZAR"]
HEADINGS = ["N", "E", "S", "O"]
DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]           # desplazamiento de cada orientación
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]    # obstáculo: quedarse o 4 vecinos

# Percepción de una casilla (sensor frontal)
LIBRE, RIESGO, OCUPADA, PARED = 0, 1, 2, 3

DEFAULTS = {
    "tamano": 8,
    "obstaculos": 5,
    "max_pasos": 200,
    "recompensas": {
        "meta": 100,
        "colision": -100,
        "paso": -1,            # cada AVANZAR que no termina el episodio
        "giro": -1,            # cada giro que no termina el episodio
        "accion_invalida": -10,
    },
}


class ObstacleGridEnv(gym.Env):
    """Ambiente del cuarto 8x8.

    Estado completo (``full_state``): posición y orientación del agente y posiciones de los
    5 obstáculos, más la condición terminal. Observación para la Q-tabla (``observe``):
    (fila, columna, orientación, frente, aquí, izquierda, derecha), derivada del estado completo.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        if isinstance(config, (str, Path)):
            with open(config, encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        cfg = deepcopy(DEFAULTS)
        user = deepcopy(config or {})
        cfg["recompensas"].update(user.pop("recompensas", {}))
        cfg.update(user)
        self.config = cfg
        self.size = cfg["tamano"]
        self.n_obstacles = cfg["obstaculos"]
        self.max_steps = cfg["max_pasos"]
        self.rewards = cfg["recompensas"]
        if self.size < 3 or not 0 <= self.n_obstacles <= self.size ** 2 - 2 or self.max_steps < 1:
            raise ValueError("Tamaño, número de obstáculos o límite de pasos inválidos")
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([self.size, self.size, 4, 4, 2, 2, 2])
        self.action_names = ACTION_NAMES
        self._fig = None
        self.reset()

    # ------------------------------------------------------------------ ciclo
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.start
        self.heading = 0                                  # Norte, como en la figura del enunciado
        free = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in (self.start, self.goal)]
        picks = self.np_random.choice(len(free), size=self.n_obstacles, replace=False)
        self.obstacles = [free[i] for i in picks]
        self.steps = 0
        self.outcome = None                               # None, "meta" o "colision"
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.trail = [self.agent]
        return self.observe(), self._info()

    # ----------------------------------------------------------------- estado
    def inside(self, cell):
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def ahead(self, heading=None, cell=None):
        r, c = cell or self.agent
        dr, dc = DELTA[self.heading if heading is None else heading]
        return r + dr, c + dc

    def _threat(self, cell):
        """Obstáculo en la casilla o en una vecina (podría entrar en el próximo paso)."""
        return any(abs(o[0] - cell[0]) + abs(o[1] - cell[1]) <= 1 for o in self.obstacles)

    def sense(self, cell):
        if not self.inside(cell):
            return PARED
        if cell in self.obstacles:
            return OCUPADA
        return RIESGO if self._threat(cell) else LIBRE

    def observe(self):
        """Estado tabular φ(s) = (fila, col, θ, frente, aquí, izq, der)."""
        if self.outcome is not None:
            return np.zeros(7, dtype=np.int64)
        r, c = self.agent
        side = lambda h: int(self.sense(self.ahead(h)) in (RIESGO, OCUPADA))
        here = int(any(abs(o[0] - r) + abs(o[1] - c) == 1 for o in self.obstacles))
        return np.array([r, c, self.heading, self.sense(self.ahead()), here,
                         side((self.heading - 1) % 4), side((self.heading + 1) % 4)], dtype=np.int64)

    def full_state(self):
        return dict(agente=self.agent, orientacion=HEADINGS[self.heading], obstaculos=list(self.obstacles),
                    terminal=self.outcome)

    def action_mask(self):
        """A(s): girar siempre; avanzar solo si la casilla de enfrente existe; nada en terminales."""
        if self.outcome is not None:
            return np.zeros(3, dtype=bool)
        return np.array([True, True, self.inside(self.ahead())])

    def _info(self, **extra):
        return dict(time=self.steps, action_mask=self.action_mask(), state=self.full_state(),
                    metrics=dict(steps=self.steps, success=int(self.outcome == "meta"),
                                 collision=int(self.outcome == "colision")), **extra)

    # -------------------------------------------------------------- dinámica
    def _move_obstacles(self):
        """Cada obstáculo elige al azar entre quedarse o ir a un vecino libre (no la meta)."""
        new = list(self.obstacles)
        for i in self.np_random.permutation(len(new)):
            taken = set(new[:i] + new[i + 1:])
            options = []
            for dr, dc in MOVES:
                cell = (new[i][0] + dr, new[i][1] + dc)
                if self.inside(cell) and cell != self.goal and (cell not in taken or (dr, dc) == (0, 0)):
                    options.append(cell)
            new[i] = options[self.np_random.integers(len(options))]
        return new

    def step(self, action):
        if self.outcome is not None:
            raise RuntimeError("El episodio terminó; llama reset()")
        action = int(action)
        r = self.rewards
        old_agent, old_obstacles = self.agent, list(self.obstacles)
        invalid = action == AVANZAR and not self.inside(self.ahead())
        if action == GIRAR_IZQ:
            self.heading = (self.heading - 1) % 4
        elif action == GIRAR_DER:
            self.heading = (self.heading + 1) % 4
        elif not invalid:
            self.agent = self.ahead()
        # Los obstáculos se mueven al mismo tiempo que el agente.
        self.obstacles = self._move_obstacles()
        self.steps += 1
        swapped = self.agent != old_agent and any(
            o_old == self.agent and o_new == old_agent for o_old, o_new in zip(old_obstacles, self.obstacles))
        if self.agent in self.obstacles or swapped:
            self.outcome, reward = "colision", r["colision"]
        elif self.agent == self.goal:
            self.outcome, reward = "meta", r["meta"]
        elif invalid:
            reward = r["accion_invalida"]
        else:
            reward = r["giro"] if action != AVANZAR else r["paso"]
        terminated = self.outcome is not None
        truncated = not terminated and self.steps >= self.max_steps
        self.last_action, self.last_reward = action, float(reward)
        self.total_reward += reward
        self.trail.append(self.agent)
        if self.render_mode == "human":
            self.render()
        return self.observe(), float(reward), terminated, truncated, self._info(swap=swapped, invalid=invalid)

    # ----------------------------------------------------------------- render
    def render(self):
        from .render import draw
        return draw(self)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
