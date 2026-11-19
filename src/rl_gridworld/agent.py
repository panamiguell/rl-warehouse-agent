"""Agentes tabulares (Q-learning y SARSA) con máscara de acciones y Q-tabla persistente."""
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGORITHMS = ("q_learning", "sarsa")


class TabularAgent:
    """Q-tabla dispersa {estado (tupla): valores de las 3 acciones} con política ε-greedy sobre A(s)."""

    def __init__(self, n_actions=3, algorithm="q_learning", alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=1.0, seed=0):
        if algorithm not in ALGORITHMS:
            raise ValueError("Algoritmo desconocido")
        self.n_actions, self.algorithm = n_actions, algorithm
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.episodes_trained = 0

    def values(self, state):
        key = tuple(int(x) for x in state)
        return self.q[key] if key in self.q else np.zeros(self.n_actions)

    def act(self, state, mask, explore=True):
        actions = np.flatnonzero(mask)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        values = self.values(state)[actions]
        return int(self.rng.choice(actions[values == values.max()]))

    def episode(self, env, seed=None, train=True):
        state, info = env.reset(seed=seed)
        action = self.act(state, info["action_mask"], train)
        total = 0.0
        while True:
            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = None if done else self.act(nxt, info["action_mask"], train)
            if train:
                key = tuple(int(x) for x in state)
                if terminated:
                    future = 0.0                 # meta o colisión: no hay valor futuro
                elif self.algorithm == "sarsa" and next_action is not None:
                    future = self.values(nxt)[next_action]
                else:
                    # Q-learning, o corte por límite de pasos (s' no es terminal: se sigue estimando).
                    future = self.values(nxt)[info["action_mask"]].max()
                self.q[key][action] += self.alpha * (reward + self.gamma * future - self.q[key][action])
            total += reward
            if done:
                break
            state, action = nxt, next_action
        if train:
            self.episodes_trained += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return dict(reward=total, truncated=int(truncated), **info["metrics"])

    # ------------------------------------------------------ guardar / cargar
    def save(self, path):
        """Guarda la Q-tabla y los hiperparámetros en un .npz comprimido."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.q)
        np.savez_compressed(
            path,
            states=np.array(keys, dtype=np.int64).reshape(len(keys), -1),
            values=np.array([self.q[k] for k in keys]).reshape(len(keys), self.n_actions),
            meta=np.array([self.algorithm, self.alpha, self.gamma, self.epsilon, self.epsilon_min,
                           self.epsilon_decay, self.n_actions, self.episodes_trained], dtype=object),
        )
        return path

    @classmethod
    def load(cls, path, seed=0):
        """Carga una Q-tabla guardada con ``save`` y devuelve el agente listo para usar."""
        data = np.load(path, allow_pickle=True)
        algorithm, alpha, gamma, eps, eps_min, decay, n_actions, trained = data["meta"]
        agent = cls(int(n_actions), str(algorithm), float(alpha), float(gamma), float(eps), float(eps_min),
                    float(decay), seed)
        agent.episodes_trained = int(trained)
        for state, values in zip(data["states"], data["values"]):
            agent.q[tuple(int(x) for x in state)] = np.array(values, dtype=float)
        return agent

    def to_rows(self, action_names):
        names = ["fila", "columna", "orientacion", "frente", "aqui", "izquierda", "derecha"]
        rows = []
        for key in sorted(self.q):
            row = dict(zip(names, key))
            row.update({f"Q_{a}": round(float(v), 4) for a, v in zip(action_names, self.q[key])})
            rows.append(row)
        return rows
