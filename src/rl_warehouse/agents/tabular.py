"""Control tabular (Q-learning, SARSA, Monte Carlo) con tablas dispersas y máscara de acciones."""
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGORITHMS = ("q_learning", "sarsa", "monte_carlo")


class TabularAgent:
    """Agente tabular epsilon-greedy.

    La Q-tabla es un diccionario {estado (tupla): vector de valores por acción}; solo
    se almacenan los estados visitados. epsilon decae multiplicativamente por episodio
    desde ``epsilon`` hasta ``epsilon_min``.
    """

    def __init__(self, n_actions, algorithm="q_learning", alpha=0.1, gamma=0.99, epsilon=0.1,
                 epsilon_min=None, epsilon_decay=1.0, seed=0, time_unit=None):
        if algorithm not in ALGORITHMS:
            raise ValueError("Algoritmo desconocido")
        self.n_actions = n_actions
        self.algorithm, self.alpha, self.gamma = algorithm, alpha, gamma
        # time_unit (min): descuento semi-Markov gamma**(minutos transcurridos / time_unit).
        # None: un factor gamma por decisión.
        self.time_unit = time_unit
        self.epsilon = epsilon
        self.epsilon_min = epsilon if epsilon_min is None else epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.counts = defaultdict(int)
        self.episodes_trained = 0

    # ------------------------------------------------------------- política
    def act(self, state, mask, explore=True):
        actions = np.flatnonzero(mask)
        if not len(actions):
            raise ValueError("No hay acciones aplicables")
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        values = self.q[tuple(state)][actions]
        return int(self.rng.choice(actions[values == values.max()]))

    def greedy(self, state, mask):
        """Acción con mayor Q sin exploración ni desempate aleatorio (para demos)."""
        actions = np.flatnonzero(mask)
        key = tuple(int(x) for x in state)
        if key not in self.q:
            return None
        return int(actions[np.argmax(self.q[key][actions])])

    # --------------------------------------------------------- entrenamiento
    def episode(self, env, seed=None, train=True):
        state, info = env.reset(seed=seed)
        action = self.act(state, info["action_mask"], train)
        history, total = [], 0.0
        while True:
            nxt, reward, done, truncated, next_info = env.step(action)
            done = done or truncated
            next_action = None if done else self.act(nxt, next_info["action_mask"], train)
            key = tuple(int(x) for x in state)
            discount = self._discount(next_info["time"] - info["time"])
            history.append((key, action, reward, discount))
            if train and self.algorithm != "monte_carlo":
                if done:
                    future = 0.0
                elif self.algorithm == "sarsa":
                    future = self.q[tuple(nxt)][next_action]
                else:
                    future = self.q[tuple(nxt)][next_info["action_mask"]].max()
                self.q[key][action] += self.alpha * (reward + discount * future - self.q[key][action])
            total += reward
            if done:
                break
            state, action, info = nxt, next_action, next_info
        if train and self.algorithm == "monte_carlo":
            returns, value = [0.0] * len(history), 0.0
            for i in reversed(range(len(history))):
                value = history[i][2] + history[i][3] * value
                returns[i] = value
            seen = set()
            for (key, action, _, _), value in zip(history, returns):
                if (key, action) not in seen:
                    seen.add((key, action))
                    self.counts[(key, action)] += 1
                    self.q[key][action] += (value - self.q[key][action]) / self.counts[(key, action)]
        if train:
            self.episodes_trained += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return dict(reward=total, steps=len(history), **next_info["metrics"])

    def _discount(self, elapsed):
        if self.time_unit is None:
            return self.gamma
        return self.gamma ** (max(elapsed, 0.0) / self.time_unit)

    # ------------------------------------------------------ guardar / cargar
    def save(self, path):
        """Guarda la Q-tabla y los hiperparámetros en un .npz comprimido."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.q)
        states = np.array(keys, dtype=np.int64) if keys else np.zeros((0, 0), dtype=np.int64)
        values = np.array([self.q[k] for k in keys]) if keys else np.zeros((0, self.n_actions))
        count_keys = list(self.counts)
        width = len(count_keys[0][0]) if count_keys else 0
        np.savez_compressed(
            path, states=states, values=values,
            count_states=np.array([k for k, _ in count_keys], dtype=np.int64).reshape(len(count_keys), width),
            count_actions=np.array([a for _, a in count_keys], dtype=np.int64),
            count_values=np.array([self.counts[k] for k in count_keys], dtype=np.int64),
            meta=np.array([self.algorithm, self.alpha, self.gamma, self.epsilon, self.epsilon_min,
                           self.epsilon_decay, self.n_actions, self.episodes_trained,
                           self.time_unit], dtype=object),
        )
        return path

    @classmethod
    def load(cls, path, seed=0):
        """Carga una Q-tabla guardada con ``save`` y devuelve un agente listo para usar."""
        data = np.load(path, allow_pickle=True)
        algorithm, alpha, gamma, eps, eps_min, decay, n_actions, trained, time_unit = data["meta"]
        agent = cls(int(n_actions), str(algorithm), float(alpha), float(gamma), float(eps),
                    float(eps_min), float(decay), seed, None if time_unit is None else float(time_unit))
        agent.episodes_trained = int(trained)
        for state, values in zip(data["states"], data["values"]):
            agent.q[tuple(int(x) for x in state)] = np.array(values, dtype=float)
        for state, action, count in zip(data["count_states"], data["count_actions"], data["count_values"]):
            agent.counts[(tuple(int(x) for x in state), int(action))] = int(count)
        return agent

    def to_table(self, env):
        """Q-tabla como lista de filas legibles (estado desglosado + valor por acción)."""
        names = ([f"w_{z}" for z in env.zones] + [f"c_{z}" for z in env.zones]
                 + [f"u_{z}" for z in env.zones] + ["o", "lambda"])
        rows = []
        for key, values in sorted(self.q.items()):
            row = dict(zip(names, key))
            row.update({f"Q_{a}": round(float(v), 4) for a, v in zip(env.action_names, values)})
            rows.append(row)
        return rows
