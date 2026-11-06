"""Control tabular con tablas dispersas y enmascaramiento de acciones."""
from collections import defaultdict
import numpy as np


class TabularAgent:
    def __init__(self, n_actions, algorithm="q_learning", alpha=0.1, gamma=0.99, epsilon=0.1, seed=0):
        if algorithm not in {"q_learning", "sarsa", "monte_carlo"}:
            raise ValueError("Algoritmo desconocido")
        self.algorithm, self.alpha, self.gamma, self.epsilon = algorithm, alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.counts = defaultdict(int)

    def act(self, state, mask, explore=True):
        actions = np.flatnonzero(mask)
        if not len(actions):
            raise ValueError("No hay acciones aplicables")
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        values = self.q[tuple(state)][actions]
        return int(self.rng.choice(actions[values == values.max()]))

    def episode(self, env, seed=None, train=True):
        state, info = env.reset(seed=seed)
        action = self.act(state, info["action_mask"], train)
        history, total = [], 0.0
        while True:
            nxt, reward, done, truncated, next_info = env.step(action)
            done = done or truncated
            next_action = None if done else self.act(nxt, next_info["action_mask"], train)
            key = tuple(state)
            history.append((key, action, reward))
            if train and self.algorithm != "monte_carlo":
                future = 0 if done else self.q[tuple(nxt)][next_action] if self.algorithm == "sarsa" else max(self.q[tuple(nxt)][next_info["action_mask"]])
                self.q[key][action] += self.alpha*(reward+self.gamma*future-self.q[key][action])
            total += reward
            if done:
                break
            state, action = nxt, next_action
        if train and self.algorithm == "monte_carlo":
            returns, value = [0.0]*len(history), 0.0
            for i in reversed(range(len(history))):
                value = history[i][2]+self.gamma*value
                returns[i] = value
            seen = set()
            for (key, action, _), value in zip(history, returns):
                pair = (key, action)
                if pair not in seen:
                    seen.add(pair)
                    self.counts[pair] += 1
                    self.q[key][action] += (value-self.q[key][action])/self.counts[pair]
        return dict(reward=total, **next_info["metrics"])
