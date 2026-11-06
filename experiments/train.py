"""python experiments/train.py --episodes 100 --eval-episodes 10"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rl_warehouse.env import WarehouseEnv
from rl_warehouse.agents import TabularAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/logs/comparison.csv"))
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes) < 1:
        parser.error("El número de episodios debe ser positivo")
    rows = []
    for algorithm in ["q_learning", "sarsa", "monte_carlo"]:
        env = WarehouseEnv(args.config)
        gamma = env.config.get("episodio", {}).get("gamma", 0.99)
        agent = TabularAgent(env.action_space.n, algorithm, gamma=gamma, seed=args.seed)
        for i in range(args.episodes):
            agent.episode(env, seed=args.seed+i)
        for i in range(args.eval_episodes):
            rows.append(dict(algorithm=algorithm, episode=i, **agent.episode(env, seed=args.seed+args.episodes+i, train=False)))
        model = Path("results/models") / f"{algorithm}.npz"
        model.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model, states=np.array(list(agent.q)), values=np.array(list(agent.q.values())))
        selected = [row for row in rows if row["algorithm"] == algorithm]
        print(json.dumps(dict(algorithm=algorithm, mean_reward=float(np.mean([r["reward"] for r in selected])), visited_states=len(agent.q))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
