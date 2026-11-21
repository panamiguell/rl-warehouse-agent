"""Pruebas del cuarto 8x8: cada tripla (s, a, s') del documento de definición del ambiente."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_gridworld import AVANZAR, GIRAR_DER, GIRAR_IZQ, ObstacleGridEnv, TabularAgent
from rl_gridworld.baselines import POLICIES, run_policy


def scenario(agent, heading, obstacles, moved=None):
    """Fija el estado completo y, si se da, la posición de los obstáculos tras el paso."""
    env = ObstacleGridEnv()
    env.reset(seed=0)
    env.agent, env.heading, env.obstacles = agent, heading, list(obstacles)
    if moved is not None:
        env._move_obstacles = lambda: list(moved)
    return env


FAR = [(4, 4), (4, 6), (6, 2), (2, 6), (3, 3)]             # obstáculos lejos del agente


def test_gym_contract_and_initial_state():
    check_env(ObstacleGridEnv(), skip_render_check=True)
    env = ObstacleGridEnv()
    for seed in range(50):
        env.reset(seed=seed)
        assert env.agent == (0, 0) and env.heading == 0
        assert len(set(env.obstacles)) == 5 and not {(0, 0), (7, 7)} & set(env.obstacles)


def test_action_mask_walls():
    for agent, heading, ok in [((0, 0), 0, False), ((0, 0), 3, False), ((0, 0), 1, True), ((0, 0), 2, True),
                               ((7, 3), 2, False), ((3, 7), 1, False), ((3, 3), 0, True)]:
        env = scenario(agent, heading, FAR)
        assert list(env.action_mask()) == [True, True, ok]


def test_t1_t2_turns():
    env = scenario((0, 0), 0, FAR, FAR)
    obs, reward, done, _, _ = env.step(GIRAR_IZQ)
    assert env.heading == 3 and reward == -1 and not done and env.agent == (0, 0)
    env = scenario((0, 0), 0, FAR, FAR)
    obs, reward, done, _, _ = env.step(GIRAR_DER)
    assert env.heading == 1 and reward == -1 and not done


def test_t3_advance_free():
    env = scenario((0, 0), 1, FAR, FAR)
    _, reward, done, _, _ = env.step(AVANZAR)
    assert env.agent == (0, 1) and reward == -1 and not done


def test_t4_reach_goal():
    env = scenario((7, 6), 1, FAR, FAR)
    obs, reward, done, _, info = env.step(AVANZAR)
    assert done and reward == 100 and env.outcome == "meta" and not obs.any()
    assert not info["action_mask"].any()                       # T9: sin acciones en el terminal


def test_t5_collision_advancing():
    obstacles = [(2, 3)] + FAR[1:]
    env = scenario((2, 2), 1, obstacles, obstacles)           # el obstáculo se queda en (2,3)
    _, reward, done, _, _ = env.step(AVANZAR)
    assert done and reward == -100 and env.outcome == "colision"


def test_t6_collision_by_swap():
    obstacles = [(2, 3)] + FAR[1:]
    moved = [(2, 2)] + FAR[1:]                                # el obstáculo pasa a la casilla del agente
    env = scenario((2, 2), 1, obstacles, moved)
    _, reward, done, _, info = env.step(AVANZAR)
    assert done and reward == -100 and info["swap"]


def test_t7_collision_while_turning():
    obstacles = [(3, 2)] + FAR[1:]
    moved = [(2, 2)] + FAR[1:]                                # el obstáculo entra en la casilla del agente
    env = scenario((2, 2), 0, obstacles, moved)
    _, reward, done, _, _ = env.step(GIRAR_DER)
    assert done and reward == -100


def test_t8_invalid_advance_into_wall():
    env = scenario((0, 0), 0, FAR, FAR)
    _, reward, done, _, info = env.step(AVANZAR)
    assert env.agent == (0, 0) and reward == -10 and not done and info["invalid"]


def test_t10_no_step_limit_only_safety_cutoff():
    env = ObstacleGridEnv({"obstaculos": 0})
    env.reset(seed=0)
    for _ in range(999):                                      # el MDP no tiene límite de pasos
        _, reward, terminated, truncated, _ = env.step(GIRAR_DER)
        assert reward == -1 and not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(GIRAR_DER)
    assert truncated and not terminated                      # corte de seguridad del simulador


def test_agent_does_not_see_obstacles():
    env = scenario((2, 2), 1, FAR)
    first = env.observe()
    env.obstacles = [(2, 3), (3, 2), (1, 2), (2, 1), (6, 6)]  # rodeado: la observación no cambia
    assert list(first) == list(env.observe()) == [2, 2, 1]


def test_obstacles_avoid_start_and_goal_and_move_one_cell():
    env = ObstacleGridEnv()
    for seed in range(5):
        env.reset(seed=seed)
        env.obstacles = [(0, 1), (1, 0), (6, 7), (7, 6), (3, 3)]  # junto a la salida y a la meta
        for _ in range(300):
            before = list(env.obstacles)
            after = env._move_obstacles()
            assert not {(0, 0), (7, 7)} & set(after) and len(set(after)) == 5
            assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 for a, b in zip(before, after))
            env.obstacles = after


def test_obstacles_random_walk_reaches_board():
    env = ObstacleGridEnv()
    env.reset(seed=3)
    for _ in range(500):
        before = list(env.obstacles)
        after = env._move_obstacles()
        assert (7, 7) not in after and len(set(after)) == 5
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 for a, b in zip(before, after))
        env.obstacles = after


@pytest.mark.parametrize("name", list(POLICIES))
def test_baselines_run(name):
    policy = POLICIES[name]
    env = ObstacleGridEnv()
    result = run_policy(env, policy, seed=1)
    assert result["success"] + result["collision"] + result["truncated"] == 1


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_agent_learns_and_roundtrip(algorithm, tmp_path):
    env = ObstacleGridEnv()
    agent = TabularAgent(3, algorithm, epsilon=0.5, seed=0)
    for i in range(50):
        agent.episode(env, seed=i)
    path = agent.save(tmp_path / "q.npz")
    loaded = TabularAgent.load(path)
    assert loaded.algorithm == algorithm and loaded.episodes_trained == 50
    assert set(loaded.q) == set(agent.q)
    assert all(np.allclose(loaded.q[k], agent.q[k]) for k in agent.q)
