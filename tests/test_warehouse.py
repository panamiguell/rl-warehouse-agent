import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from rl_warehouse.env import WarehouseEnv, Order, Line
from rl_warehouse.agents import TabularAgent


def quiet(**sim):
    return WarehouseEnv({"simulacion": {"ordenes_por_minuto": 0, **sim}})


def test_gym_contract():
    check_env(WarehouseEnv(), skip_render_check=True)


def test_invalid_action_preserves_state():
    env = quiet()
    state, _ = env.reset(seed=1)
    nxt, reward, done, _, _ = env.step(0)
    assert np.array_equal(state, nxt) and reward == -50 and not done and env.time == 0


def test_consolidation_and_immediate_decision():
    env = quiet()
    env.orders = [Order(i, 0, 100, [Line(0, 1)]) for i in range(5)] + [Order(5, 0, 100, [Line(1, 2)])]
    _, reward, _, _, info = env.step(0)
    assert reward == 10 and info["tau"] == 0
    assert env.stats["visits"] == 1 and env.stats["picked_lines"] == 5
    assert len(env.jobs[0][1]) == 5


def test_deadline_once_and_terminal_pending():
    env = WarehouseEnv({"episodio": {"duracion_turno_min": 6}, "simulacion": {"ordenes_por_minuto":0}})
    env.orders = [Order(0, 0, 1, [Line(0, 1)])]
    _, reward, _, _, info = env.step(env.wait)
    assert reward == -30 and info["n_venc"] == 1
    _, reward, _, _, info = env.step(env.wait)
    assert reward == -10 and info["n_venc"] == 0
    _, reward, done, _, info = env.step(env.wait)
    assert done and reward == -40 and info["n_pend"] == 1
    assert not info["action_mask"].any()
    with pytest.raises(RuntimeError):
        env.step(env.wait)


def test_capacity_and_deadline_priority():
    env = quiet()
    env.orders = [Order(i, 0, 100-i, [Line(0,i)]) for i in range(25)]
    env.step(0)
    assigned = [o.id for o in env.orders if o.lines[0].assigned]
    assert assigned == list(range(5,25))


def test_completion_and_seed_reproducibility():
    env = quiet()
    env.orders = [Order(0, 0, 100, [Line(0, 1)])]
    _, reward, _, _, info = env.step(0)
    assert reward == 4 and info["n_ok"] == 1
    a, b = WarehouseEnv(), WarehouseEnv()
    a.reset(seed=7)
    b.reset(seed=7)
    for _ in range(30):
        sa, ra, da, _, _ = a.step(a.wait)
        sb, rb, db, _, _ = b.step(b.wait)
        assert np.array_equal(sa,sb) and (ra,da,a.time) == (rb,db,b.time)


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa", "monte_carlo"])
def test_agents_complete_episode(algorithm):
    env = WarehouseEnv({"episodio":{"duracion_turno_min":20}})
    agent = TabularAgent(env.action_space.n, algorithm)
    result = agent.episode(env, seed=1)
    assert np.isfinite(result["reward"]) and env.terminated
