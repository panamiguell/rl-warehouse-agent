import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_warehouse.agents import POLICIES, TabularAgent, run_policy
from rl_warehouse.env import Line, Order, WarehouseEnv

CONFIG = "configs/env_default.yaml"


def quiet(**sim):
    """Ambiente sin llegadas: las órdenes se inyectan a mano."""
    return WarehouseEnv({"simulacion": {"ordenes_por_minuto": 0, **sim}})


def inject(env, orders):
    env.orders = list(orders)
    env.open = list(orders)
    env.unassigned = sum(len(o.lines) for o in orders)


def test_gym_contract():
    check_env(WarehouseEnv(), skip_render_check=True)


def test_invalid_action_preserves_state():
    env = quiet()
    state, _ = env.reset(seed=1)
    nxt, reward, done, _, _ = env.step(0)
    assert np.array_equal(state, nxt) and reward == -50 and not done and env.time == 0


def test_consolidation_rewards_saved_visits():
    env = quiet()
    inject(env, [Order(i, 0, 100, [Line(0, 1)]) for i in range(5)] + [Order(5, 0, 100, [Line(1, 2)])])
    _, reward, _, _, info = env.step(0)
    # 5 líneas del mismo SKU en una visita: 4 visitas ahorradas (+2 c/u) y un viaje (-1).
    assert reward == 2 * 4 - 1 and info["n_ahorr"] == 4 and info["tau"] == 0
    assert env.stats["visits"] == 1 and env.stats["picked_lines"] == 5
    assert len(env.jobs[0][1]) == 5


def test_clock_advances_at_most_delta_with_free_workers():
    env = quiet()
    inject(env, [Order(0, 0, 100, [Line(0, 1)])])
    _, reward, _, _, info = env.step(0)
    # Quedan operarios libres y no hay más trabajo: el reloj avanza Δt = 2, no hasta la finalización.
    assert info["tau"] == 1 and env.time == pytest.approx(2) and reward == -1
    env.step(env.wait)                      # t = 4
    _, reward, _, _, info = env.step(env.wait)  # termina en t = 5.2 (4 + 1 visita + 0.2 por línea)
    assert env.time == pytest.approx(5.2)
    assert info["n_ok"] == 1 and env.orders[0].completed and reward == 3


def test_deadline_once_and_terminal_pending():
    env = WarehouseEnv({"episodio": {"duracion_turno_min": 6},
                        "simulacion": {"ordenes_por_minuto": 0, "corte_llegadas_min": 0}})
    inject(env, [Order(0, 0, 1, [Line(0, 1)])])
    _, reward, _, _, info = env.step(env.wait)
    # Esperar con orden crítica (-10), vence (-20), zona sigue crítica (-1), 1 línea x 2 min de WIP (-0.1).
    assert reward == pytest.approx(-31.1) and info["n_venc"] == 1
    _, reward, _, _, info = env.step(env.wait)
    assert reward == pytest.approx(-11.1) and info["n_venc"] == 0
    _, reward, done, _, info = env.step(env.wait)
    assert done and reward == pytest.approx(-10 - 0.1 - 30) and info["n_pend"] == 1
    assert not info["action_mask"].any()
    with pytest.raises(RuntimeError):
        env.step(env.wait)


def test_capacity_and_deadline_priority():
    env = quiet()
    inject(env, [Order(i, 0, 100 - i, [Line(0, i)]) for i in range(25)])
    env.step(0)
    assigned = [o.id for o in env.orders if o.lines[0].assigned]
    assert assigned == list(range(5, 25))


def test_urgent_without_urgency_is_penalized():
    env = quiet()
    inject(env, [Order(0, 0, 300, [Line(0, 1), Line(1, 2)])])
    _, reward, _, _, _ = env.step(env.urgent)
    assert reward == -1 - 4


def test_neglect_penalty_when_other_zone_critical():
    env = quiet()
    inject(env, [Order(0, 0, 10, [Line(0, 1)]), Order(1, 0, 300, [Line(1, 2)]), Order(2, 0, 300, [Line(1, 3)])])
    _, reward, _, _, _ = env.step(1)
    assert reward == -1 - 10


def test_arrival_cutoff_and_seed_reproducibility():
    env = WarehouseEnv(CONFIG)
    env.reset(seed=3)
    while not env.terminated:
        env.step(env.wait)
    assert max(o.arrival for o in env.orders) < env.shift - env.cutoff
    a, b = WarehouseEnv(CONFIG), WarehouseEnv(CONFIG)
    a.reset(seed=7)
    b.reset(seed=7)
    for _ in range(30):
        sa, ra, da, _, _ = a.step(a.wait)
        sb, rb, db, _, _ = b.step(b.wait)
        assert np.array_equal(sa, sb) and (ra, da, a.time) == (rb, db, b.time)


def test_render_returns_frame():
    env = WarehouseEnv(CONFIG, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(env.wait)
    frame = env.render()
    env.close()
    assert frame.ndim == 3 and frame.shape[2] == 3


@pytest.mark.parametrize("name", list(POLICIES))
def test_baselines_complete_episode(name):
    env = WarehouseEnv({"episodio": {"duracion_turno_min": 60}})
    result = run_policy(env, POLICIES[name], seed=0)
    assert np.isfinite(result["reward"]) and env.terminated


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa", "monte_carlo"])
def test_agents_complete_episode_and_roundtrip(algorithm, tmp_path):
    env = WarehouseEnv({"episodio": {"duracion_turno_min": 60}})
    agent = TabularAgent(env.action_space.n, algorithm, epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.5)
    result = agent.episode(env, seed=1)
    assert np.isfinite(result["reward"]) and env.terminated and agent.epsilon == 0.25
    path = agent.save(tmp_path / "q.npz")
    loaded = TabularAgent.load(path)
    assert loaded.algorithm == algorithm and loaded.episodes_trained == 1
    assert set(loaded.q) == set(agent.q)
    for key in agent.q:
        assert np.allclose(loaded.q[key], agent.q[key])


# ---------------------------------------------------------------------------
# Triplas (s, a, s') del PDF entregado en la Entrega 1 (cuadro 9), esquema "entrega1".
# ---------------------------------------------------------------------------
ENTREGA1 = "configs/env_entrega1.yaml"


def entrega1(**sim):
    import yaml
    with open(ENTREGA1, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["simulacion"] = {"ordenes_por_minuto": 0, **sim}
    config["centro_distribucion"]["zonas"] = list("ABCD")      # las triplas del PDF usan 4 zonas
    return WarehouseEnv(config)


def critical_a_and_consolidable_b():
    """s = (1,2,·,·|0,2,·,·|3,1,·,·|o,·): una orden crítica en A y 5 órdenes con el mismo SKU en B."""
    orders = [Order(0, 0, 10, [Line(0, 1)])]
    orders += [Order(i, 0, 300, [Line(1, 7)]) for i in range(1, 6)]
    orders += [Order(6, 0, 300, [Line(1, 8), Line(1, 9), Line(1, 10)])]
    return orders


def test_entrega1_t1_lote_critico():
    env = entrega1()
    inject(env, critical_a_and_consolidable_b())
    s = env._state()
    assert list(s[[0, 1, 4, 5, 8, 9]]) == [1, 2, 0, 2, 3, 1]
    _, reward, _, _, info = env.step(0)
    assert reward == 1 + 8 and info["tau"] == 0          # P_c(0) + B_u(3), sin avance del reloj


def test_entrega1_t1_alternativa_descuido():
    env = entrega1()
    inject(env, critical_a_and_consolidable_b())
    _, reward, _, _, _ = env.step(1)                      # ir a B con A crítica
    assert reward == 10 + 0 - 6                           # +4, como dice el texto bajo el cuadro 9


def test_entrega1_t3_urgente_sin_urgencia():
    env = entrega1()
    inject(env, [Order(0, 0, 300, [Line(0, 1), Line(0, 2), Line(1, 3)]),
                 Order(1, 0, 300, [Line(0, 1), Line(0, 4), Line(0, 5), Line(0, 6), Line(0, 7)])])
    _, reward, _, _, info = env.step(env.urgent)
    assert reward == -4 and info["tau"] == 0


def test_entrega1_t4_esperar_con_critica():
    env = entrega1()
    inject(env, [Order(0, 0, 1, [Line(2, 1)])])
    _, reward, _, _, info = env.step(env.wait)
    assert reward == -10 - 20 and info["n_venc"] == 1


def test_entrega1_t6_esperar_sin_operarios():
    env = entrega1(operarios=1)
    done = Order(0, 0, 300, [Line(1, 0, assigned=True)])
    backlog = [Order(i, 0, 300, [Line(0, i)]) for i in range(1, 17)]
    inject(env, [done] + backlog)
    env.unassigned = 16
    env.jobs = [(1.0, [(done, done.lines[0])])]
    s = env._state()
    assert s[0] == 3 and s[-2] == 0                       # A saturada, o = 0
    _, reward, _, _, info = env.step(env.wait)
    assert reward == 0 + 3 - 2 and info["n_ok"] == 1


def test_entrega1_t8_cierre_con_pendientes():
    env = entrega1(operarios=1, corte_llegadas_min=0)
    env.shift = 2
    busy = Order(0, 0, 300, [Line(1, 0, assigned=True)])
    critical = Order(1, 0, 10, [Line(0, 1)])
    inject(env, [busy, critical])
    env.unassigned = 1
    env.jobs = [(100.0, [(busy, busy.lines[0])])]
    assert env._state()[8] == 3
    _, reward, done, _, info = env.step(env.wait)
    assert done and reward == -60 and info["n_pend"] == 2


def test_entrega1_t9_accion_invalida():
    env = entrega1()
    inject(env, critical_a_and_consolidable_b())
    state, reward, _, _, _ = env.step(2)                  # LOTE_C sin WIP en C
    assert reward == -50 and np.array_equal(state, env._state())


def test_entrega1_clock_jumps_to_completion():
    env = entrega1(operarios=1)
    inject(env, [Order(0, 0, 300, [Line(0, 1)])])
    _, _, _, _, info = env.step(0)
    assert env.time == pytest.approx(5.2) and info["n_ok"] == 1    # sin tope de Δt


def test_final_t3_urgente_sin_urgencia():
    env = quiet(operarios=1)
    inject(env, [Order(0, 0, 300, [Line(0, 1), Line(0, 2), Line(1, 3)]),
                 Order(1, 0, 400, [Line(0, i) for i in range(4, 10)])])
    _, reward, _, _, info = env.step(env.urgent)
    # -1 viaje, -4 sin urgencia; quedan 6 líneas en A durante Δt = 2 min: -0,6 de WIP.
    assert reward == pytest.approx(-5.6) and info["tau"] == 1


def test_evaluation_does_not_grow_qtable():
    env = WarehouseEnv({"episodio": {"duracion_turno_min": 60}})
    agent = TabularAgent(env.action_space.n)
    agent.episode(env, seed=1, train=False)
    assert len(agent.q) == 0
    agent.episode(env, seed=1)
    trained = len(agent.q)
    agent.episode(env, seed=2, train=False)
    assert len(agent.q) == trained > 0
