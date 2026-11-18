# Implementación

El simulador (`src/rl_warehouse/env/warehouse.py`) implementa el MDP de la Entrega 1 (`docs/entrega1`, el PDF tal como se entregó). Los estados, la discretización y las acciones son exactamente los del documento. La función de recompensa y el avance del reloj existen en dos esquemas, seleccionables desde la configuración:

| Esquema | Configuración | Qué es |
| --- | --- | --- |
| `entrega1` | `configs/env_entrega1.yaml` | La especificación entregada, sin cambios: pago por consolidación y bono por urgencia en cada lote, −2 por zona saturada, reloj que salta hasta que un operario quede libre, γ = 0,99 por decisión. Las pruebas `test_entrega1_*` reproducen las triplas del cuadro 9 del PDF. |
| `final` | `configs/env_default.yaml` | La especificación ajustada tras los experimentos: +2 por visita ahorrada y −1 por viaje, costo de WIP, −1 por zona crítica, descuido −10, reloj con avance máximo de Δt y descuento por tiempo. Es la que usa el agente entregado. |

La justificación de cada cambio, con la evidencia experimental, está en el reporte final (`docs/entrega_final`). Se implementan Q-learning, SARSA y Monte Carlo de primera visita (`src/rl_warehouse/agents/tabular.py`) con política ε-greedy, decaimiento de ε y máscara de acciones, además de seis reglas fijas de referencia (`src/rl_warehouse/agents/baselines.py`).

## Supuestos del simulador (el documento no los fija)

- **Zonas.** Los experimentos usan la versión tabular de 2 zonas que propone la sección 2 de la Entrega 1 (9 409 estados). La de 4 zonas tiene 7,4 millones de estados (`--zones ABCD`).
- **Demanda.** Proceso de Poisson no homogéneo con dos picos (`simulacion.perfil_demanda`: 0,3 órdenes/min en valle y 1,2 en pico). No llegan órdenes en los últimos 30 min (`corte_llegadas_min`), para que la penalización de cierre mida órdenes que se pudieron despachar.
- **Órdenes.** Entre 1 y 6 líneas, zona uniforme, una unidad por línea y deadline entre 30 y 120 min desde la llegada. Cada zona tiene 30 SKU con popularidad Zipf (`zipf_s: 1.0`), como una curva ABC, y por eso hay SKU repetidos entre órdenes.
- **Tiempo de servicio.** 4 min de desplazamiento, 1 min por ubicación visitada y 0,2 min por línea. Consolidar ahorra tiempo real, no solo recompensa.
- **Operarios.** Tres durante todo el turno.
- **Órdenes urgentes.** URGENTE recoge las líneas aún no asignadas de la orden con menor holgura; las líneas ya en ejecución terminan con su operario. La holgura se estima con el tiempo de servicio de las líneas no terminadas.
- **Vencimientos y cierre.** Las órdenes vencidas siguen disponibles para picking y se penalizan una sola vez. Terminar exactamente en el deadline cuenta como a tiempo. Al cierre, n_pend cuenta las órdenes no completadas, incluidas las que están en ejecución.
- **Estado terminal.** Vector de ceros con `terminated=True` y máscara vacía.

## Protocolo experimental

`experiments/train.py` usa tres conjuntos de semillas disjuntos: entrenamiento (`seed·100 000 + i`), validación (500 000 + i: curva de aprendizaje y selección del mejor punto de control) y prueba (1 000 000 + i: evaluación final de agentes y reglas fijas, 100 turnos). Las evaluaciones son greedy, sin exploración. La Q-tabla guardada es la del punto de control con mejor recompensa en validación (`--last` guarda la del último episodio).

## Limitaciones

La observación agregada omite la identidad de las órdenes, los trabajos en ejecución, la hora del turno y el tiempo restante de cada orden. Hay aliasing de estados: situaciones distintas comparten el mismo estado, así que el problema no es estrictamente de Markov y los métodos tabulares oscilan. Los resultados son una base experimental sobre un simulador, no evidencia de mejora en una operación real.

## Ejecución

Desde la raíz, con Python 3.10 o superior:

```powershell
conda env create -f environment.yml
conda activate rl-warehouse-agent
python -m pytest
python experiments/train.py                                                        # especificación final
python experiments/train.py --config configs/env_entrega1.yaml --out results/entrega1   # especificación entregada
python experiments/compare_specs.py
python experiments/analyze_policy.py --model results/models/q_learning.npz --compare lote_mas_urgente --out results/logs/politica.md
python experiments/demo.py --model results/models/q_learning.npz
```

| Archivo | Contenido |
| --- | --- |
| `results/models/<algoritmo>.npz` | Q-tabla e hiperparámetros; `TabularAgent.load` la recupera (incluye los contadores de Monte Carlo para seguir entrenando) |
| `results/models/<algoritmo>_qtable.csv` | Q-tabla legible: el estado desglosado y el valor Q de cada acción (0 en acciones nunca probadas en ese estado) |
| `results/logs/training_<algoritmo>.csv` | Recompensa y métricas de cada episodio de entrenamiento |
| `results/logs/learning_curve.csv` | Evaluación greedy periódica en validación |
| `results/logs/comparison.csv`, `summary.md`, `summary.json` | Evaluación final en prueba de agentes y reglas fijas |
| `results/logs/comparacion_especificaciones.md`, `politica.md` | Especificación entregada frente a final; política aprendida por situación |
| `results/entrega1/` | Lo mismo, con la especificación entregada |
| `results/figures/` | Curvas, comparaciones y el video del agente (`demo.mp4`) |
