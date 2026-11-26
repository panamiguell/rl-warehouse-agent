# Agente de RL en un cuarto 8x8 con obstáculos móviles

Repositorio del proyecto del curso **Aprendizaje por Refuerzo** (Maestría en Inteligencia Artificial, Universidad de los Andes).

## Proyecto del curso: cuarto 8x8 con obstáculos móviles

Un agente (triángulo) parte de la casilla (0,0) mirando al norte y debe llegar a la meta (7,7) sin chocar con cinco obstáculos que se mueven al azar en cada paso, al mismo tiempo que él. Puede girar 90° a la izquierda o a la derecha, o avanzar a la casilla de enfrente si existe. Siguiendo las aclaraciones del curso, el agente **no ve los obstáculos** (su Q-tabla depende solo de posición y orientación), los obstáculos no se superponen ni entran en la salida ni en la meta, y no hay límite de pasos.

| Entregable | Archivo |
| --- | --- |
| Definición de estados, acciones y recompensas | [`docs/proyecto_8x8/definicion_ambiente.pdf`](docs/proyecto_8x8/definicion_ambiente.pdf) |
| Reporte final | [`docs/proyecto_8x8/reporte_final.pdf`](docs/proyecto_8x8/reporte_final.pdf) |
| Video del agente | [`results/cuarto_8x8/figures/demo.mp4`](results/cuarto_8x8/figures/demo.mp4) |
| Q-tabla | [`results/cuarto_8x8/models/q_learning.npz`](results/cuarto_8x8/models/q_learning.npz) · [CSV legible](results/cuarto_8x8/models/q_learning_qtable.csv) |
| Registros de las pruebas | [`results/cuarto_8x8/logs/`](results/cuarto_8x8/logs) |
| Código | [`src/rl_gridworld/`](src/rl_gridworld) · [`experiments/cuarto_8x8/`](experiments/cuarto_8x8) · [`tests/test_gridworld.py`](tests/test_gridworld.py) |

```powershell
conda env create -f environment.yml
conda activate rl-cuarto-8x8
python -m pytest tests/test_gridworld.py                # 18 pruebas, incluidas las triplas (s, a, s') del documento
python experiments/cuarto_8x8/grid_search.py            # búsqueda de hiperparámetros (unos 30 min con 2 núcleos)
python experiments/cuarto_8x8/train.py                  # entrena Q-learning y SARSA y los compara con reglas fijas
python experiments/cuarto_8x8/demo.py                   # carga la Q-tabla y graba el video
python experiments/cuarto_8x8/demo.py --human           # lo muestra en una ventana
```

Los hiperparámetros de `configs/cuarto_8x8.yaml` (γ = 0,999, tasa por visitas con ω = 0,9, ε mínimo 0,01) salen de la búsqueda en grid de `grid_search.py`: 54 combinaciones más 10 corridas de confirmación, elegidas solo con episodios de validación (resultados en `results/cuarto_8x8/grid/` y en la sección 3 del reporte).

La corrida completa de `train.py` tarda poco más de un minuto en un procesador de 2 núcleos sin GPU y es determinista: vuelve a dar la misma Q-tabla y los mismos resultados. El detalle de tiempos, equipo y software está en `results/cuarto_8x8/logs/summary.json` y en la sección 6 del reporte.

La Q-tabla se guarda con `TabularAgent.save(ruta)` y se carga con `TabularAgent.load(ruta)` (`src/rl_gridworld/agent.py`).

**Resultados** (2 000 episodios de prueba; ± es el intervalo de confianza del 95 %):

| Política | Llega a la meta | Choca | Pasos si llega |
|---|---|---|---|
| **Q-learning** | 34,4 % ± 2,1 | 65,6 % | 17,0 |
| SARSA | 35,1 % ± 2,1 | 64,9 % | 16,0 |
| Regla: por el borde | 35,1 % ± 2,1 | 64,9 % | 16,0 |
| Regla: escalera por el centro | 15,9 % ± 1,6 | 84,1 % | 28,0 |
| Aleatoria (sin entrenar) | 0,4 % ± 0,3 | 99,6 % | 71,4 |

El agente aprendió por su cuenta una ruta por el borde: Q-learning baja por la columna 0 y va a la derecha por la fila 7; SARSA, la ruta espejo (fila 0 y columna 7), que es la misma de la regla por el borde. Las dos rutas tienen la misma probabilidad de llegar: sin ver los obstáculos, ir por el borde con pocos giros es lo de menos riesgo.

![Ruta aprendida](docs/proyecto_8x8/figuras/ruta_aprendida.png)
