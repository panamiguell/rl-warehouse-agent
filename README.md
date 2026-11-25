# rl-warehouse-agent

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
conda activate rl-warehouse-agent
python -m pytest tests/test_gridworld.py                # 18 pruebas, incluidas las triplas (s, a, s') del documento
python experiments/cuarto_8x8/train.py                  # entrena Q-learning y SARSA y los compara con reglas fijas
python experiments/cuarto_8x8/demo.py                   # carga la Q-tabla y graba el video
python experiments/cuarto_8x8/demo.py --human           # lo muestra en una ventana
```

La corrida completa de `train.py` tarda unos 2 minutos en un procesador de 2 núcleos sin GPU (cada agente se entrena en menos de un minuto) y es determinista: vuelve a dar la misma Q-tabla y los mismos resultados. El detalle de tiempos, equipo y software está en `results/cuarto_8x8/logs/summary.json` y en la sección 5 del reporte.

La Q-tabla se guarda con `TabularAgent.save(ruta)` y se carga con `TabularAgent.load(ruta)` (`src/rl_gridworld/agent.py`).

**Resultados** (2 000 episodios de prueba; ± es el intervalo de confianza del 95 %):

| Política | Llega a la meta | Choca | Pasos si llega |
|---|---|---|---|
| **Q-learning** | 35,1 % ± 2,1 | 64,9 % | 16,0 |
| SARSA | 34,4 % ± 2,1 | 65,6 % | 17,0 |
| Regla: por el borde | 35,1 % ± 2,1 | 64,9 % | 16,0 |
| Regla: escalera por el centro | 15,9 % ± 1,6 | 84,1 % | 28,0 |
| Aleatoria (sin entrenar) | 0,4 % ± 0,3 | 99,6 % | 71,4 |

El agente aprendió por su cuenta la ruta por el borde (derecha por la fila 0 y abajo por la columna 7, con dos giros), que es la de menos riesgo cuando no se ven los obstáculos.

![Ruta aprendida](docs/proyecto_8x8/figuras/ruta_aprendida.png)
