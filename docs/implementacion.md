# Implementación del documento de Entrega 1

El entorno (`src/rl_warehouse/env/warehouse.py`) conserva las acciones, el orden del vector de observación, los umbrales y las recompensas del LaTeX (`docs/entrega1`). Los valores numéricos están en `configs/env_default.yaml`; si se cambia uno, hay que cambiar el otro. Las tablas Q son dispersas: solo se almacenan los estados visitados. Se implementan Q-learning, SARSA y Monte Carlo de primera visita con política epsilon-greedy, decaimiento de epsilon y máscara de acciones, además de seis reglas fijas de referencia (`src/rl_warehouse/agents/baselines.py`).

## Supuestos que el documento no fija

- **Demanda.** Proceso de Poisson no homogéneo con dos picos en el turno (`simulacion.perfil_demanda`: 0,3 órdenes/min en valle y 1,2 en pico). No llegan órdenes en los últimos 30 min (`corte_llegadas_min`), para que la penalización de cierre mida órdenes que se pudieron despachar.
- **Órdenes.** Entre 1 y 6 líneas, zona uniforme, una unidad por línea y deadline entre 30 y 120 min desde la llegada. Cada zona tiene 30 SKU con popularidad Zipf (`zipf_s: 1.0`): pocos SKU concentran la demanda, como en una curva ABC, y por eso hay SKU repetidos entre órdenes.
- **Tiempo de servicio.** 4 min de desplazamiento, 1 min por ubicación visitada y 0,2 min por línea (tomar y separar en la put-wall). Consolidar ahorra tiempo de verdad, no solo recompensa. No se modela la geometría de los pasillos.
- **Operarios.** Tres durante todo el turno; no hay pausas ni ausencias.
- **Órdenes urgentes.** URGENTE recoge todas las líneas aún no asignadas de la orden con menor holgura; las líneas ya en ejecución terminan con su operario original. Una orden se completa cuando terminan todas sus líneas. La holgura se estima con el tiempo de servicio de las líneas aún no terminadas.
- **Vencimientos.** Las órdenes vencidas siguen disponibles para picking y solo se penalizan una vez. Completar exactamente en el deadline cuenta como a tiempo. Al cierre se cuentan también las órdenes en ejecución.
- **Estado terminal.** Se codifica como vector de ceros con `terminated=True` y máscara vacía.

## Por qué el reloj avanza como máximo Δt y el descuento es por tiempo

La primera versión dejaba que el reloj saltara hasta la siguiente finalización cuando un lote ocupaba al último operario libre (a veces más de 20 min). Con eso, todos los costos de ese intervalo (WIP, vencimientos) llegaban juntos y sin descontar en la recompensa del lote, mientras que, al esperar, los mismos costos llegaban repartidos en pasos de 2 min y descontados. Los agentes aprendían que esperar "costaba menos" y llegaban a tener más vencimientos que una política aleatoria. Por eso:

1. El reloj avanza a lo sumo `delta_t_min` = 2 min por decisión. Con todos los operarios ocupados la única acción es ESPERAR.
2. El descuento es por minuto transcurrido: `gamma ** (minutos / delta_t)`, con `gamma = 0.95` por cada 2 min. Las decisiones inmediatas (asignar varios operarios en el mismo instante) no se descuentan.
3. Esperar nunca es gratis: hay un costo de WIP de −0,05 por línea pendiente sin asignar por minuto. Sin ese costo, ESPERAR en un estado que se repite parecía tener un valor casi nulo.

Con `gamma = 0.99` el aprendizaje diverge: el estado agregado no distingue cuánto han envejecido las órdenes dentro de cada rango de holgura (aliasing), y un horizonte largo amplifica ese error. Con 0.95 (unos 40 min de horizonte efectivo) es estable.

## Limitaciones

La observación agregada omite identidades, trabajos en ejecución, la hora del turno y el tiempo restante de cada orden, así que hay aliasing de estados y el problema no es estrictamente de Markov. Los resultados son una base experimental, no evidencia de que una política entrenada mejore la operación real. La versión de 4 zonas tiene más de 7 millones de estados posibles; por eso los experimentos principales usan 2 zonas (`--zones AB`, 9 409 estados).

## Ejecución

Desde la raíz, con Python 3.10 o superior:

```powershell
conda env create -f environment.yml
conda activate rl-warehouse-agent
python -m pytest
python experiments/train.py --zones AB                      # 2 400 episodios por algoritmo
python experiments/demo.py --zones AB --model results/models/q_learning.npz
```

`train.py` usa las mismas semillas de evaluación para los tres algoritmos y las reglas fijas, sin exploración en la evaluación. Escribe en `results/`:

| Archivo | Contenido |
| --- | --- |
| `models/<algoritmo>.npz` | Q-tabla e hiperparámetros; se carga con `TabularAgent.load` (también guarda los contadores de Monte Carlo, así que se puede seguir entrenando) |
| `models/<algoritmo>_qtable.csv` | Q-tabla legible: cada estado desglosado y el valor Q de cada acción (0 en acciones que nunca se probaron en ese estado) |
| `logs/training_<algoritmo>.csv` | Recompensa y métricas de cada episodio de entrenamiento |
| `logs/learning_curve.csv` | Evaluación greedy periódica |
| `logs/comparison.csv`, `logs/summary.md` | Evaluación final de agentes y reglas fijas |
| `figures/curvas_aprendizaje.png`, `figures/comparacion.png` | Gráficas para el reporte |

`demo.py` carga una Q-tabla, juega un turno con la política greedy y lo graba en `results/figures/demo.mp4` (o `.gif`). Con `--policy lote_mas_urgente` graba una regla fija para comparar, y con `--human` lo muestra en una ventana. Los archivos de configuración se cargan mediante una ruta explícita, sin depender de una instalación junto al código fuente.
