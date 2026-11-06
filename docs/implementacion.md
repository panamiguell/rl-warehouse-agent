# Implementación del documento de Entrega 1

El entorno conserva las acciones, el orden del vector de observación, los umbrales y las recompensas del LaTeX. Las tablas Q son dispersas: solo se almacenan estados visitados. Se implementan Q-learning, SARSA y Monte Carlo de primera visita con política epsilon-greedy y máscara de acciones.

## Supuestos que el documento no fija

Las llegadas siguen un proceso de Poisson homogéneo, con 0.5 órdenes/minuto por defecto. Cada orden tiene 1-6 líneas, zona y SKU uniformes (10 SKU por zona), una unidad por línea y deadline entre 30 y 120 minutos desde su llegada. El tiempo de servicio es 3 minutos base más 0.5 por ubicación única y 0.1 por línea. Incluye de forma agregada recogida y separación; no modela geometría ni cantidades de varias unidades. Todos estos valores se configuran en `simulacion`.

La primera versión usa tres operarios durante todo el turno. No implementa pausas ni ausencias. Una orden urgente recoge todas sus líneas aún no asignadas; las líneas ya en ejecución terminan con su operario original. Una orden solo se completa cuando terminan todas sus líneas. La holgura se estima con el tiempo de servicio de las líneas aún no terminadas de la orden.

Las órdenes vencidas permanecen disponibles para picking y solo reciben una penalización por vencimiento. Completar exactamente al deadline cuenta como a tiempo. Al cierre se cuentan también las órdenes en ejecución. El estado terminal se codifica como vector cero con `terminated=True` y máscara vacía.

Se siguen las reglas formales de transición: asignar conserva el reloj únicamente si quedan operarios libres y líneas sin asignar; en otro caso avanza hasta la próxima finalización o cierre. ESPERAR avanza como máximo dos minutos. Las llegadas intermedias se procesan sin interrumpir ese avance. Algunas triplas ilustrativas del LaTeX (por ejemplo T3) no concuerdan con esa regla.

La observación agregada omite identidades, trabajos en ejecución y tiempo restante: puede haber aliasing de estados y no se garantiza que sea estrictamente Markov. Los resultados son una base experimental, no evidencia de que una política entrenada ya mejore la operación real.

## Ejecución

Desde la raíz, con Python 3.10 o superior:

```powershell
conda env create -f environment.yml
conda activate rl-warehouse-agent
python -m pytest
python experiments/train.py --episodes 100 --eval-episodes 10
```

La comparación usa las mismas semillas de entrenamiento y evaluación para los tres métodos, sin exploración epsilon en evaluación. Exporta métricas por episodio a `results/logs/comparison.csv` y tablas Q (estados y valores) a `results/models/*.npz`. Las tablas exportadas son políticas para inspección o evaluación; no incluyen contadores Monte Carlo para reanudar aprendizaje. Se reportan recompensa, órdenes llegadas/completadas/a tiempo/vencidas/pendientes, ubicaciones visitadas y líneas asignadas a picking.

Puede reducirse `centro_distribucion.zonas` a `[A, B]` para el experimento tabular de dos zonas. Los archivos de configuraci?n se cargan mediante una ruta explícita, sin depender de una instalación junto al código fuente.
