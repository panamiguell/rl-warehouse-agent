# Resultados: esquema entrega1, 2 zonas, 6000 episodios de entrenamiento, 100 turnos de evaluación

Promedios por turno de 8 h; ± es el intervalo de confianza del 95 % de la media.

| Política | Tipo | Recompensa | A tiempo | Vencidas | Pendientes al cierre | Visitas/línea | Viajes | WIP medio (líneas) |
|---|---|---|---|---|---|---|---|---|
| lote_mas_urgente | regla | 873 ± 205 | 319.3 | 33.3 ± 7.2 | 5.6 | 0.590 | 88.1 | 58.6 |
| umbral_consolidacion | regla | 693 ± 204 | 315.7 | 35.7 ± 7.5 | 6.5 | 0.584 | 74.3 | 60.8 |
| lote_mas_consolidado | regla | -72 ± 346 | 282.1 | 70.7 ± 13.0 | 6.5 | 0.591 | 88.2 | 58.6 |
| sarsa | rl | -858 ± 319 | 274.4 | 72.3 ± 10.2 | 19.8 | 0.576 | 74.6 | 80.1 |
| q_learning | rl | -1296 ± 413 | 266.0 | 78.6 ± 11.7 | 27.2 | 0.583 | 77.9 | 87.7 |
| monte_carlo | rl | -3792 ± 566 | 183.0 | 160.7 ± 14.9 | 58.5 | 0.607 | 105.2 | 124.0 |
| aleatoria | regla | -4168 ± 413 | 165.1 | 175.6 ± 13.0 | 50.5 | 0.605 | 101.0 | 124.5 |
| urgente_siempre | regla | -10681 ± 202 | 51.6 | 287.3 ± 4.3 | 184.6 | 0.925 | 175.2 | 362.8 |
| esperar_siempre | regla | -20663 ± 196 | 0.0 | 338.9 ± 3.9 | 356.9 | 0.000 | 0.0 | 660.6 |
