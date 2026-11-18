# Resultados: esquema final, 2 zonas, 6000 episodios de entrenamiento, 100 turnos de evaluación

Promedios por turno de 8 h; ± es el intervalo de confianza del 95 % de la media.

| Política | Tipo | Recompensa | A tiempo | Vencidas | Pendientes al cierre | Visitas/línea | Viajes | WIP medio (líneas) |
|---|---|---|---|---|---|---|---|---|
| lote_mas_urgente | regla | -426 ± 307 | 318.5 | 33.8 ± 7.2 | 5.6 | 0.592 | 90.7 | 58.1 |
| umbral_consolidacion | regla | -561 ± 308 | 317.2 | 34.8 ± 7.4 | 5.9 | 0.585 | 75.1 | 59.8 |
| q_learning | rl | -599 ± 339 | 318.2 | 33.9 ± 7.9 | 6.7 | 0.584 | 80.6 | 62.3 |
| sarsa | rl | -732 ± 331 | 317.2 | 33.5 ± 7.8 | 7.9 | 0.583 | 77.1 | 64.5 |
| lote_mas_consolidado | regla | -1390 ± 450 | 283.1 | 69.6 ± 12.6 | 6.2 | 0.592 | 90.7 | 57.9 |
| monte_carlo | rl | -2433 ± 688 | 277.4 | 70.7 ± 14.6 | 18.3 | 0.590 | 87.5 | 76.8 |
| aleatoria | regla | -7322 ± 586 | 164.5 | 176.1 ± 12.6 | 50.5 | 0.607 | 102.6 | 124.8 |
| urgente_siempre | regla | -20366 ± 399 | 52.4 | 286.5 ± 4.3 | 184.1 | 0.925 | 175.8 | 360.7 |
| esperar_siempre | regla | -36071 ± 405 | 0.0 | 338.9 ± 3.9 | 356.9 | 0.000 | 0.0 | 660.6 |
