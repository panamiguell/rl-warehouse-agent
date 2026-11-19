# Resultados en el cuarto 8x8 (2000 episodios de prueba)

± es el intervalo de confianza del 95 %.

| Política | Llega a la meta | Choca | Límite de pasos | Pasos (si llega) | Recompensa |
|---|---|---|---|---|---|
| Q-learning | 65.2 % ± 2.1 | 34.8 % | 0.0 % | 23.5 | 10.8 ± 4.0 |
| SARSA | 62.9 % ± 2.1 | 37.1 % | 0.0 % | 21.7 | 7.7 ± 4.1 |
| Regla cautelosa (con sensores) | 44.5 % ± 2.2 | 55.5 % | 0.0 % | 22.3 | -28.0 ± 4.2 |
| Regla: por el borde | 35.1 % ± 2.1 | 64.9 % | 0.0 % | 16.0 | -39.3 ± 4.0 |
| Regla: escalera por el centro | 15.9 % ± 1.6 | 84.1 % | 0.0 % | 28.0 | -80.7 ± 2.9 |
| Aleatoria (sin entrenar) | 0.4 % ± 0.3 | 99.6 % | 0.0 % | 71.4 | -130.2 ± 1.2 |
