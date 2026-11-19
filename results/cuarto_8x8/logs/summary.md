# Resultados en el cuarto 8x8 (2000 episodios de prueba)

± es el intervalo de confianza del 95 %.

| Política | Llega a la meta | Choca | Límite de pasos | Pasos (si llega) | Recompensa |
|---|---|---|---|---|---|
| SARSA | 59.4 % ± 2.2 | 40.6 % | 0.0 % | 23.0 | 0.8 ± 4.1 |
| Q-learning | 58.6 % ± 2.2 | 41.4 % | 0.0 % | 23.3 | -0.9 ± 4.1 |
| Regla cautelosa | 53.1 % ± 2.2 | 46.9 % | 0.0 % | 19.9 | -8.9 ± 4.2 |
| Regla directa | 33.1 % ± 2.1 | 66.8 % | 0.0 % | 16.0 | -42.7 ± 3.9 |
| Aleatoria (sin entrenar) | 0.4 % ± 0.3 | 99.7 % | 0.0 % | 65.3 | -125.4 ± 1.2 |
