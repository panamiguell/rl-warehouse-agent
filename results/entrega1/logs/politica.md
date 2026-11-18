# Política aprendida por situación (30 turnos de prueba, % de las decisiones)

### Agente q_learning (q_learning)

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 1861 | 48 % | 3 % | 0 % | 0 % | 7 % | 43 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 36 | 0 % | 0 % | 58 % | 11 % | 8 % | 22 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 917 | 0 % | 0 % | 71 % | 8 % | 7 % | 14 % |
| S4c: Sin urgencia, resto de casos | 1099 | 0 % | 0 % | 18 % | 8 % | 12 % | 61 % |

### Agente sarsa (sarsa)

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 1564 | 51 % | 14 % | 0 % | 0 % | 4 % | 31 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 58 | 0 % | 0 % | 14 % | 7 % | 5 % | 74 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 1029 | 0 % | 0 % | 67 % | 4 % | 2 % | 27 % |
| S4c: Sin urgencia, resto de casos | 1427 | 0 % | 0 % | 13 % | 4 % | 11 % | 72 % |

### Regla fija lote_mas_urgente

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 676 | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 58 | 0 % | 0 % | 98 % | 2 % | 0 % | 0 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 892 | 0 % | 0 % | 100 % | 0 % | 0 % | 0 % |
| S4c: Sin urgencia, resto de casos | 1000 | 0 % | 0 % | 40 % | 60 % | 0 % | 0 % |

