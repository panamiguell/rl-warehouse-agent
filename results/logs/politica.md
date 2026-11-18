# Política aprendida por situación (30 turnos de prueba, % de las decisiones)

### Agente q_learning (q_learning)

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 746 | 90 % | 2 % | 0 % | 0 % | 2 % | 6 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 51 | 0 % | 0 % | 84 % | 4 % | 4 % | 8 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 970 | 0 % | 0 % | 95 % | 1 % | 0 % | 4 % |
| S4c: Sin urgencia, resto de casos | 1393 | 0 % | 0 % | 25 % | 17 % | 10 % | 48 % |

### Agente sarsa (sarsa)

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 776 | 89 % | 2 % | 0 % | 0 % | 2 % | 6 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 49 | 0 % | 0 % | 57 % | 4 % | 2 % | 37 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 1004 | 0 % | 0 % | 87 % | 7 % | 1 % | 5 % |
| S4c: Sin urgencia, resto de casos | 1646 | 0 % | 0 % | 24 % | 4 % | 7 % | 65 % |

### Regla fija lote_mas_urgente

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 658 | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 70 | 0 % | 0 % | 93 % | 7 % | 0 % | 0 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 906 | 0 % | 0 % | 100 % | 0 % | 0 % | 0 % |
| S4c: Sin urgencia, resto de casos | 1064 | 0 % | 0 % | 36 % | 64 % | 0 % | 0 % |

### Regla fija umbral_consolidacion

| Situación | Decisiones | Lote zona crítica | Lote otra zona (descuido) | Lote zona más consolidable | Otro lote | Urgente | Esperar |
|---|---|---|---|---|---|---|---|
| S3: Hay una orden crítica (u_max = 3) | 706 | 100 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| S4a: Sin urgencia, alta demanda (λ = 2) y consolidación baja (c ≤ 1): esperar es razonable (D4) | 57 | 0 % | 0 % | 93 % | 2 % | 0 % | 5 % |
| S4b: Sin urgencia, alguna zona con consolidación alta (c = 2) | 909 | 0 % | 0 % | 99 % | 1 % | 0 % | 0 % |
| S4c: Sin urgencia, resto de casos | 1639 | 0 % | 0 % | 28 % | 6 % | 0 % | 66 % |

