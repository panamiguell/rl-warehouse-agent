# rl-warehouse-agent

Repositorio del proyecto del curso **Aprendizaje por Refuerzo** (Maestría en Inteligencia Artificial, Universidad de los Andes). Contiene la definición del ambiente, los agentes y los experimentos para entrenar y comparar algoritmos de RL en un problema de toma de decisiones secuenciales.

**Problema:** un agente decide qué trabajo de picking liberar en un centro de distribución (un lote consolidado por zona de racks, una orden urgente o esperar) para recoger en una sola visita los SKU que piden varias órdenes, sin incumplir deadlines y considerando operarios libres y WIP.

## Estructura

```
rl-warehouse-agent/
├── docs/                  Documentación y entregas del curso
│   ├── entrega1/          Entrega 1: estados, acciones y recompensas (LaTeX + PDF)
│   └── referencias/       Artículos y material de apoyo
├── data/
│   ├── raw/               Datos originales (órdenes, ubicaciones de SKU) - no versionados
│   ├── processed/         Datos limpios listos para el simulador - no versionados
│   └── simulated/         Escenarios de órdenes generados - no versionados
├── src/rl_warehouse/      Código fuente
│   ├── env/               Simulador del centro de distribución (MDP)
│   ├── agents/            Agentes: Monte Carlo, SARSA, Q-learning, ...
│   └── utils/             Semillas, logging, métricas
├── configs/               Parámetros del ambiente y de entrenamiento (YAML)
├── experiments/           Scripts de entrenamiento y comparación de agentes
├── notebooks/             Exploración y análisis de resultados
├── results/
│   ├── figures/           Gráficas para los informes
│   ├── models/            Tablas Q / modelos entrenados - no versionados
│   └── logs/              Registros de entrenamiento - no versionados
├── tests/                 Pruebas unitarias
└── requirements.txt
```

## Entregas

| Entrega | Contenido | Archivo |
| --- | --- | --- |
| 1 | Definición de estados, acciones y función de recompensa R(s, a, s') | [`docs/entrega1/entrega1_definicion_mdp.pdf`](docs/entrega1/entrega1_definicion_mdp.pdf) |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Compilar los documentos LaTeX

```bash
cd docs/entrega1
latexmk -pdf entrega1_definicion_mdp.tex     # o: pdflatex entrega1_definicion_mdp.tex (dos veces)
latexmk -c                                   # limpia archivos auxiliares
```

También se puede subir la carpeta `docs/entrega1` a Overleaf y compilar con pdfLaTeX.

Los valores numéricos del documento (discretización y recompensas) están replicados en `configs/env_default.yaml`; si se cambia uno, hay que cambiar el otro.
