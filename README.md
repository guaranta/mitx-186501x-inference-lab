# mitx-186501x-inference-lab

**MITx 18.6501x — Fundamentals of Statistics**

Laboratório de estatística inferencial: intervalos de confiança, delta method, testes de hipótese e **monitoramento de drift** em scores de modelos de produção.

---

## Resultados — model drift (synthetic)

| Métrica | Valor | Threshold |
|---------|-------|-----------|
| KS statistic | **0.178** | — |
| KS p-value | **< 0.001** | α = 0.05 → drift detectado |
| PSI | **0.191** | alerta se > 0.2 |
| Δ mean score | 0.702 → 0.659 | −6.2% shift |

![Distribuições e PSI por decil](docs/figures/drift_detection.png)

---

## Módulos

| Módulo | Técnica | Comando |
|--------|---------|---------|
| `notebooks/` | 10 notebooks (MLE, CIs, hypothesis tests) | Jupyter |
| `homeworks/` | Delta method, Bernoulli, two-sample | `python homeworks/hw2_delta_method.py` |
| `model-drift/` | KS test + PSI | `python model-drift/run.py` |

## Fundamentos

**Delta method:** `Var(g(X̄)) ≈ [g'(μ)]² · Var(X̄)/n`

**KS test:** `D = sup_x |F_ref(x) − F_cur(x)|`

**PSI:** `Σ (p_cur − p_ref) · ln(p_cur/p_ref)` — Population Stability Index

## Setup

```bash
pip install -r requirements.txt
python docs/generate_figures.py
```

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)
