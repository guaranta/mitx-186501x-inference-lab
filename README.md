# mitx-186501x-inference-lab

**MITx 18.6501x — Fundamentals of Statistics**

Laboratório de estatística inferencial: MLE, intervalos de confiança, testes de hipótese, delta method e detecção de drift em modelos de produção.

| Módulo | Conteúdo | Comando |
|--------|----------|---------|
| `notebooks/` | 10 notebooks (CIs, delta method, hypothesis testing, MLE) | Jupyter |
| `scripts/` | Versões Python dos notebooks iniciais | `python scripts/...` |
| `homeworks/` | Delta method, testes Bernoulli e two-sample | `python homeworks/hw2_delta_method.py` |
| `model-drift/` | KS test + PSI para scores de modelo | `python model-drift/run.py` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model-drift/run.py
```

## Origem acadêmica

Notebooks baseados em [mitx-stats-notes](https://github.com/guaranta/mitx-stats-notes) (comunidade 18.6501x). Homeworks e model-drift são implementações originais inspiradas nos problem sets do curso.

## Portfólio

- [Portfolio AI Engineer / CTO](https://portfolio-ai-cto-guaranta.netlify.app)
- [Fórmulas ATE, DiD, power](docs/portfolio-link.md)

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)
