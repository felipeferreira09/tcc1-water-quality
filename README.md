# tcc1-water-quality
# Classificação da Qualidade da Água

TCC 1 — Ciência da Computação — PUCRS  
Autor: Felipe Rosa Ferreira

---

## Descrição

Implementação e avaliação de modelos de Machine Learning para classificar a potabilidade da água a partir de dados físico-químicos ambientais.

Modelos utilizados: Naive Bayes e Regressão Logística  
Técnica de balanceamento: SMOTE

---

## Dataset

- **Nome:** Water Potability (Kaggle)
- **Amostras:** 3.276
- **Features:** ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
- **Alvo:** Potability (0 = Não Potável, 1 = Potável)

---

## Como executar

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fn1edB3YYbYOh3FFBG1CGyDjxIgJyfM3?usp=sharing)


## Resultados

| Modelo | Acurácia | F1-Score | AUC-ROC |
|---|---|---|---|
| Naive Bayes | 0.5808 | 0.4802 | 0.5949 |
| Regressão Logística | 0.5213 | 0.4491 | 0.5279 |

AUC-ROC em torno de 0.59 indica que modelos lineares são insuficientes para este problema, justificando o uso de MLP no TCC 2.

---

## Estrutura

```
tcc1-water-quality/
├── data/               # Dataset
├── notebooks/          # Notebook Colab
├── src/                # Scripts Python
│   ├── modelo.py
│   └── modelo_smote.py
└── results/            # Gráficos e resultados
```
