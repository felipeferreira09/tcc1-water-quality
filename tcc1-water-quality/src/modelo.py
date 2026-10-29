import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)

# ── 1. Carregar e pré-processar ──────────────────────────────────────────────
df = pd.read_excel('/mnt/user-data/attachments/7ffbe7ad-83a7-57ec-bb4c-c495e259392c/575afc45-datasetcorretoPortugues.xlsx')

# Preencher valores ausentes com a média de cada coluna
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop('Potability', axis=1)
y = df['Potability']

print(f"Dataset: {df.shape[0]} amostras | {X.shape[1]} features")
print(f"Classe 0 (Não Potável): {(y==0).sum()} | Classe 1 (Potável): {(y==1).sum()}")
print()

# ── 2. Divisão treino/teste 80/20 ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização (essencial para Regressão Logística)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. Naive Bayes ───────────────────────────────────────────────────────────
nb = GaussianNB()
nb.fit(X_train_sc, y_train)
y_pred_nb  = nb.predict(X_test_sc)
y_prob_nb  = nb.predict_proba(X_test_sc)[:, 1]

print("=" * 50)
print("  NAIVE BAYES  ")
print("=" * 50)
print(classification_report(y_test, y_pred_nb,
      target_names=['Não Potável', 'Potável']))
auc_nb = roc_auc_score(y_test, y_prob_nb)
print(f"  AUC-ROC: {auc_nb:.4f}")
print()

# ── 4. Regressão Logística ───────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr  = lr.predict(X_test_sc)
y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]

print("=" * 50)
print("  REGRESSÃO LOGÍSTICA  ")
print("=" * 50)
print(classification_report(y_test, y_pred_lr,
      target_names=['Não Potável', 'Potável']))
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"  AUC-ROC: {auc_lr:.4f}")
print()

# ── 5. Tabela comparativa ────────────────────────────────────────────────────
print("=" * 50)
print("  COMPARATIVO FINAL  ")
print("=" * 50)
for nome, yp, yb in [("Naive Bayes", y_pred_nb, y_prob_nb),
                      ("Reg. Logística", y_pred_lr, y_prob_lr)]:
    print(f"{nome:16s} | Acc={accuracy_score(y_test,yp):.4f} "
          f"| Prec={precision_score(y_test,yp):.4f} "
          f"| Rec={recall_score(y_test,yp):.4f} "
          f"| F1={f1_score(y_test,yp):.4f} "
          f"| AUC={roc_auc_score(y_test,yb):.4f}")

# ── 6. Matrizes de Confusão + Curvas ROC ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Naive Bayes vs Regressão Logística — Water Potability', fontsize=14, fontweight='bold')

labels = ['Não Potável', 'Potável']
colors = ['#4A90D9', '#C0392B']

for i, (nome, yp, yb, cor) in enumerate([
        ("Naive Bayes",      y_pred_nb, y_prob_nb, colors[0]),
        ("Regressão Logística", y_pred_lr, y_prob_lr, colors[1])]):

    # Matriz de confusão
    cm = confusion_matrix(y_test, yp)
    ax = axes[0][i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13})
    ax.set_title(f'Matriz de Confusão — {nome}', fontweight='bold')
    ax.set_ylabel('Real')
    ax.set_xlabel('Previsto')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, yb)
    auc = roc_auc_score(y_test, yb)
    ax2 = axes[1][i]
    ax2.plot(fpr, tpr, color=cor, lw=2, label=f'AUC = {auc:.4f}')
    ax2.plot([0,1],[0,1],'k--', lw=1)
    ax2.set_title(f'Curva ROC — {nome}', fontweight='bold')
    ax2.set_xlabel('Taxa de Falsos Positivos')
    ax2.set_ylabel('Taxa de Verdadeiros Positivos')
    ax2.legend(loc='lower right')
    ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/resultados_modelos.png', dpi=200, bbox_inches='tight')
print("\nGráfico salvo!")
