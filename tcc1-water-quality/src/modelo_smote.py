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
from imblearn.over_sampling import SMOTE

df = pd.read_excel('/mnt/user-data/attachments/7ffbe7ad-83a7-57ec-bb4c-c495e259392c/575afc45-datasetcorretoPortugues.xlsx')
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE apenas no treino
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

print(f"Treino antes SMOTE: {y_train.value_counts().to_dict()}")
print(f"Treino após SMOTE:  {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")
print()

resultados = {}

for nome, modelo in [("Naive Bayes", GaussianNB()),
                     ("Regressão Logística", LogisticRegression(max_iter=1000, random_state=42))]:
    modelo.fit(X_train_sc, y_train_sm)
    yp = modelo.predict(X_test_sc)
    yb = modelo.predict_proba(X_test_sc)[:, 1]
    resultados[nome] = (yp, yb)

    print(f"{'='*50}")
    print(f"  {nome} (com SMOTE)")
    print(f"{'='*50}")
    print(classification_report(y_test, yp, target_names=['Não Potável', 'Potável']))
    print(f"  AUC-ROC: {roc_auc_score(y_test, yb):.4f}\n")

print("=" * 60)
print("  COMPARATIVO FINAL (com SMOTE)")
print("=" * 60)
print(f"{'Modelo':<20} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'AUC':>6}")
print("-" * 60)
for nome, (yp, yb) in resultados.items():
    print(f"{nome:<20} | {accuracy_score(y_test,yp):.4f} | "
          f"{precision_score(y_test,yp):.4f} | {recall_score(y_test,yp):.4f} | "
          f"{f1_score(y_test,yp):.4f} | {roc_auc_score(y_test,yb):.4f}")

# Gráfico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Naive Bayes vs Regressão Logística (com SMOTE)\nDataset: Water Potability',
             fontsize=13, fontweight='bold')

labels = ['Não Potável', 'Potável']
cores  = ['#4A90D9', '#C0392B']

for i, (nome, (yp, yb), cor) in enumerate(zip(
        resultados.keys(), resultados.values(), cores)):
    cm = confusion_matrix(y_test, yp)
    ax = axes[0][i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13})
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp)
    ax.set_title(f'Matriz de Confusão — {nome}\nAcc={acc:.3f} | F1={f1:.3f}', fontweight='bold')
    ax.set_ylabel('Real'); ax.set_xlabel('Previsto')

    fpr, tpr, _ = roc_curve(y_test, yb)
    auc = roc_auc_score(y_test, yb)
    ax2 = axes[1][i]
    ax2.plot(fpr, tpr, color=cor, lw=2.5, label=f'AUC = {auc:.4f}')
    ax2.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
    ax2.fill_between(fpr, tpr, alpha=0.08, color=cor)
    ax2.set_title(f'Curva ROC — {nome}', fontweight='bold')
    ax2.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax2.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])
    ax2.grid(alpha=0.3)

plt.tight_layout()
out = '/mnt/user-data/outputs/resultados_smote.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nGráfico salvo em: {out}")
