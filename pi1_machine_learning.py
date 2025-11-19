# PI1 - MACHINE LEARNING SUPERVISIONADO
# Tema: Estresse e Produtividade no Trabalho
# Modelos: Regressão Logística + Random Forest
#
#Problema A – Classificação (Regressão Logística)
#
#Prever se um funcionário pode ter risco de desenvolver estresse baseado em:
#
#-horas de sono
#-horas de tela (celular/computador)
#-quantidade de água ingerida no dia
#
#Problema B – Classificação / Importância de variáveis (Random Forest)
#
#Prever se o funcionário terá alta produtividade, analisando:
#-horas trabalhadas
#-pausas durante o dia
#-atividade física semanal

# 1) IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2) ETL – CRIAÇÃO E LIMPEZA DOS DADOS

# ---------- Dados fictícios para risco de estresse ----------
estresse = pd.DataFrame({
    "sono_horas": [8, 7, 6, 5, 4, 7, 6, 3, 8, 5],
    "tela_horas": [3, 4, 6, 7, 9, 5, 8, 10, 2, 7],
    "agua_litros": [2.0, 1.5, 1.0, 1.2, 0.8, 1.7, 1.3, 0.5, 2.2, 1.1],
    "risco_estresse": [0,0,1,1,1,0,1,1,0,1]
})

# ---------- Dados fictícios para produtividade ----------
produtividade = pd.DataFrame({
    "horas_trabalhadas": [6, 7, 8, 9, 10, 7, 6, 5, 9, 8],
    "pausas": [4, 3, 3, 2, 1, 3, 4, 5, 2, 3],
    "atividade_fisica": [4, 3, 2, 1, 0, 2, 3, 4, 1, 2],
    "alta_produtividade": [1,1,1,0,0,1,1,1,0,1]
})

# 3) VISUALIZAÇÕES DOS DADOS

# ---------- Gráfico 1: Sono x Estresse ----------
plt.figure(figsize=(6,4))
plt.scatter(estresse["sono_horas"], estresse["risco_estresse"])
plt.xlabel("Horas de Sono")
plt.ylabel("Risco de Estresse")
plt.title("Sono x Estresse")
plt.grid()
plt.show()

# ---------- Gráfico 2: Horas de Tela x Estresse ----------
plt.figure(figsize=(6,4))
sns.boxplot(x="risco_estresse", y="tela_horas", data=estresse)
plt.xlabel("Risco de Estresse (0 = Não / 1 = Sim)")
plt.ylabel("Horas de Tela")
plt.title("Horas de Tela e Estresse")
plt.show()

# ---------- Gráfico 3: Trabalho x Produtividade ----------
plt.figure(figsize=(6,4))
plt.scatter(produtividade["horas_trabalhadas"], produtividade["alta_produtividade"])
plt.xlabel("Horas Trabalhadas")
plt.ylabel("Alta Produtividade")
plt.title("Horas Trabalhadas x Produtividade")
plt.grid()
plt.show()

# 4) MODELO 1 – REGRESSÃO LOGÍSTICA

print("\n==============================")
print(" MODELO 1 – REGRESSÃO LOGÍSTICA")
print("==============================\n")

X = estresse[["sono_horas", "tela_horas", "agua_litros"]]
y = estresse["risco_estresse"]

# divisão em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_log = LogisticRegression()
modelo_log.fit(X_train, y_train)

y_pred = modelo_log.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 5) MODELO 2 – RANDOM FOREST

print("\n==============================")
print(" MODELO 2 – RANDOM FOREST")
print("==============================\n")

X = produtividade[["horas_trabalhadas", "pausas", "atividade_fisica"]]
y = produtividade["alta_produtividade"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_rf.fit(X_train, y_train)

y_pred = modelo_rf.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# ---------- Importância das variáveis ----------
print("\nImportância das variáveis no Random Forest:")
for nome, valor in zip(X.columns, modelo_rf.feature_importances_):
    print(f"{nome}: {valor:.4f}")

# 6) ANÁLISE E INTERPRETAÇÃO DOS RESULTADOS

print("\n===== INTERPRETAÇÃO DOS RESULTADOS =====\n")

# INTERPRETAÇÃO DO MODELO 1 – REGRESSÃO LOGÍSTICA

print("MODELO 1 – Regressão Logística (Risco de Estresse)\n")

print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("""
A regressão logística foi usada para prever se o funcionário tem risco de estresse
(0 = baixo risco, 1 = alto risco).

► Como interpretar:
- A acurácia indica a porcentagem de previsões corretas.
- O classification_report mostra precisão, recall e F1-score para cada classe.
- Se os valores das métricas forem altos, significa que o modelo está conseguindo
  identificar corretamente funcionários com e sem risco de estresse.

► O que observar:
- Funcionários com poucas horas de sono e muitas horas de tela tendem a ser previstos como risco 1.
- Maior ingestão de água geralmente reduz o risco.
""")

print("\nMatriz de Confusão - Modelo 1:")
print(confusion_matrix(y_test, y_pred))

# INTERPRETAÇÃO DO MODELO 2 – RANDOM FOREST

print("\nMODELO 2 – Random Forest (Alta Produtividade)\n")

print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("""
O Random Forest foi usado para prever se o funcionário terá alta produtividade (1)
ou não (0).

► Como interpretar:
- A acurácia mostra o desempenho geral do modelo.
- O classification_report detalha os acertos por classe.

► Importância das variáveis:
O modelo também informa quais fatores mais influenciam a produtividade,
como:
- horas trabalhadas
- quantidade de pausas
- atividade física semanal

Isso ajuda a entender quais hábitos influenciam o desempenho no trabalho.
""")

print("\nImportância das variáveis do Random Forest:")
for nome, valor in zip(X.columns, modelo_rf.feature_importances_):
    print(f"{nome}: {valor:.4f}")

print("\nProjeto executado com sucesso!")
