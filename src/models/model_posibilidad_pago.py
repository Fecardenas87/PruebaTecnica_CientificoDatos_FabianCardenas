"""
MODELO_POSIBILIDAD_PAGO

Objetivo:
Entrenar un modelo de clasificación binaria que estime la probabilidad
de que un producto realice al menos un pago (target_pago = 1) en el
periodo analizado.

Definición del target:
- target_pago = 1 → El producto registra al menos un pago
- target_pago = 0 → El producto no registra pagos

Alcance:
Este modelo se plantea como una primera aproximación analítica.
No se busca optimización avanzada sino validación del pipeline
ETL → Features → Modelo → Evaluación.

Modelo utilizado:
- Regresión Logística (baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression


"CARGA DE DATOS"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "Datos_Procesados"

DATA_PATH = DATA_PROCESSED / "df_evolucion_enriquecida.txt"

df = pd.read_csv(DATA_PATH, sep="|")

print(df.head())
print(df.shape)

"DEFINICION DE LA VARIABLE TARGET -- target_pago = 1 si el producto registra al menos un pago"
# df["target_pago"] = (df["num_pagos"] > 0).astype(int)
# df["target_pago"] = (
#     df.groupby("identificacion")["num_pagos"]
#       .transform("sum") > 0 ---------No me funciono porque el corte mensual me deja el dataset muy desbalanceado
# ).astype(int)

df["target_pago"] = (df["num_pagos"] > 0).astype(int)
print("Distribución del target:")
print(df["target_pago"].value_counts())

print("\nDistribución normalizada:")
print(df["target_pago"].value_counts(normalize=True))


#VISUALIZACIÓN DE LA DISTRIBUCIÓN DEL TARGET
plt.figure(figsize=(5, 4))

df["target_pago"].value_counts().plot(
    kind="bar",
    title="Distribución del Target: Pago por producto"
)

plt.xticks(rotation=0)
plt.xlabel("Clase (0 = No paga, 1 = Paga)")
#plt.xlabel("Clase")
#plt.ylabel("Cantidad de clientes")
plt.ylabel("Cantidad de productos")
plt.show()

print("Distribución normalizada del target:")
print(df["target_pago"].value_counts(normalize=True))

"VARIABLES -- Variables numéricas disponibles post ETL"
features = [
    "saldo_capital_mes",
    "pago_minimo",
    "dias_mora",
    #"total_pagos",          # Las retiro para corregir el modelo
    #"num_pagos",
    "saldo_total_cliente",
]

X = df[features]
y = df["target_pago"]


"ENTRENAMIENTO Y PRUEBA -- Estratificada para conservar la proporción del target"
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

"ESCALADO --  Necesario para regresión logística"
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"ENTRENAMIENTO DEL MODELO -- Utilizo class_weight='balanced' para mitigar desbalance"
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

"EVALUACIÓN"
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- MATRIZ DE CONFUSIÓN ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_proba))

"MATRIZ DE CONFUSIÓN -- visual"
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión – Modelo Posibilidad de Pago")
plt.show()
