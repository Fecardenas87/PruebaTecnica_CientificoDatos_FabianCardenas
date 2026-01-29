# 📊 Prueba Técnica - Científico de Datos
## Fabian Cárdenas

📌 Descripción general

Este proyecto corresponde a una prueba técnica de ciencia de datos, cuyo objetivo es construir un pipeline completo de datos, desde la ingestión y transformación (ETL) hasta el entrenamiento y evaluación de un modelo de machine learning que estime la posibilidad de pago de un cliente/producto.

El enfoque principal está en:

* Correcta preparación de los datos
* Trazabilidad del proceso
* Claridad en la lógica de negocio
* Implementación de un modelo base (baseline)
* No se busca optimización avanzada del modelo sino validar el proceso end-to-end.

🗂️ Estructura del proyecto

PruebaTecnica_FabianCardenas/
│
├── data/
│   ├── raw/                      # Datos originales (fuente)
│   └── Datos_Procesados/         # Datos procesados por el ETL
│       ├── evolucion_enriquecida.csv
│       └── df_evolucion_enriquecida.txt
│
├── src/
│   ├── etl/
│   │   └── etl_evolucion.py      # Proceso ETL principal
│   │
│   └── models/
│       └── model_posibilidad_pago.py  # Entrenamiento y evaluación del modelo
│
├── venv/                         # Entorno virtual (no versionado)
│
├── README.md                     # Documentación del proyecto
└── requirements.txt              # Dependencias del proyecto

⚙️ Requisitos técnicos

* Python 3.9+
* Librerías principales:
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

Instalación de dependencias:

* pip install -r requirements.txt

🔄 Proceso ETL – etl_evolucion.py

Objetivo
Construir un dataset analítico unificado a partir de:

* Información de evolución de obligaciones
* Información histórica de pagos
* Variables de negocio relevantes
* Principales pasos del ETL
* Carga de datos base
* Limpieza y estandarización de llaves
* Normalización de identificadores de obligación y cuenta
* Agregación de pagos
* Total de pagos
* Número de pagos aprobados
* Cruce evolución ↔ pagos
* Validaciones
* Conteo de registros con pagos
* Verificación de llaves cruzadas
* Exportación del dataset final
* Salida del ETL
* Archivo generado:
* data/Datos_Procesados/df_evolucion_enriquecida.txt

🤖 Modelo de Machine Learning – model_posibilidad_pago.py

Objetivo del modelo

Clasificar si un producto tiene probabilidad de realizar al menos un pago.

Definición del target
target_pago = 1 → El producto registra al menos un pago
target_pago = 0 → El producto no registra pagos

Variables utilizadas:

* saldo_capital_mes
* pago_minimo
* dias_mora
* total_pagos
* num_pagos

Modelo seleccionado:

Regresión Logística
Usada como modelo baseline
class_weight="balanced" para manejar desbalance de clases

Evaluación
Se utilizan las siguientes métricas:

* Matriz de confusión
* Precision, Recall y F1-score
* ROC AUC

El desempeño del modelo es consistente con un enfoque exploratorio y sirve como base para mejoras futuras mediante:

* Feature engineering adicional
* Variables temporales
* Modelos más complejos

📊 Resultados principales

Dataset con clases desbalanceadas (≈12% clase positiva)

Modelo baseline funcional

Pipeline reproducible y trazable

Código modular y documentado

🚀 Ejecución del proyecto
1️- Ejecutar ETL
python src/etl/etl_evolucion.py

2️- Entrenar y evaluar el modelo
python src/models/model_posibilidad_pago.py

🔮 Posibles mejoras futuras

* Ingeniería de variables temporales (ventanas móviles)
* Modelos más robustos (Random Forest, XGBoost)
* Validación cruzada
* Ajuste de umbrales de decisión
* Análisis de interpretabilidad (SHAP)

👤 Autor

Fabian Cárdenas
Prueba Técnica – Científico de Datos Cobranzas Beta
