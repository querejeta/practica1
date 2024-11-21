#!/usr/bin/env python
# coding: utf-8

# ## Codificación de las variables categóricas y escalado

# In[3]:


import pandas as pd 
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import metrics
get_ipython().system('pip install category_encoders')
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc,                             silhouette_score, recall_score, precision_score, make_scorer,                             roc_auc_score, f1_score, precision_recall_curve, accuracy_score, roc_auc_score,                             classification_report, confusion_matrix

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


# In[5]:


import os
print("Directorio actual:", os.getcwd())


# In[6]:


os.chdir('/Users/admin/Desktop/ap/practica1/data')


# In[9]:


pd_loan_train=pd.read_csv("train_pd_data_preprocessing_missing_outlier.csv")
pd_loan_test= pd.read_csv("test_pd_data_preprocessing_missing_outlier.csv")


# In[10]:


pd_loan_train.columns


# In[11]:


pd_loan_train.dtypes


# ## Separación train y test de la variable objetivo

# In[8]:


# Separación de características (X) y variable objetivo (y)
X_train = pd_loan_train.drop('TARGET', axis=1)
X_test = pd_loan_test.drop('TARGET', axis=1)
y_train = pd_loan_train['TARGET']
y_test = pd_loan_test['TARGET']

# Verificar las dimensiones de los conjuntos
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[13]:


# Identificar las columnas categóricas en X_train y X_test
list_columns_cat = list(X_train.select_dtypes(include=["object", "category"]).columns)

# Dividir variables categóricas en pocas categorías y muchas categorías
few_categories = [col for col in list_columns_cat if X_train[col].nunique() <= 10]
many_categories = [col for col in list_columns_cat if X_train[col].nunique() > 10]

# Verificar la clasificación de las variables categóricas
print("Variables con pocas categorías:", few_categories)
print("Variables con muchas categorías:", many_categories)

# Importar la librería para codificación
import category_encoders as ce

# Codificación para variables con pocas categorías: OneHotEncoder
ohe = ce.OneHotEncoder(cols=few_categories)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)

# Codificación para variables con muchas categorías: CatBoostEncoder
catboost_encoder = ce.CatBoostEncoder(cols=many_categories, random_state=42)
X_train_catboost = catboost_encoder.fit_transform(X_train[many_categories], y_train)
X_test_catboost = catboost_encoder.transform(X_test[many_categories])

# Combinar los conjuntos codificados
X_train_combined = pd.concat([X_train_ohe.drop(columns=many_categories), X_train_catboost], axis=1)
X_test_combined = pd.concat([X_test_ohe.drop(columns=many_categories), X_test_catboost], axis=1)

# Verificar las dimensiones del conjunto transformado
print("Dimensiones de X_train después de combinar codificaciones:", X_train_combined.shape)
print("Dimensiones de X_test después de combinar codificaciones:", X_test_combined.shape)

# Opcional: Verificar los tipos de datos en el conjunto combinado
print("Tipos de datos en X_train_combined:", X_train_combined.dtypes.value_counts())


# Hemos utilizado un enfoque híbrido para codificar las variables categóricas según la cantidad de categorías:
# 
# ##### __OneHotEncoding:__
# 
# __Aplicado a variables con pocas categorías (≤ 10).__
# 
# Es simple y efectivo, ya que no aumenta significativamente la dimensionalidad.
# 
# ##### __CatBoostEncoder:__
# 
# __Utilizado para variables con muchas categorías (> 10).__
# 
# Reduce la dimensionalidad al transformar categorías en valores numéricos basados en su relación con la variable objetivo (TARGET).
# 
# __Justificación:__
# 
# Este enfoque equilibra la eficiencia computacional y la capacidad del modelo para capturar relaciones importantes.
# 
# Evitamos problemas de overfitting y complejidad excesiva, maximizando la información útil para el modelo.
# 
# Resultado: El dataset transformado es más manejable y adecuado para modelos avanzados como CatBoost, LightGBM o XGBoost.

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Función para graficar con límites
def plot_boxplot_with_limits(df, col, target, upper_percentile=99):
    # Calcular los límites
    upper_limit = df[col].quantile(upper_percentile / 100)
    
    # Filtrar valores solo para la visualización
    df_filtered = df[df[col] <= upper_limit]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x=target, y=col)
    plt.title(f'Boxplot de {col} (hasta el percentil {upper_percentile})')
    plt.show()

# Iterar para las variables continuas seleccionadas
vars_to_plot = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
data_grouped = pd.concat([X_train_combined[vars_to_plot], y_train], axis=1)
for var in vars_to_plot:
    plot_boxplot_with_limits(data_grouped, col=var, target='TARGET')


# ## Escalado de variables

# In[14]:


from sklearn.preprocessing import StandardScaler
import pandas as pd

# Inicializar el escalador
scaler = StandardScaler()

# Ajustar el escalador con los datos de entrenamiento y transformar
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_combined),
    columns=X_train_combined.columns,
    index=X_train_combined.index
)

# Transformar los datos de prueba
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_combined),
    columns=X_test_combined.columns,
    index=X_test_combined.index
)

# Mostrar las estadísticas descriptivas de las variables escaladas
print(X_train_scaled.describe())


# Con este proceso, las variables transformadas tienen media cercana a 0 y desviación estándar cercana a 1, lo que mejora la eficiencia en modelos sensibles a la escala.

# # Análisis: ¿Hay algún tipo de clientes más propensos a no devolver un préstamo?
# 
# ## *Conclusión General*
# Sí, existen perfiles específicos de clientes que son más propensos a no devolver un préstamo, según los datos analizados. Estos perfiles están asociados con características como el tipo de ingreso, la ocupación, el propósito del préstamo y el tipo de vivienda. A continuación, se detalla esta conclusión:
# 
# ## *Clientes más propensos a no devolver un préstamo*
# 1.⁠ ⁠*Tipo de ingreso:*
#    - Los clientes con tipo de ingreso *‘Working’* tienen el mayor número de pagos incumplidos.
#    - Profesionales como *‘Working’, *‘Commercial associates’* y *‘State servants’** son los más propensos a incumplir.
# 
# 2.⁠ ⁠*Ocupación:*
#    - Las ocupaciones más propensas al incumplimiento incluyen:
#      - *Mano de obra no calificada (Labourers)*.
#      - *Personal de ventas (Sales Staff)*.
#      - *Conductores (Drivers)*.
#    - En el caso de las mujeres, las profesiones más propensas incluyen:
#      - *Meseras (Waiters), **Personal de servicios privados, **Agentes inmobiliarios (Realty agents), **Recursos Humanos (HR Staff), **IT Staff* y *Secretarias*.
# 
# 3.⁠ ⁠*Tipo de préstamo:*
#    - Los préstamos con propósito de *‘Repair’* (reparaciones) tienen una alta tasa de incumplimiento.
# 
# 4.⁠ ⁠*Tipo de vivienda:*
#    - Los clientes que residen en viviendas diferentes a *‘Co-op apartment’* tienen mayor probabilidad de incumplir.
# 
# 5.⁠ ⁠*Propiedad de un coche:*
#    - Las *mujeres sin coche* tienen una mayor tasa de incumplimiento en comparación con otros perfiles.
# 
# 6.⁠ ⁠*Rango de ingresos:*
#    - La mayoría de los incumplidores se encuentran en los rangos de ingresos *bajos*.
# 
# ## *Clientes menos propensos a no devolver un préstamo*
# 1.⁠ ⁠*Tipo de ingreso:*
#    - Los clientes con tipo de ingreso *‘Student’, *‘Businessman’** son los menos propensos a incumplir.
# 
# 2.⁠ ⁠*Tipo de vivienda:*
#    - Los clientes que residen en vivienda *‘With parents’* presentan la menor tasa de incumplimiento.
# 
# ---
# 
# *Recomendación:*
# Basándose en el análisis, los bancos deberían enfocar esfuerzos adicionales en perfiles de alto riesgo, como trabajadores de ingresos bajos, ocupaciones con alta rotación (conductores, meseras, personal de ventas), y clientes sin activos como coches. Por otro lado, reforzar relaciones con perfiles de menor riesgo, como estudiantes, empresarios y clientes que residen con sus padres, podría aumentar la tasa de éxito en los pagos.
