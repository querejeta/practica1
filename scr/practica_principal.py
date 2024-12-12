#!/usr/bin/env python
# coding: utf-8

# # Práctica Final
# 
# **Carmen Querejeta, Carmen Huerta y Olivia Ares**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored, cprint
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
import pyarrow as pa
import pyarrow.parquet as pq
import importlib
import sys

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import pickle
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[126]:


import os
print("Directorio actual:", os.getcwd())


# In[127]:


os.chdir('c:/Users/huert/OneDrive/Desktop/APRENDIZAJE/practica2/data')


# In[128]:


import pandas as pd

# Cargar datasets con rutas relativas
train_path = './train_pd_data_preprocessing_missing_outlier.csv'
test_path = './test_pd_data_preprocessing_missing_outlier.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Ver las primeras filas de los datasets para confirmar
print("Datos de entrenamiento:")
print(train_data.head())

print("\nDatos de prueba:")
print(test_data.head())


# ## Feature Engineering 

# In[129]:


# Función para clasificar variables por tipo
def clasificar_columnas(df):
    cat_vars = df.select_dtypes(include=['object']).columns.tolist()
    num_vars = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    return cat_vars, num_vars

# Aplicar la función al dataset de entrenamiento
cat_vars, num_vars = clasificar_columnas(train_data)

print(f"Variables categóricas: {len(cat_vars)}")
print(cat_vars)
print(f"\nVariables numéricas: {len(num_vars)}")
print(num_vars)


# Definimos una función para clasificar las columnas de train_data según su tipo de dato en variables categóricas y numéricas. La función identifica columnas categóricas (object) y numéricas (float64 y int64), devolviendo ambas categorías en listas.

# In[130]:


# Obtener información sobre las variables categóricas
for col in cat_vars:
    print(f"\nColumna: {col}")
    print(f"- Valores únicos: {train_data[col].unique()}")
    print(f"- Número de valores únicos: {train_data[col].nunique()}")


# Analizamos cada variable categórica en train_data mostrando su nombre, los valores únicos presentes y la cantidad de estos valores. Esto permite entender la distribución de categorías en el dataset y facilita el diseño de estrategias de codificación o tratamiento de datos.

# ## Codificación de Variables Categóricas

# En este proceso, transformamos las variables categóricas en un formato numérico para que puedan ser utilizadas por los modelos de Machine Learning. Utilizamos dos técnicas principales según el número de categorías:
# 
# 1.⁠ ⁠*One-Hot Encoding*:
#    - Aplicado a variables con pocas categorías.
#    - Transforma cada categoría en una nueva columna binaria (0 o 1).
#    - Variables codificadas: 
#      ⁠ NAME_CONTRACT_TYPE ⁠, ⁠ CODE_GENDER ⁠, ⁠ FLAG_OWN_CAR ⁠, ⁠ FLAG_OWN_REALTY ⁠, 
#      ⁠ NAME_TYPE_SUITE ⁠, ⁠ NAME_INCOME_TYPE ⁠, ⁠ NAME_EDUCATION_TYPE ⁠, 
#      ⁠ NAME_FAMILY_STATUS ⁠, ⁠ NAME_HOUSING_TYPE ⁠, ⁠ WEEKDAY_APPR_PROCESS_START ⁠, 
#      ⁠ EMERGENCYSTATE_MODE ⁠.
# 
# 2.⁠ ⁠*Target Encoding*:
#    - Aplicado a variables con muchas categorías.
#    - Sustituye cada categoría por la media del valor objetivo (⁠ TARGET ⁠) para evitar explosión dimensional.
#    - Variables codificadas: 
#      ⁠ OCCUPATION_TYPE ⁠, ⁠ ORGANIZATION_TYPE ⁠.
# 
# El dataset resultante contiene únicamente valores numéricos, lo que lo hace apto para ser utilizado en algoritmos de Machine Learning

# In[131]:


# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder


# In[132]:


from category_encoders import TargetEncoder
from sklearn.model_selection import KFold

# Función para procesar variables categóricas
def procesar_categoricas(train_data, test_data, cat_vars):
    # Identificar variables con pocas y muchas categorías
    pocas_categorias = [var for var in cat_vars if train_data[var].nunique() <= 10]
    muchas_categorias = [var for var in cat_vars if train_data[var].nunique() > 10]
    
    print(f"Variables con pocas categorías (One-Hot): {pocas_categorias}")
    print(f"Variables con muchas categorías (Target Encoding): {muchas_categorias}")

    # Gestión de valores raros
    def gestionar_valores_raros(df, columna, umbral=0.01):
        # Calculamos la frecuencia de las categorías
        frecuencias = df[columna].value_counts(normalize=True)
        # Identificamos las categorías raras
        categorias_raras = frecuencias[frecuencias < umbral].index
        # Sustituimos las categorías raras por 'Other'
        df[columna] = df[columna].apply(lambda x: 'Other' if x in categorias_raras else x)

    # Gestionar valores raros en todas las variables categóricas
    for col in cat_vars:
        gestionar_valores_raros(train_data, col)
        gestionar_valores_raros(test_data, col)
    
    # Aplicar One-Hot Encoding para variables con pocas categorías
    train_data = pd.get_dummies(train_data, columns=pocas_categorias, drop_first=True)
    test_data = pd.get_dummies(test_data, columns=pocas_categorias, drop_first=True)
    
    # Asegurar consistencia en columnas de One-Hot entre train y test
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

    # Aplicar Target Encoding para variables con muchas categorías
    target_encoder = TargetEncoder(cols=muchas_categorias)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train_data):
        # Dividir en train y validación
        train_split, val_split = train_data.iloc[train_idx], train_data.iloc[val_idx]
        # Ajustar el codificador en el conjunto de entrenamiento
        target_encoder.fit(train_split[muchas_categorias], train_split['TARGET'])
        # Transformar en validación y test
        train_data.iloc[val_idx, train_data.columns.get_indexer(muchas_categorias)] = target_encoder.transform(val_split[muchas_categorias])
    
    # Transformar el conjunto de test
    test_data[muchas_categorias] = target_encoder.transform(test_data[muchas_categorias])
    
    return train_data, test_data

# Llamar a la función para procesar las variables categóricas
cat_vars = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'WEEKDAY_APPR_PROCESS_START',
            'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'EMERGENCYSTATE_MODE']

train_data_preprocessed, test_data_preprocessed = procesar_categoricas(train_data, test_data, cat_vars)


# In[133]:


print(train_data_preprocessed.head())
print(test_data_preprocessed.head())


# ## Separación X e Y

# In[134]:


# Separar X e y para el conjunto de entrenamiento
y_train = train_data_preprocessed['TARGET']
X_train = train_data_preprocessed.drop(columns=['TARGET'])

# Separar X e y para el conjunto de prueba
y_test = test_data_preprocessed['TARGET']
X_test = test_data_preprocessed.drop(columns=['TARGET'])

# Verificar las dimensiones de los conjuntos
print("Dimensiones del conjunto de entrenamiento:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

print("\nDimensiones del conjunto de prueba:")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


# Hemos separado las variables predictoras (X_train y X_test) de la variable objetivo (y_train y y_test) para los conjuntos de entrenamiento y prueba. Esto es esencial para preparar los datos para el entrenamiento y la evaluación del modelo. También hemos verificado las dimensiones de los conjuntos resultantes para confirmar que la separación se realizó correctamente, mostrando el número de observaciones y características de cada conjunto.

# In[ ]:


print("Distribución de la variable TARGET en el conjunto de entrenamiento:")
print(y_train.value_counts(normalize=True))


# ## Feature Selection

# In[136]:


from imblearn.over_sampling import SMOTE


# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


# In[137]:


from sklearn.preprocessing import StandardScaler

# Instanciar el escalador
scaler = StandardScaler()

# Escalar los conjuntos de entrenamiento y prueba
X_train_scaled = scaler.fit_transform(X_train_balanced)  # Ajustar y transformar en el conjunto de entrenamiento balanceado
X_test_scaled = scaler.transform(test_data_preprocessed.drop(columns=['TARGET']))  # Transformar el conjunto de prueba


# In[138]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Entrenar un modelo Random Forest en los datos de entrenamiento
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Obtener la importancia de las características
importances = rf.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Ordenar las características por importancia
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualizar las características más importantes
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15][::-1], importance_df['Importance'][:15][::-1], align='center')
plt.xlabel('Importancia')
plt.title('Características más importantes (Random Forest)')
plt.show()


# Hemos entrenado un modelo de clasificación Random Forest para identificar las características más relevantes del conjunto de datos. Utilizamos la métrica de importancia de características que proporciona el modelo para cuantificar el impacto de cada variable en la predicción. Luego, visualizamos las 15 características más importantes en un gráfico de barras horizontales, donde se observa que EXT_SOURCE_2, EXT_SOURCE_3 y DAYS_BIRTH son las variables más influyentes en el modelo. Esto nos ayuda a comprender cuáles son los factores clave que afectan a la variable objetivo (TARGET) 

# In[139]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Solo usar X_train


# In[140]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Escalar las características de entrenamiento
print(f"Dimensiones de X_train_scaled: {X_train_scaled.shape}")


# In[141]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Escalar después del balanceo
X_train_scaled = scaler.fit_transform(X_train_balanced)
y_train = y_train_balanced  # Actualizar y_train con la versión balanceada
print(f"Dimensiones de X_train_scaled: {X_train_scaled.shape}")
print(f"Dimensiones de y_train: {y_train.shape}")


# Hemos aplicado la técnica de sobremuestreo SMOTE (Synthetic Minority Oversampling Technique) para equilibrar las clases de la variable objetivo (y_train) en el conjunto de entrenamiento. Esto genera ejemplos sintéticos para la clase minoritaria, mejorando la representación de ambas clases en el modelo.
# 
# Posteriormente, escalamos las variables predictoras (X_train_balanced) para normalizar sus valores, lo cual es necesario para modelos sensibles a la escala. Finalmente, actualizamos y_train con la versión balanceada y verificamos las dimensiones del conjunto resultante. Este paso es crucial para mitigar el impacto del desequilibrio de clases en el rendimiento del modelo.

# In[142]:


# Verificar las dimensiones de X_train_scaled
num_filas, num_columnas = X_train_scaled.shape

# Mostrar el número de filas y columnas
print(f"Número de filas en X_train_scaled: {num_filas}")
print(f"Número de columnas en X_train_scaled: {num_columnas}")


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt

# Crear un subconjunto para entrenamiento rápido
subset_size = 200000  
X_train_subset = X_train_scaled[:subset_size]
y_train_subset = y_train[:subset_size]

# Regularización Lasso con subset
sel_lasso = SelectFromModel(
    LogisticRegression(C=1, penalty='l1', solver='liblinear'),
    threshold=0.08  
)

# Ajustar el modelo Lasso al subconjunto
sel_lasso.fit(X_train_subset, y_train_subset)

# Obtener las características seleccionadas
selected_feat_lasso = X_train.columns[sel_lasso.get_support()]
print(f"Características seleccionadas por Lasso:\n{selected_feat_lasso}")

# Mostrar el número total de características y las seleccionadas
print(f'Total de características: {X_train_scaled.shape[1]}')
print(f'Características seleccionadas: {len(selected_feat_lasso)}')

# Coeficientes del modelo Lasso
df_coeficientes_lasso = pd.DataFrame({
    'predictor': X_train.columns,
    'coef': sel_lasso.estimator_.coef_.flatten()
}).sort_values(by='coef', key=abs, ascending=False)

# Visualización de los coeficientes del modelo Lasso
fig, ax = plt.subplots(figsize=(16, 3.84))
ax.stem(df_coeficientes_lasso['predictor'], df_coeficientes_lasso['coef'], markerfmt=' ')
plt.xticks(rotation=90, ha='right', size=10)
ax.set_xlabel('Variable')
ax.set_ylabel('Coeficientes')
ax.set_title('Coeficientes del modelo Lasso')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# Hemos aplicado la regularización Lasso (L1), implementada a través de LogisticRegression con penalización l1, para realizar una selección de características en el conjunto de entrenamiento. Esta técnica fuerza a ciertos coeficientes de las variables a ser exactamente cero, eliminando aquellas características menos relevantes.
# 
# Para facilitar el proceso, hemos trabajado con un subconjunto de los datos de entrenamiento ya que su tamaño es muy grande y tardaba mucho en ejecutar, hemos seleccionado una muestra de 200.000 valores. Tras entrenar el modelo, seleccionamos las características que superan un umbral predefinido de importancia (threshold=0.08), quedándonos únicamente con las más relevantes.

# Tras aplicar la regularización Lasso al subconjunto de datos de entrenamiento, hemos identificado un total de 19 características relevantes de las 107 disponibles. Estas características fueron seleccionadas porque sus coeficientes no fueron reducidos a cero, lo que indica que tienen una contribución significativa en la predicción de la variable objetivo (TARGET).
# 
# Entre las características seleccionadas, variables como EXT_SOURCE_2, EXT_SOURCE_3, y DAYS_EMPLOYED destacan como factores clave. Además, se observan variables categóricas codificadas como CODE_GENDER_M y NAME_INCOME_TYPE_Pensioner, lo que refleja la importancia de ciertas categorías en las predicciones.
# 
# La visualización de los coeficientes del modelo muestra el impacto de cada característica seleccionada, evidenciando qué variables están asociadas positivamente o negativamente con la probabilidad de la clase objetivo. Esta selección simplifica el modelo, mejora la interpretabilidad y elimina ruido al descartar características irrelevantes.
# 
# Esta decisión de utilizar Lasso para la selección de variables se basa en su capacidad para realizar una reducción dimensional eficiente, conservando las variables más importantes y ayudando a construir un modelo más parsimonioso y manejable.

# In[175]:


df_coeficientes_lasso


# In[144]:


import pandas as pd

# Convertir los arrays escalados de vuelta a DataFrame, usando los nombres originales de las columnas
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Definir las características seleccionadas por Lasso
selected_features_lasso = [
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'FLAG_EMP_PHONE',
    'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'YEARS_BEGINEXPLUATATION_MEDI',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
    'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'DAYS_EMPLOYED_YEARS',
    'CODE_GENDER_M', 'FLAG_OWN_CAR_Y', 'NAME_INCOME_TYPE_Other',
    'NAME_INCOME_TYPE_Pensioner',
    'NAME_EDUCATION_TYPE_Secondary / secondary special'
]

# Filtrar las características seleccionadas en los conjuntos de datos
X_train_lasso = X_train_scaled_df[selected_features_lasso]
X_test_lasso = X_test_scaled_df[selected_features_lasso]

# Verificar las dimensiones de los nuevos conjuntos
print("Dimensiones del conjunto de entrenamiento después de Lasso:")
print(f"X_train_lasso: {X_train_lasso.shape}, y_train: {y_train.shape}")

print("\nDimensiones del conjunto de prueba después de Lasso:")
print(f"X_test_lasso: {X_test_lasso.shape}, y_test: {y_test.shape}")


# ## MODELADO

# In[145]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


# In[146]:


# Separar características numéricas y categóricas seleccionadas
numeric_features = [
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED',
    'FLAG_EMP_PHONE', 'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_2',
    'EXT_SOURCE_3', 'YEARS_BEGINEXPLUATATION_MEDI',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
    'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'DAYS_EMPLOYED_YEARS'
]

categorical_features = [
    'CODE_GENDER_M', 'FLAG_OWN_CAR_Y', 'NAME_INCOME_TYPE_Other',
    'NAME_INCOME_TYPE_Pensioner', 'NAME_EDUCATION_TYPE_Secondary / secondary special'
]


# In[147]:


# Asegurarse de que las columnas categóricas sean de tipo object
X_train_lasso[categorical_features] = X_train_lasso[categorical_features].astype('object')
X_test_lasso[categorical_features] = X_test_lasso[categorical_features].astype('object')


# In[ ]:


# Definir los pasos en el pipeline numérico
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())                   
])


# Definimos un pipeline de transformación para variables numéricas, que incluye dos pasos esenciales:
# 
#     - Imputación de valores faltantes: Utilizamos SimpleImputer con la estrategia de la mediana para reemplazar los valores nulos en las variables numéricas. Esto es útil para manejar valores ausentes de manera robusta, especialmente en datos con distribuciones sesgadas.
# 
#     - Estandarización: Aplicamos StandardScaler para escalar las variables numéricas, ajustándolas a una media de 0 y una desviación estándar de 1. Esto asegura que todas las variables tengan la misma escala, lo cual es crucial para muchos algoritmos de aprendizaje automático sensibles a la magnitud de los datos.

# In[ ]:


# Definir los pasos en el pipeline categórico
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                     
])


# Definimos un pipeline de transformación para variables categóricas, compuesto por los siguientes pasos:
# 
#     - Imputación de valores faltantes: Usamos SimpleImputer con la estrategia 'constant', reemplazando los valores nulos con la etiqueta 'missing'. Esto asegura que las categorías faltantes sean manejadas explícitamente en lugar de ignorarse o eliminarse.
# 
#     - Codificación One-Hot: Aplicamos OneHotEncoder para transformar las variables categóricas en representaciones binarias. La opción handle_unknown='ignore' asegura que cualquier categoría desconocida encontrada durante la inferencia no cause errores, sino que sea ignorada.

# In[150]:


# Crear un transformador de columnas para procesar numéricas y categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[151]:


# Aplicar el preprocesador a los conjuntos de datos
X_train_preprocessed = preprocessor.fit_transform(X_train_lasso)
X_test_preprocessed = preprocessor.transform(X_test_lasso)


# In[152]:


# Verificar dimensiones después del preprocesado
print("Dimensiones después del preprocesado:")
print(f"X_train_preprocessed: {X_train_preprocessed.shape}")
print(f"X_test_preprocessed: {X_test_preprocessed.shape}")


# In[153]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Lista de clasificadores a evaluar, incluyendo XGBoost
classifiers = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42, n_estimators=100),
    AdaBoostClassifier(random_state=42, n_estimators=100),
    GradientBoostingClassifier(random_state=42, n_estimators=100),
    XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
]

# Evaluar los modelos
for classifier in classifiers:
    # Crear el pipeline que incluye el preprocesador y el clasificador
    pipe = Pipeline(steps=[
        ('classifier', classifier)
    ])
    
    # Ajustar el modelo al conjunto de entrenamiento preprocesado
    pipe.fit(X_train_preprocessed, y_train)
    
    # Predecir en el conjunto de prueba preprocesado
    y_pred = pipe.predict(X_test_preprocessed)
    
    # Calcular la precisión y mostrar el informe de clasificación
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Modelo: {classifier.__class__.__name__}")
    print(f"Precisión del modelo: {accuracy:.3f}")
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)


# ### Comentario:
# 
# En este bloque hemos evaluado diferentes clasificadores, incluyendo **Decision Tree**, **Random Forest**, **AdaBoost**, **Gradient Boosting**, y **XGBoost**, para comparar su desempeño en el conjunto de prueba. 
# 
# Para cada modelo:
# 1. Creamos un pipeline que incluye directamente el clasificador.
# 2. Entrenamos el modelo con los datos preprocesados de entrenamiento (`X_train_preprocessed`, `y_train`).
# 3. Realizamos predicciones en el conjunto de prueba preprocesado (`X_test_preprocessed`).
# 4. Calculamos la **precisión** y mostramos un informe de clasificación con métricas como **precision**, **recall** y **f1-score** para cada clase.
# 
# ### Resultados obtenidos:
# - **Precisión Global:** La precisión de los modelos varía entre 81.5% y 90.3%, siendo el modelo **XGBoost** el que logra la mayor precisión (90.3%).
# - **Balance de Clases:** En todos los modelos, la clase mayoritaria (`0.0`) tiene una precisión mucho mayor que la clase minoritaria (`1.0`).
# 
# 
# ### Conclusiones:
# - XGBoost muestra el mejor desempeño general en términos de precisión.

# In[154]:


# Convertir columnas categóricas a tipo "category"
categorical_columns = [
    'CODE_GENDER_M', 'FLAG_OWN_CAR_Y', 'NAME_INCOME_TYPE_Other',
    'NAME_INCOME_TYPE_Pensioner', 'NAME_EDUCATION_TYPE_Secondary / secondary special'
]

for col in categorical_columns:
    X[col] = X[col].astype('category').cat.codes  # Convertir a códigos numéricos

# Aplicar SMOTE nuevamente
X_smote, y_smote = smote.fit_resample(X, y)

# Dividir el conjunto balanceado en entrenamiento y validación
xtrain, xval, ytrain, yval = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

from xgboost import XGBClassifier

# Ajustar el peso de la clase positiva
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_model = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_model.fit(xtrain, ytrain)

# Evaluar el modelo
predictions = xgb_model.predict(xval)
print("\nInforme de clasificación después de SMOTE:")
print(classification_report(yval, predictions))

print("\nMatriz de confusión:")
print(confusion_matrix(yval, predictions))


# ### Análisis de resultados:
# 
# Tras aplicar **SMOTE** y entrenar el modelo XGBoost con ajuste de pesos (`scale_pos_weight`), el modelo obtuvo un desempeño sólido, como se refleja en el informe de clasificación y la matriz de confusión:
# 
# #### **Informe de clasificación:**
# - **Clase 0.0 (Mayoritaria):**
#   - **Precisión (Precision):** 90%, lo que indica que el 90% de las predicciones para la clase `0.0` son correctas.
#   - **Recall:** 98%, lo que significa que el modelo identifica correctamente el 98% de las instancias reales de la clase `0.0`.
#   - **F1-Score:** 94%, que combina precisión y recall en una única métrica.
# 
# - **Clase 1.0 (Minoritaria):**
#   - **Precisión:** 98%, lo que indica que casi todas las predicciones para la clase `1.0` son correctas.
#   - **Recall:** 89%, lo que significa que el modelo detecta correctamente el 89% de las instancias reales de la clase `1.0`.
#   - **F1-Score:** 93%, reflejando un buen balance entre precisión y recall.
# 
# - **Exactitud (Accuracy):** 94%, indicando que el modelo clasifica correctamente el 94% de todas las instancias.
# 
# 
# #### **Conclusión:**
# El modelo muestra un desempeño equilibrado entre ambas clases gracias a la combinación de **SMOTE** y el ajuste de pesos en XGBoost:
# 1. La clase minoritaria (`1.0`) tiene un recall aceptable del 89%, lo que indica que la mayoría de los positivos son correctamente identificados.
# 2. La clase mayoritaria (`0.0`) mantiene un excelente desempeño en precisión y recall.
# 3. Los falsos negativos en la clase minoritaria aún existen (5,143 casos), pero el modelo logra un equilibrio sólido entre precisión y recall.

# In[155]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Definir el rango de hiperparámetros
param_grid = {
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0]
}

# Crear el modelo base
xgb_model = xgb.XGBClassifier(random_state=42)

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Número de combinaciones a probar
    scoring='f1',  # Métrica para evaluar
    cv=3,  # Número de folds para validación cruzada
    random_state=42,
    n_jobs=-1  # Usar todos los núcleos disponibles
)

# Ajustar el modelo con los datos balanceados
random_search.fit(xtrain, ytrain)

# Mostrar los mejores parámetros y el mejor puntaje
print("Mejores parámetros encontrados:")
print(random_search.best_params_)
print(f"Mejor puntaje obtenido en CV: {random_search.best_score_:.4f}")

# Evaluar el modelo optimizado en el conjunto de validación
best_model = random_search.best_estimator_
predictions = best_model.predict(xval)

print("\nInforme de clasificación después de la optimización:")
print(classification_report(yval, predictions))
print("\nMatriz de confusión:")
print(confusion_matrix(yval, predictions))


# ### Análisis de resultados:
# 
# En este bloque hemos utilizado **RandomizedSearchCV** para realizar una búsqueda eficiente de hiperparámetros óptimos para el modelo **XGBoost**, con el objetivo de maximizar su desempeño. Los pasos realizados y los resultados obtenidos son los siguientes:
# 
# #### **1. Definición del rango de hiperparámetros:**
# Se especificaron los posibles valores de los hiperparámetros clave:
# - `colsample_bytree`: Proporción de columnas utilizadas en cada árbol.
# - `learning_rate`: Tasa de aprendizaje del modelo.
# - `max_depth`: Profundidad máxima de los árboles.
# - `n_estimators`: Número de árboles en el modelo.
# - `subsample`: Proporción de datos usados para entrenar cada árbol.
# 
# Esto permite explorar combinaciones diversas para optimizar el desempeño del modelo.
# 
# #### **2. Configuración de RandomizedSearchCV:**
# - **Número de iteraciones (`n_iter`):** Se probaron 50 combinaciones aleatorias de hiperparámetros.
# - **Métrica (`scoring`):** Se utilizó la métrica **f1-score**, que balancea precisión y recall, ideal para problemas con desequilibrio de clases.
# - **Validación cruzada (`cv`):** Se utilizó 3 folds para evaluar cada combinación de hiperparámetros.
# - **Paralelización (`n_jobs`):** Se emplearon todos los núcleos disponibles para acelerar el proceso.
# 
# #### **3. Mejores parámetros encontrados:**
# Los hiperparámetros óptimos encontrados son:
# - `subsample`: 1.0 (utiliza el 100% de las muestras en cada árbol).
# - `n_estimators`: 300
# - `max_depth`: 8 (árboles más profundos permiten capturar patrones complejos).
# - `learning_rate`: 0.2 (una tasa de aprendizaje moderada para convergencia eficiente).
# - `colsample_bytree`: 0.8 (usa el 80% de las columnas para evitar sobreajuste).
# 
# El mejor puntaje de **f1-score** obtenido durante la validación cruzada fue **0.9345**, reflejando un modelo equilibrado entre precisión y recall.
# 
# #### **4. Evaluación del modelo optimizado:**
# Tras entrenar el modelo con los mejores parámetros, se evaluó en el conjunto de validación (`xval`, `yval`):
# - **Informe de clasificación:**
#   - **Clase 0.0 (Mayoritaria):** Excelente desempeño, con una precisión del **90%** y un recall del **99%**, resultando en un F1-score de **94%**.
#   - **Clase 1.0 (Minoritaria):** Muy buen desempeño, con una precisión del **98%** y un recall del **89%**, resultando en un F1-score de **94%**.
# 
# 
# #### **Conclusión:**
# La optimización de hiperparámetros mejoró el desempeño del modelo, especialmente para la clase minoritaria (`1.0`). El balance entre precisión y recall es excelente, y el modelo es capaz de capturar patrones complejos en los datos sin sobreajuste. Este modelo optimizado es adecuado para producción, ya que mantiene un alto desempeño general y trata ambas clases de manera equitativa.

# In[157]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predecir con el mejor modelo
predictions = best_model.predict(xval)

# Generar la matriz de confusión
cm = confusion_matrix(yval, predictions)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title("Matriz de Confusión - XGBoost Optimizado")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.show()


# In[158]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predecir con el mejor modelo
predictions = best_model.predict(xval)

# Generar la matriz de confusión normalizada
cm_normalized = confusion_matrix(yval, predictions, normalize='true')

# Visualizar la matriz de confusión normalizada
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title("Matriz de Confusión Normalizada - XGBoost Optimizado")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.show()


# La primera matriz de confusión muestra el número absoluto de predicciones correctas e incorrectas, reflejando un excelente desempeño del modelo XGBoost optimizado con una gran cantidad de aciertos en ambas clases (0 y 1). La mayoría de los falsos negativos y falsos positivos son mínimos en comparación con los verdaderos positivos y negativos.
# 
# La segunda matriz, normalizada, complementa la información al mostrar las proporciones relativas. Se observa que el modelo clasifica correctamente el 99% de los casos de la clase 0 y el 89% de los casos de la clase 1. Esto resalta el balance entre precisión y recall en términos proporcionales, evidenciando el buen rendimiento del modelo en ambas clases a pesar del desequilibrio inicial.

# In[ ]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Obtener las probabilidades de predicción para la clase positiva (1)
yhat = best_model.predict_proba(xval)[:, 1]  

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(yval, yhat)

# Calcular el área bajo la curva (AUC)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Sin habilidades (AUC = 0.50)')
plt.plot(fpr, tpr, marker='.', color='blue', label=f'Modelo XGBoost (AUC = {roc_auc:.2f})')

# Etiquetas de los ejes
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Modelo XGBoost Optimizado')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# La **Curva ROC** del modelo **XGBoost optimizado** muestra un excelente desempeño en la clasificación. El área bajo la curva (AUC) es de **0.97**, lo cual indica que el modelo tiene una alta capacidad para distinguir entre las dos clases (`0` y `1`).
# 
# - **Interpretación del AUC:** Un AUC cercano a 1 implica que el modelo tiene una capacidad casi perfecta para separar correctamente los positivos de los negativos. Esto confirma que el modelo optimizado es robusto y confiable.
# 
# - **Curva ROC:** La curva se acerca al vértice superior izquierdo, lo que indica una alta **tasa de verdaderos positivos (TPR)** y una baja **tasa de falsos positivos (FPR)** en la mayoría de los umbrales de decisión.
# 
# Este resultado valida la calidad del modelo y su capacidad para manejar el desequilibrio de clases, combinando SMOTE, ajuste de pesos y optimización de hiperparámetros. El modelo es adecuado para producción, manteniendo un buen balance entre precisión y recall.

# In[160]:


from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Convertir columnas categóricas a códigos numéricos
categorical_columns = [
    'CODE_GENDER_M', 'FLAG_OWN_CAR_Y', 'NAME_INCOME_TYPE_Other',
    'NAME_INCOME_TYPE_Pensioner', 'NAME_EDUCATION_TYPE_Secondary / secondary special'
]

for col in categorical_columns:
    X[col] = X[col].astype('category').cat.codes  # Convertir a códigos numéricos

# Aplicar SMOTE para balancear el dataset
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Dividir el conjunto balanceado en entrenamiento y validación
xtrain, xval, ytrain, yval = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Entrenar el modelo XGBoost
xgb_model = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

# Ajustar el modelo
xgb_model.fit(xtrain, ytrain)

# Generar probabilidades para la clase positiva
prob_predictions = xgb_model.predict_proba(xval)[:, 1]

# Calcular precision y recall
precision, recall, thresholds = precision_recall_curve(yval, prob_predictions)

# Calcular el valor base de "No Skill" (proporción de la clase positiva)
no_skill = len(yval[yval == 1]) / len(yval)

# Calcular el área bajo la curva (AUC) de Precision-Recall
pr_auc = auc(recall, precision)

# Gráfica de la curva Precision-Recall
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Sin habilidades (Baseline)')
plt.plot(recall, precision, marker='.', label=f'Modelo XGBoost (AUC = {pr_auc:.2f})')

# Etiquetas de los ejes
plt.xlabel('Recall (Tasa de Verdaderos Positivos)')
plt.ylabel('Precision (Precisión)')
plt.title('Curva Precision-Recall - Modelo XGBoost Optimizado')
plt.legend()
plt.grid()
plt.show()


# La **Curva Precision-Recall** del modelo **XGBoost optimizado** refuerza la solidez del modelo en el manejo del desequilibrio de clases, con un área bajo la curva (AUC) de **0.98**.
# 
# - **Precision (Precisión):** Muestra qué proporción de las predicciones positivas son correctas.
# - **Recall (Sensibilidad):** Representa qué proporción de los casos reales positivos son correctamente identificados.
# 
# #### **Interpretación:**
# 1. **Desempeño General:** El modelo mantiene altos niveles de precisión y recall a través de diferentes umbrales, reflejando un balance robusto.
# 2. **Comparación con la Línea Base (No Skill):** El desempeño del modelo supera significativamente el valor base (línea de puntos), que representa la proporción de la clase positiva en el conjunto de datos balanceado.
# 3. **Área Bajo la Curva (AUC):** Un valor de **0.98** indica que el modelo logra capturar de manera eficiente los positivos reales con un mínimo de falsos positivos.
# 
# #### **Conclusión:**
# Esta gráfica confirma la efectividad del modelo para clasificar correctamente la clase minoritaria (`1`), manteniendo una alta precisión y recall, lo cual es crucial en problemas con desequilibrio de clases. Este análisis, combinado con la curva ROC, valida la robustez del modelo optimizado para su implementación en producción.

# In[161]:


import numpy as np
import matplotlib.pyplot as plt

# Asegurarnos de que las probabilidades estén en el formato correcto
if len(prob_predictions.shape) == 1:  # Si es unidimensional
    prob_positive = prob_predictions  # Usar directamente
else:  # Si es bidimensional
    prob_positive = prob_predictions[:, 1]  # Usar la probabilidad de la clase positiva

# Verificar la dimensión de yval
yval = np.array(yval)

# Función para calcular ganancia acumulativa
def cumulative_gain_curve(y_true, y_probs):
    # Ordenar por probabilidades
    sorted_indices = np.argsort(-y_probs)
    y_true_sorted = y_true[sorted_indices]

    # Calcular el número total de positivos
    total_positives = np.sum(y_true)

    # Inicializar los valores de ganancia
    gains = np.cumsum(y_true_sorted) / total_positives
    percentages = np.arange(1, len(y_true) + 1) / len(y_true)

    return percentages, gains

# Calcular la curva de ganancia acumulativa
percentages, gains = cumulative_gain_curve(yval, prob_positive)

# Graficar la curva de ganancia acumulativa
plt.figure(figsize=(8, 6))
plt.plot(percentages, gains, label="Modelo XGBoost")
plt.plot([0, 1], [0, 1], linestyle='--', label="Sin habilidades (Baseline)")
plt.title("Curva de Ganancia Acumulativa")
plt.xlabel("Porcentaje de Muestras")
plt.ylabel("Ganancia Acumulativa")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# La **Curva de Ganancia Acumulativa** evalúa la capacidad del modelo para identificar correctamente los casos positivos conforme se incrementa el porcentaje de muestras analizadas.
# 
# - **Modelo XGBoost:** La curva azul muestra que el modelo captura una gran proporción de los casos positivos (ganancia acumulativa) analizando un porcentaje reducido de muestras. Por ejemplo, con solo el 40% de las muestras, el modelo logra capturar cerca del 80% de los casos positivos.
# - **Línea Base (Sin Habilidades):** Representada por la línea punteada, refleja un modelo aleatorio que captura los casos positivos de manera proporcional al porcentaje de muestras analizadas.

# In[162]:


import numpy as np
import matplotlib.pyplot as plt

# Ordenar las probabilidades y los valores reales
sorted_indices = np.argsort(-prob_predictions)  # Indices ordenados por probabilidad descendente
yval_sorted = np.array(yval)[sorted_indices]
prob_predictions_sorted = np.array(prob_predictions)[sorted_indices]

# Dividir en deciles
n = len(yval_sorted)
decile_size = n // 10  # Tamaño de cada decil
deciles = [np.sum(yval_sorted[:i * decile_size]) for i in range(1, 11)]  # Suma acumulativa en cada decil
deciles = np.array(deciles) / np.sum(yval_sorted)  # Proporción acumulativa de positivos

# Calcular Lift
baseline = np.linspace(0, 1, 10)  # Línea base (sin habilidades)
lift = deciles / baseline  # Lift en cada decil

# Graficar la curva Lift
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0.1, 1.0, 10), lift, marker='o', label='Modelo XGBoost')
plt.axhline(1, color='r', linestyle='--', label='Sin habilidades (Lift = 1)')
plt.title('Curva Lift - Modelo XGBoost')
plt.xlabel('Proporción de Muestras')
plt.ylabel('Lift')
plt.legend()
plt.grid()
plt.show()


# La **Curva Lift** evalúa la capacidad del modelo para identificar correctamente los casos positivos en comparación con una selección aleatoria (línea base, Lift=1).
# 
# - **Modelo XGBoost:** La curva muestra que el modelo tiene un **Lift superior a 3.5** en el primer decil, indicando que es más de 3.5 veces mejor que una selección aleatoria para capturar casos positivos en el 10% de las muestras más probables.
# - A medida que se analizan más muestras (hacia el 100%), el **Lift** disminuye y converge a 1, ya que en este punto se consideran todas las muestras, igualando el comportamiento de una selección aleatoria.
# 

# In[ ]:


from sklearn.metrics import accuracy_score

# Generar predicciones para entrenamiento y validación
y_train_pred = xgb_model.predict(xtrain)
y_val_pred = xgb_model.predict(xval)

# Calcular precisión para entrenamiento y validación
train_accuracy = accuracy_score(ytrain, y_train_pred)
val_accuracy = accuracy_score(yval, y_val_pred)

print(f"Precisión en Entrenamiento: {train_accuracy:.4f}")
print(f"Precisión en Validación: {val_accuracy:.4f}")

# Verificar si hay signos de overfitting
if train_accuracy - val_accuracy > 0.1:  
    print("Posible overfitting detectado. Considera ajustar el modelo.")
else:
    print("No hay signos claros de overfitting.")


# n este bloque hemos evaluado la precisión del modelo **XGBoost optimizado** tanto en el conjunto de entrenamiento como en el de validación para detectar posibles signos de **overfitting**. 
# 
# - **Resultados:**
#   - **Precisión en Entrenamiento:** 94.05%, indicando que el modelo predice correctamente la mayoría de las instancias en los datos de entrenamiento.
#   - **Precisión en Validación:** 93.52%, lo cual es muy cercano a la precisión del entrenamiento.
#   - La diferencia entre ambas precisiones es menor al umbral de **0.1**, lo que sugiere que el modelo generaliza bien a datos no vistos.
# 
# - **Conclusión:**
#   No se detectaron signos claros de overfitting, lo que indica que el modelo tiene un buen equilibrio entre el ajuste a los datos de entrenamiento y su capacidad para generalizar a datos nuevos. Esto confirma que la configuración del modelo, incluyendo los hiperparámetros optimizados, es adecuada para el problema en cuestión.

# # Explicabilidad

# In[98]:


# Instalar SHAP si no está instalado
get_ipython().system('pip install shap')

# Importar la librería
import shap


# In[163]:


# Crear explicador SHAP directamente para el modelo XGBoost del pipeline
explainer = shap.TreeExplainer(xgb_model)


# In[ ]:


# Obtener el preprocesador del pipeline del modelo XGBoost
preprocessor = xgb_pipeline.named_steps["preprocessor"]

# Transformar los datos con el preprocesador 
X_preprocessed = preprocessor.transform(xval)

# Verificar las dimensiones después del preprocesado
print(f"Dimensiones después del preprocesado: {X_preprocessed.shape}")


# In[165]:


import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split



# Dividir el conjunto en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamos el modelo XGBoost optimizado
xgb_model = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

# Entrenamos el modelo
xgb_model.fit(X_train, y_train)

# Generamos los valores SHAP para el modelo entrenado
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Generamos el gráfico de resumen (summary_plot) que muestra la importancia de las características
shap.summary_plot(shap_values, features=X_test, feature_names=X.columns, max_display=10)

# Mostrar el gráfico
plt.show()


# ### Explicación del gráfico SHAP
# 
# Este gráfico de resumen muestra la **importancia y el impacto de las características** en las predicciones realizadas por el modelo **XGBoost optimizado**, utilizando los valores SHAP (SHapley Additive exPlanations). Los valores SHAP son una herramienta poderosa para la interpretabilidad de modelos complejos, ya que permiten entender cómo cada característica influye en las predicciones.
# 
# #### **Ejes del gráfico:**
# - **Eje vertical:** Lista de las características más relevantes del modelo, ordenadas de mayor a menor importancia.
# - **Eje horizontal:** Valores SHAP, que representan el impacto de cada característica en la salida del modelo. 
#   - Los valores positivos (a la derecha) indican que la característica aumenta la probabilidad de una predicción positiva (clase `1`).
#   - Los valores negativos (a la izquierda) indican que la característica reduce la probabilidad de una predicción positiva.
# 
# #### **Colores:**
# - Los colores representan los valores de las características:
#   - **Rojo:** Valores altos de la característica.
#   - **Azul:** Valores bajos de la característica.
# 
# #### **Interpretación de los patrones:**
# 1. **EXT_SOURCE_2 y EXT_SOURCE_3:**
#    - Estas características tienen los mayores valores SHAP, lo que confirma su relevancia en el modelo.
#    - Valores altos (en rojo) tienden a aumentar la probabilidad de una predicción positiva, mientras que valores bajos (en azul) la disminuyen.
# 
# 2. **DAYS_EMPLOYED_YEARS:**
#    - Valores altos (empleo prolongado) tienen un impacto negativo en las predicciones (reduciendo la probabilidad de clase positiva).
# 
# 3. **FLAG_DOCUMENT_3:**
#    - Parece tener un impacto moderado y más balanceado, contribuyendo de manera variable a las predicciones positivas y negativas.
# 
# 4. **REGION_RATING_CLIENT_W_CITY:**
#    - Valores altos o bajos de esta característica también afectan significativamente las predicciones, con un patrón más disperso que refleja variabilidad en su impacto.
# 
# #### **Conclusión:**
# Este gráfico destaca qué características son más importantes y cómo sus valores afectan las predicciones del modelo. Por ejemplo:
# - **EXT_SOURCE_2 y EXT_SOURCE_3** son las características más influyentes.
# - Los valores SHAP ofrecen una explicación clara sobre el impacto positivo o negativo de cada característica, ayudando a interpretar cómo el modelo toma decisiones.
# 

# # Conclusiones Finales del Trabajo
# 
# ## Desempeño General del Modelo
# El modelo principal elegido, basado en **XGBoost**, demuestra un desempeño razonable en términos de precisión general y capacidad predictiva. Sin embargo, se observó una disparidad significativa en las métricas de precisión y recall entre las clases 0 y 1, esto indica que el modelo predice mejor una clase que la otra.
# 
# ## Evaluación de Métricas por Clase
# - **Clase 0 (mayoritaria):**  
#   El modelo alcanza altos niveles de precisión y recall, lo que demuestra su capacidad para identificar correctamente la mayoría de las instancias de esta clase.
# - **Clase 1 (minoritaria):**  
#   A pesar de los ajustes realizados con **SMOTE** y la optimización de hiperparámetros, el recall es más bajo. Esto indica que el modelo sigue teniendo dificultades para identificar correctamente las instancias positivas.
# 
# ## Curva ROC y AUC
# - La curva ROC generada muestra una separación clara entre las clases, confirmando que el modelo tiene un buen desempeño general para discriminar entre ambas.
# 
# ## Uso de Técnicas de Balanceo de Datos
# - La implementación de **SMOTE** ayudó a mitigar en cierta medida el problema de desbalanceo de clases, pero no logró resolverlo por completo.
# - Esto puede deberse a una representación insuficiente de características clave en los datos sintéticos generados.
# 
# ## Evaluación de Sobreajuste
# - Los análisis realizados muestran que el modelo **no presenta signos claros de sobreajuste**.  
#   Las métricas entre los conjuntos de entrenamiento y validación son consistentes, lo que indica que el modelo generaliza adecuadamente.
# 
# ## Aspectos de Explicabilidad
# - La utilización de **SHAP** permitió identificar las variables más influyentes en las predicciones del modelo, lo que agrega valor en términos de interpretabilidad.
# - Las características como **EXT_SOURCE_2** y **EXT_SOURCE_3** fueron especialmente relevantes, lo que sugiere su importancia en el conjunto de datos.
# 
# 
# 
