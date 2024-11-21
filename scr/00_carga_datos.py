#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import openpyxl


# In[3]:


import os
print("Directorio actual:", os.getcwd())


# In[4]:


os.chdir('/Users/admin/Desktop/ap/practica1/data')


# In[5]:


# Cargar los archivos de datos
# Ruta relativa a la carpeta de datos
data = pd.read_csv('application_data.csv', delimiter=',')



# In[21]:


# Cargar el diccionario de variables
dictionary = pd.read_excel('columns_description.xlsx', engine='openpyxl')
print(dictionary.head())


# In[20]:


#Mostrar las primeras filas para verificar
print(data.head())


# In[22]:


# Dimensiones del dataset
print("Dimensiones del dataset:")
print(data.shape)

# Tipos de datos
print("\nTipos de datos de las columnas:")
print(data.dtypes)

# Valores nulos
print("\nCantidad de valores nulos por columna:")
print(data.isnull().sum())


# #### 1. Dimesiones del data set
# - El dataset contiene 307,511 filas (observaciones o registros) y 122 columnas (variables).
# - Cada fila representa una persona o cliente, y las columnas son las características asociadas a cada uno.
# 
# #### 2. Tipos de datos
# - int64: Variables numéricas enteras. Por ejemplo, SK_ID_CURR parece ser un identificador único de cliente y TARGET es el objetivo que estamos tratando de analizar.
# - object: Variables categóricas o texto. Por ejemplo, NAME_CONTRACT_TYPE podría indicar el tipo de contrato y CODE_GENDER el género.
# - float64 (mostrado más adelante): Variables numéricas decimales, como las solicitudes de crédito (AMT_REQ_CREDIT_BUREAU_DAY).
# 
# #### 3. Valores nulos
# - Columnas con 0 valores nulos: Estas variables están completas, es decir, no tienen datos faltantes.
#     - Ejemplo: SK_ID_CURR, TARGET, CODE_GENDER no tienen valores nulos.
# - Columnas con valores nulos (como AMT_REQ_CREDIT_BUREAU_YEAR):
#     - Esta columna tiene 41,519 valores nulos, lo que representa una proporción significativa del total.

# In[24]:


# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(data.describe())


# In[25]:


# Relación entre las columnas del dataset y el diccionario
print("\nRelación entre columnas del dataset y diccionario de variables:")
print(dictionary[['Row', 'Description']])


# In[26]:


# Porcentaje de valores nulos por columna
missing_values = (data.isnull().sum() / len(data)) * 100
missing_values = missing_values[missing_values > 0]  # Filtrar columnas con valores nulos
missing_values = missing_values.sort_values(ascending=False)

print("Porcentaje de valores nulos por columna:")
print(missing_values)


# In[9]:


import plotly
print("Plotly está instalado correctamente. Versión:", plotly.__version__)


# In[35]:




# Calcular proporción de cada clase en TARGET
target_distribution = data['TARGET'].value_counts(normalize=True).mul(100).reset_index()
target_distribution.columns = ['TARGET', 'percent']

# Calcular el conteo absoluto de cada clase
target_counts = data['TARGET'].value_counts().reset_index()
target_counts.columns = ['TARGET', 'count']

# Mostrar la distribución
print("Distribución de TARGET (proporción y conteo):")
print(pd.concat([target_distribution, target_counts['count']], axis=1))

# Graficar distribución de TARGET
fig = px.bar(target_distribution, x="TARGET", y="percent", title="Distribución de la variable TARGET", text="percent")
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(xaxis_title="TARGET", yaxis_title="Porcentaje")
fig.show()


# In[10]:



# Proporción y conteo de clases en TARGET
target_distribution = data['TARGET'].value_counts(normalize=True).mul(100).rename('percent')
target_count = data['TARGET'].value_counts()

# Crear un DataFrame para organizar los resultados
target_analysis = pd.DataFrame({
    'TARGET': target_count.index,
    'count': target_count.values,
    'percent': target_distribution.values
})

print("Distribución de TARGET:")
print(target_analysis)

# Gráfico de barras de la distribución de TARGET con Matplotlib
plt.figure(figsize=(8, 6))
sns.barplot(x=target_analysis['TARGET'], y=target_analysis['percent'], palette="viridis")
plt.title("Distribución de la variable TARGET")
plt.xlabel("TARGET")
plt.ylabel("Porcentaje (%)")
plt.xticks([0, 1], ['No incumple (0)', 'Incumple (1)'])
plt.show()

# Gráfico interactivo con Plotly
fig = px.bar(
    target_analysis, x='TARGET', y='percent',
    title="Distribución de la variable TARGET",
    text='percent'
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No incumple (0)', 'Incumple (1)']))
fig.show()


# La gráfica muestra la distribución de la variable objetivo TARGET, donde se observa que el 91.93% de los clientes no incumplen con sus pagos (TARGET = 0), mientras que solo el 8.07% presentan incumplimiento (TARGET = 1). 

# # Selección de threshold por filas y columnas para eliminar valores missing

# In[11]:


# Calcular valores nulos por columnas y filas
null_columns = data.isnull().sum().sort_values(ascending=False)  # Valores nulos por columna
null_rows = data.isnull().sum(axis=1).sort_values(ascending=False)  # Valores nulos por fila

# Mostrar las dimensiones
print(f"Dimensiones de valores nulos por columnas: {null_columns.shape}")
print(f"Dimensiones de valores nulos por filas: {null_rows.shape}")

# Crear DataFrames para análisis
null_columns_df = pd.DataFrame(null_columns, columns=['nulos_columnas'])
null_rows_df = pd.DataFrame(null_rows, columns=['nulos_filas'])

# Agregar porcentaje de nulos
null_columns_df['porcentaje_columnas'] = null_columns_df['nulos_columnas'] / data.shape[0]
null_rows_df['porcentaje_filas'] = null_rows_df['nulos_filas'] / data.shape[1]

# Si deseas agregar información sobre TARGET a las filas:
null_rows_df['TARGET'] = data['TARGET'].copy()

# Mostrar resultados
print("Valores nulos por columnas (ordenado):")
print(null_columns_df)

print("\nValores nulos por filas (ordenado):")
print(null_rows_df)


# **Selección de threshold por filas y columnas para valores nulos**
# 
# El análisis de los valores nulos revela lo siguiente:
# 
# __Por columnas:__
# 
# Existen 122 columnas en el dataset, de las cuales varias tienen un porcentaje significativo de valores nulos.
# 
# Las columnas COMMONAREA_MED, COMMONAREA_AVG y COMMONAREA_MODE presentan cerca del 69.87% de valores faltantes, lo que las hace candidatas para ser eliminadas o imputadas según su relevancia en el análisis.
# 
# Algunas columnas, como NAME_HOUSING_TYPE y NAME_EDUCATION_TYPE, no presentan valores nulos y están completas.
# Por filas:
# 
# Hay 307,511 filas en el dataset, y un número considerable tiene hasta un 50% de sus valores faltantes.
# 
# Esto podría indicar la necesidad de aplicar un threshold para eliminar estas filas, especialmente si contienen poca información relevante.
# Relevancia de TARGET en las filas con nulos:
# 
# Se incluyó la variable TARGET para observar cómo se distribuyen los valores faltantes en relación con la variable objetivo. Esto permitirá evaluar si las filas con más nulos tienen una relación significativa con el incumplimiento (TARGET = 1).
# 

# In[21]:


# Calcular el porcentaje de valores faltantes
missing = pd.DataFrame((data.isnull().sum()) * 100 / data.shape[0]).reset_index()
missing.columns = ['Column', 'Missing_Percentage']

# Configurar el gráfico con estilo mejorado
plt.figure(figsize=(18, 6))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x='Column', y='Missing_Percentage', data=missing, palette='viridis')

# Personalizar etiquetas y diseño
plt.xticks(rotation=90, fontsize=8)
plt.title("Porcentaje de Valores Faltantes en las Variables", fontsize=14, fontweight='bold')
plt.ylabel("Porcentaje de Valores Faltantes", fontsize=12)
plt.xlabel("Columnas", fontsize=12)

# Añadir etiquetas encima de las barras
for p in ax.patches:
    percentage = f"{p.get_height():.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 5), textcoords='offset points')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# El gráfico muestra el porcentaje de valores faltantes en cada una de las variables del dataset. En el eje horizontal se encuentran los nombres de las columnas, mientras que el eje vertical indica el porcentaje de valores faltantes. Las barras representan el nivel de incompletitud en cada columna, destacando que algunas variables tienen una cantidad significativa de valores nulos, llegando incluso al 70%, lo que podría influir en el análisis y el modelo.

# In[14]:


# Threshold (umbral) para eliminar columnas y filas
threshold_columnas = 0.50  # Eliminar columnas con más del 50% de valores nulos
threshold_filas = 0.50     # Eliminar filas con más del 50% de valores nulos

# Identificar columnas que superan el threshold
columnas_a_eliminar = null_columns_df[null_columns_df['porcentaje_columnas'] > threshold_columnas]
print(f"Columnas con más del {threshold_columnas*100}% de valores nulos:")
print(columnas_a_eliminar)

# Identificar filas que superan el threshold
filas_a_eliminar = null_rows_df[null_rows_df['porcentaje_filas'] > threshold_filas]
print(f"\nFilas con más del {threshold_filas*100}% de valores nulos:")
print(filas_a_eliminar)

# Eliminar columnas y filas según el threshold
data_cleaned = data.drop(columns=columnas_a_eliminar.index)  # Eliminar columnas
data_cleaned = data_cleaned.drop(filas_a_eliminar.index, axis=0)  # Eliminar filas

# Mostrar dimensiones después de limpiar
print("\nDimensiones originales del dataset:", data.shape)
print("Dimensiones después de limpiar:", data_cleaned.shape)


# Este paso se realiza para mejorar la calidad del dataset eliminando columnas y filas con demasiados valores faltantes (más del 50%). Esto simplifica el dataset, evita el sesgo de datos incompletos y asegura que las variables y observaciones restantes sean más útiles para el análisis y modelado.

# In[16]:


# Crear un DataFrame para las variables con más del 50% de valores faltantes
variables_mas_nulos = columnas_a_eliminar.index
missing_data_treatment = pd.DataFrame({
    "Variable": variables_mas_nulos,
    "Porcentaje_Faltante": null_columns_df.loc[variables_mas_nulos, 'porcentaje_columnas']
}).reset_index(drop=True)

# Gráfico para visualizar las variables con más del 50% de valores faltantes
plt.figure(figsize=(12, 6))
sns.barplot(
    x="Variable", 
    y="Porcentaje_Faltante", 
    data=missing_data_treatment.sort_values(by="Porcentaje_Faltante", ascending=False),
    palette="magma"
)
plt.title("Variables con más del 50% de Valores Faltantes (Porcentaje)")
plt.ylabel("Porcentaje de Valores Faltantes")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# **Eliminación de valores nulos según threshold**
# 
# Se aplicó un umbral del 50% tanto para columnas como para filas con valores nulos, obteniendo los siguientes resultados:
# 
# __Columnas eliminadas:__
# 
# Un total de 41 columnas fueron eliminadas del dataset, ya que más del 50% de sus valores eran nulos.
# 
# Entre las columnas eliminadas se encuentran COMMONAREA_MED, COMMONAREA_AVG y NONLIVINGAPARTMENTS_MODE, que presentaban un ~69% de valores faltantes. Estas variables probablemente contenían poca información útil debido a la gran cantidad de datos ausentes.
# Filas eliminadas:
# 
# No se eliminó ninguna fila, ya que ninguna fila superó el umbral del 50% de valores faltantes.
# Dimensiones del dataset:
# 
# Las dimensiones originales del dataset eran (307,511 filas x 122 columnas).
# Después de la limpieza, el dataset quedó reducido a (307,511 filas x 81 columnas), eliminando únicamente las columnas con demasiados valores nulos.
# 
# 

# ## __Identificar variables categóricas y numéricas__ ##

# In[37]:


# Identificar variables categóricas y numéricas
categorical_vars = data_cleaned.select_dtypes(include=['object']).columns.tolist()
numerical_vars = data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Variables categóricas:")
print(categorical_vars)

print("\nVariables numéricas:")
print(numerical_vars)


# ## __Convertir variables categóricas a tipo category__ ##

# In[38]:


# Convertir variables categóricas a tipo 'category'
data_cleaned[categorical_vars] = data_cleaned[categorical_vars].astype('category')

# Verificar los tipos de datos después de la conversión
print("\nTipos de datos después de la conversión:")
print(data_cleaned[categorical_vars].dtypes)


# ## __Ver valores únicos en variables categóricas__ ##

# In[39]:


# Ver valores únicos de cada variable categórica
for var in categorical_vars:
    print(f"\nValores únicos en la variable '{var}':")
    print(data_cleaned[var].value_counts())


# ## __Tipos: Variables categóricas y numéricas__ ##

# In[40]:


def plot_features(feature, label_rotation=False, horizontal_layout=True, figsize=(12, 12), font_scale=0.8, horizontal_bars=False, top_n=None, label_fontsize=8):
    # Estilo general
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=font_scale)
    
    temp = data[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'valores absolutos': temp.values})
    
    t1_percentage = data[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    t1_percentage.sort_values(by='TARGET', ascending=False, inplace=True)
    
    # Si top_n está definido, selecciona las principales categorías
    if top_n:
        df1 = df1.head(top_n)
        t1_percentage = t1_percentage.head(top_n)
    
    if horizontal_bars:
        figsize = (figsize[1], figsize[0])
    
    if horizontal_layout:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
    
    # Colores
    custom_palette1 = sns.color_palette("pastel")
    custom_palette2 = sns.color_palette("coolwarm", len(t1_percentage))
    
    # Gráfico 1: Valores absolutos
    if horizontal_bars:
        s1 = sns.barplot(
            ax=ax1, 
            y=feature, 
            x="valores absolutos", 
            data=df1, 
            palette=custom_palette1
        )
        ax1.set_ylabel(feature, fontsize=label_fontsize)
        ax1.set_xlabel("Frecuencia Absoluta", fontsize=label_fontsize)
        ax1.tick_params(axis='y', labelsize=label_fontsize)
    else:
        s1 = sns.barplot(
            ax=ax1, 
            x=feature, 
            y="valores absolutos", 
            data=df1, 
            palette=custom_palette1
        )
        ax1.set_xlabel(feature, fontsize=label_fontsize)
        ax1.set_ylabel("Frecuencia Absoluta", fontsize=label_fontsize)
        ax1.tick_params(axis='x', labelsize=label_fontsize)
    
    ax1.set_title(f"Distribución de {feature}", fontsize=label_fontsize)
    if label_rotation:
        s1.set_xticklabels(s1.get_xticklabels(), rotation=90, fontsize=label_fontsize)

    # Gráfico 2: Porcentaje de 'TARGET'
    if horizontal_bars:
        s2 = sns.barplot(
            ax=ax2, 
            y=feature, 
            x="TARGET", 
            order=t1_percentage[feature], 
            data=t1_percentage,
            palette=custom_palette2
        )
        ax2.set_ylabel(feature, fontsize=label_fontsize)
        ax2.set_xlabel("% No Devueltos", fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelsize=label_fontsize)
    else:
        s2 = sns.barplot(
            ax=ax2, 
            x=feature, 
            y="TARGET", 
            order=t1_percentage[feature], 
            data=t1_percentage, 
            palette=custom_palette2
        )
        ax2.set_xlabel(feature, fontsize=label_fontsize)
        ax2.set_ylabel("% No Devueltos", fontsize=label_fontsize)
        ax2.tick_params(axis='x', labelsize=label_fontsize)
    
    ax2.set_title(f"% No Devueltos por {feature}", fontsize=label_fontsize)
    if label_rotation:
        s2.set_xticklabels(s2.get_xticklabels(), rotation=90, fontsize=label_fontsize)

    # Ajustes finales
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


# In[41]:


plot_features('ORGANIZATION_TYPE', label_rotation=False, horizontal_layout=False, figsize=(16, 20), horizontal_bars=True)


# In[42]:


plot_features('ORGANIZATION_TYPE', label_rotation=False, horizontal_layout=True, figsize=(12, 10), top_n=10)


# In[44]:


plot_features('ORGANIZATION_TYPE',True,True)


# In[45]:


plot_features('NAME_FAMILY_STATUS',True,True)


# In[46]:


plot_features('NAME_INCOME_TYPE',False,False)


# In[47]:


plot_features('OCCUPATION_TYPE',True,False)


# In[48]:


plot_features('NAME_EDUCATION_TYPE',True,True)


# In[49]:


plot_features('NAME_HOUSING_TYPE',True,True)


# In[50]:


plot_features('CODE_GENDER',True,True)


# ### *Explicaciones de cada gráfico para incluir en tu informe*
# 
# #### *Gráfico 1: Distribución de ORGANIZATION_TYPE*
# El gráfico presenta la cantidad de préstamos solicitados por personas en diferentes tipos de organizaciones:
# 
# - Las categorías más frecuentes son “Family Type 3” y “XNA”, con una cantidad significativamente alta de solicitudes.
# 
# - Tipos de organización como “Transport Type 3” y algunas industrias específicas presentan mayores tasas de incumplimiento.
# 
# - Esta variable es crucial, ya que la organización donde trabaja un cliente puede influir en la estabilidad de sus ingresos y, por ende, en su capacidad de pago.
# 
# ---
# 
# #### *Gráfico 2: Distribución de NAME_FAMILY_STATUS*
# Este gráfico analiza los préstamos según el estado civil de los clientes:
# 
# - Los clientes casados representan la mayoría (~62%), seguidos de solteros/no casados (~20%).
# 
# - Los clientes solteros y en matrimonios civiles muestran mayores tasas de incumplimiento en comparación con los casados.
# 
# - El estado familiar puede afectar la carga financiera de un cliente y su capacidad para pagar, lo que lo convierte en una variable relevante para el análisis de riesgo crediticio.
# 
# ---
# 
# #### *Gráfico 3: Distribución de NAME_INCOME_TYPE*
# El gráfico ilustra la cantidad de préstamos solicitados según la fuente de ingresos de los clientes:
# 
# - La mayoría de los préstamos provienen de personas empleadas (~60%) y comerciales asociados (~23%).
# 
# - Los clientes desempleados y en baja por maternidad tienen las mayores tasas de incumplimiento.
# 
# - La fuente de ingresos refleja la estabilidad financiera de los clientes, lo cual es un factor crítico en la evaluación del riesgo crediticio.
# ---
# 
# #### *Gráfico 4: Distribución de OCCUPATION_TYPE*
# Este gráfico muestra la cantidad de préstamos según la ocupación de los clientes:
# 
# - Los “Laborers” representan el grupo con mayor número de solicitudes, seguidos de “Sales staff”.
# 
# - Las ocupaciones de baja cualificación, como “Low-skill laborers”, tienen las mayores tasas de incumplimiento.
# 
# - La ocupación indica el nivel de ingresos y la estabilidad laboral, lo que influye directamente en la capacidad de pago.
# 
# ---
# 
# #### *Gráfico 5: Distribución de NAME_EDUCATION_TYPE*
# La gráfica representa los niveles educativos de los clientes:
# 
# - La mayoría de los clientes tiene educación secundaria o secundaria especial (~71%), seguida de educación superior (~24%).
# 
# - Los niveles educativos más bajos, como educación secundaria incompleta, presentan las mayores tasas de incumplimiento.
# 
# - El nivel educativo está correlacionado con el acceso a mejores oportunidades laborales y financieras, lo que lo convierte en un indicador relevante de riesgo.
# 
# ---
# 
# #### *Gráfico 6: Distribución de NAME_HOUSING_TYPE*
# El gráfico muestra la cantidad de préstamos según el tipo de vivienda:
# 
# - La mayoría de los clientes vive en casa o apartamento propio (~88%), seguido de los que viven con sus padres o en apartamentos alquilados.
# 
# - Las personas que viven en apartamentos alquilados presentan las mayores tasas de incumplimiento.
# 
# - El tipo de vivienda puede ser un reflejo de la estabilidad financiera de los clientes y de su capacidad para asumir compromisos financieros a largo plazo.
# 
# ---
# 
# #### *Gráfico 7: Distribución de CODE_GENDER*
# El gráfico muestra la distribución del género (CODE_GENDER) de los clientes:
# 
# - La mayoría de los clientes son mujeres (~65%), mientras que los hombres representan aproximadamente el ~35%.
# 
# - Hay un número insignificante de valores registrados como XNA (menos del 0.001%), lo cual podría representar errores de registro o valores irrelevantes.
# 
# - Los hombres tienen una mayor tasa de incumplimiento en los préstamos comparado con las mujeres.
# 
# - Esta variable es relevante porque puede revelar diferencias significativas en el comportamiento crediticio entre géneros, lo cual puede influir en los resultados del modelo predictivo y las estrategias de mitigación de riesgos.
# 
# ---
# 
# ### *Explicación: ¿Por qué se eligieron estas 4 variables?*
# 
# 1. *Relación directa con el incumplimiento:* Cada variable seleccionada tiene un impacto potencial en la capacidad del cliente para cumplir con los pagos del préstamo.
# 
# 2. *Segmentación efectiva:* Los gráficos permiten identificar grupos específicos con mayor riesgo de incumplimiento.
# 
# 3. *Relevancia sociodemográfica y económica:* Factores como género, ocupación, nivel educativo y tipo de vivienda son determinantes clave en el análisis de crédito.
# 
# 4. *Optimización del modelo predictivo:* Incluir estas variables en el análisis ayuda a construir un modelo más robusto que puede predecir mejor el comportamiento de los clientes.
# 
# 
# ---
# 
# 

# A continuación, calculamos el porcentaje de desbalance, ya que la mayoría corresponde a la clase target0 y la minoría a la clase target1

# In[51]:


# Calculamos el conteo de cada categoría en la variable TARGET
target_counts = data['TARGET'].value_counts()

# Calculamos el porcentaje de desbalance
imbalance_percentage = (target_counts.min() / target_counts.max()) * 100
print(f"Imbalance Percentage: {imbalance_percentage:.2f}%")

# Graficamos el conteo de la variable TARGET
target_counts.plot(kind='barh', figsize=(8, 6), color=['skyblue', 'salmon'])
plt.xlabel("Count", labelpad=14, fontsize=12)
plt.ylabel("Target Variable", labelpad=14, fontsize=12)
plt.title("Count of TARGET Variable per category", y=1.02, fontsize=14)
plt.show()


# In[ ]:


# Función para filtrar y graficar

def plot_by_target(df, col, target_value, hue=None, log_scale=False, rotation=45, palette='bright'):
    # Filtrar el dataframe para el valor de TARGET deseado
    filtered_df = df[df['TARGET'] == target_value]
    
    # Título dinámico según el valor de TARGET
    title = f"Distribution of {col} - For Target: {target_value}"
    
    # Graficar con seaborn
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x=col, hue=hue, palette=palette)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.show()


# In[65]:


# Distribución de la columna 'OCCUPATION_TYPE' cuando TARGET es 0
plot_by_target(data, col='OCCUPATION_TYPE', target_value=0, hue='CODE_GENDER', log_scale=False)

# Distribución de la columna 'OCCUPATION_TYPE' cuando TARGET es 1
plot_by_target(data, col='OCCUPATION_TYPE', target_value=1, hue='CODE_GENDER', log_scale=False)


# # Análisis de Género vs Tipo de Ocupación en relación con la Variable Target
# 
# ### Puntos clave:
# 
# 1. **Mujeres:**
#    - Las mujeres que trabajan como **contables**, personal de servicio privado, personal de cocina, personal de recursos humanos, entre otros, tienen una mayor tasa de incumplimiento (**default**) en comparación con los hombres en estas mismas ocupaciones.
# 
# 2. **Hombres:**
#    - Los hombres que trabajan en ocupaciones como **mano de obra de baja calificación** o **conductores** tienen una mayor tasa de incumplimiento en comparación con las mujeres en estas mismas ocupaciones.
# 
# ### Conclusiones:
# 
# - **Conclusión 1:**
#    - Las mujeres en ocupaciones como:
#      - **Contables**
#      - **Personal de servicio privado**
#      - **Secretarias**
#      - **Agentes inmobiliarios**
#      - **Recursos humanos**
#    - Son las que más incumplen en comparación con los hombres en estas profesiones.
# 
# - **Conclusión 2:**
#    - Los hombres en ocupaciones como:
#      - **Mano de obra de baja calificación**
#      - **Conductores**
#    - Destacan como los que más incumplen en comparación con las mujeres en estos roles.
# 

# In[67]:


# Filtrar el dataset para TARGET=0
target0_df = data[data['TARGET'] == 0]

# Crear el gráfico usando seaborn
plt.figure(figsize=(10, 6))
sns.countplot(data=target0_df, x='FLAG_OWN_CAR', hue='CODE_GENDER', palette='bright')
plt.title('Distribution of Car Owner Flag – For Target: 0')
plt.xticks(rotation=45)
plt.show()


# In[69]:


# Filtrar el dataset para TARGET=1
target1_df = data[data['TARGET'] == 1]

# Crear el gráfico usando seaborn
plt.figure(figsize=(10, 6))
sns.countplot(data=target1_df, x='FLAG_OWN_CAR', hue='CODE_GENDER', palette='bright')
plt.title('Distribution of Car Owner Flag – For Target: 1')
plt.xticks(rotation=45)
plt.show()


# ### Análisis de Propiedad de Vehículo en Relación con los Incumplimientos
# 
# - **Insight general:** En términos generales, las personas que **no poseen coche** son las que tienen mayor tasa de incumplimiento. 
# - **Insight específico:** Al profundizar en el análisis, se observa que las **mujeres sin coche** son las que más incumplen en comparación con los hombres que tampoco poseen coche.
# 

# In[71]:


# Filtrar el dataset para TARGET=0
target0_df = data[data['TARGET'] == 0]

# Crear el gráfico usando seaborn
plt.figure(figsize=(10, 6))
sns.countplot(data=target0_df, x='NAME_CONTRACT_TYPE', hue='CODE_GENDER', palette='bright')
plt.title('Distribution of Contract Type – For Target: 0')
plt.xticks(rotation=45)
plt.show()


# In[74]:


# Filtrar el dataset para TARGET=1
target1_df = data[data['TARGET'] == 1]

# Crear el gráfico usando seaborn
plt.figure(figsize=(10, 6))
sns.countplot(data=target1_df, x='NAME_CONTRACT_TYPE', hue='CODE_GENDER', palette='bright')
plt.title('Distribution of Contract Type – For Target: 1')
plt.xticks(rotation=45)
plt.show()


# ### Análisis de Género vs Tipo de Contrato en Relación con la Variable Target
# 
# 1. **Insights generales:**
#    - El tipo de contrato **'cash loans'** tiene un mayor número de créditos en comparación con el tipo de contrato **'revolving loans'**.
#    - Las **mujeres** lideran en la solicitud de créditos, independientemente del tipo de contrato.
# 
# 2. **Insight específico:**
#      - Se observa que tanto **hombres como mujeres** con contratos de tipo **'cash loans'** tienen una tasa de incumplimiento casi igual.
#      - Por otro lado, en los contratos de tipo **'revolving loans'**, aunque el número de créditos es menor, las **mujeres** son las que más incumplen en comparación con los hombres.
# 

# In[79]:



# Crear una nueva columna agrupando AMT_INCOME_TOTAL en rangos
bins = [0, 50000, 100000, 200000, 500000, 1000000, data['AMT_INCOME_TOTAL'].max()]
labels = ['<50k', '50k-100k', '100k-200k', '200k-500k', '500k-1M', '>1M']
data['AMT_INCOME_RANGE'] = pd.cut(data['AMT_INCOME_TOTAL'], bins=bins, labels=labels)

# Filtrar los datasets para TARGET=0 y TARGET=1
target0_df = data[data['TARGET'] == 0]
target1_df = data[data['TARGET'] == 1]

# Ordenar las categorías por recuento para TARGET=0
order_target0 = (
    target0_df['AMT_INCOME_RANGE']
    .value_counts()
    .sort_values(ascending=False)
    .index
)

# Crear el gráfico para TARGET=0
plt.figure(figsize=(10, 6))
sns.countplot(
    data=target0_df,
    x='AMT_INCOME_RANGE',
    hue='CODE_GENDER',
    palette='bright',
    order=order_target0
)
plt.title('Distribution of Income Range – For Target: 0')
plt.xticks(rotation=45)
plt.show()

# Ordenar las categorías por recuento para TARGET=1
order_target1 = (
    target1_df['AMT_INCOME_RANGE']
    .value_counts()
    .sort_values(ascending=False)
    .index
)

# Crear el gráfico para TARGET=1
plt.figure(figsize=(10, 6))
sns.countplot(
    data=target1_df,
    x='AMT_INCOME_RANGE',
    hue='CODE_GENDER',
    palette='bright',
    order=order_target1
)
plt.title('Distribution of Income Range – For Target: 1')
plt.xticks(rotation=45)
plt.show()


# #### Análisis de Rango de Ingresos en Relación con la Variable Target
# 
#  **Distribución para TARGET = 0**
# 
#  **Insights Generales:**
# 1. **Rangos más comunes:**
#    - El rango **100k-200k** es el más frecuente entre las personas que **no incumplen** (TARGET=0), seguido por **200k-500k** y **50k-100k**.
#    - En el rango más frecuente (**100k-200k**), las **mujeres** tienen un número significativamente mayor de registros en comparación con los hombres.
#  **Distribución para TARGET = 1**
# 
#  **Insights Generales:**
# 1. **Rangos más comunes:**
#    - Entre los incumplidores (TARGET=1), el rango más frecuente es **100k-200k**, donde las **mujeres** tienen una representación ligeramente mayor que los hombres.
#    - El siguiente rango más común es **200k-500k**, en el cual los hombres tienen un predominio claro.

# In[82]:




# Filtrar el dataset para TARGET=0 y TARGET=1
target0_df = data[data['TARGET'] == 0]
target1_df = data[data['TARGET'] == 1]

# Ordenar categorías por recuento para TARGET=0
order_target0 = (
    target0_df['NAME_INCOME_TYPE']
    .value_counts()
    .sort_values(ascending=False)
    .index
)

# Ordenar categorías por recuento para TARGET=1
order_target1 = (
    target1_df['NAME_INCOME_TYPE']
    .value_counts()
    .sort_values(ascending=False)
    .index
)

# Gráfico para TARGET=0
plt.figure(figsize=(10, 6))
sns.countplot(
    data=target0_df,
    x='NAME_INCOME_TYPE',
    hue='CODE_GENDER',
    palette='bright',
    order=order_target0
)
plt.title('Distribution of Name of Income Type – Target: 0')
plt.xticks(rotation=45)
plt.show()

# Gráfico para TARGET=1
plt.figure(figsize=(10, 6))
sns.countplot(
    data=target1_df,
    x='NAME_INCOME_TYPE',
    hue='CODE_GENDER',
    palette='bright',
    order=order_target1
)
plt.title('Distribution of Name of Income Type – Target: 1')
plt.xticks(rotation=45)
plt.show()



# ### Análisis del Tipo de Ingreso en Relación con la Variable Target
# 
#  **Insights Generales:**
# 1. Para los tipos de ingreso **‘Working’**, **‘Commercial Associate’** , **‘State Servant’** y **‘Pensioner’** el número de créditos es mayor en comparación con otros tipos como **‘Maternity Leave’**.
# 2. En estos grupos, las **mujeres** tienen un número significativamente mayor de créditos en comparación con los hombres.
# 
#  **Insights Específicos para TARGET = 1 (Incumplidores):**
# 1. No hay registros de incumplimiento para los tipos de ingreso **‘Student’** y **‘Businessman’**, lo que indica que estas personas no realizan pagos tardíos.
# 2. Los incumplimientos son más comunes en profesionales **‘Working’**, **‘Commercial Associate’** , **‘State Servant’** y **‘Pensioner’**.
# 
# **Conclusión Clave:**
# - Las **mujeres** tienen una tasa de incumplimiento  mayor en comparación con los hombres.
# - La mayoría de los incumplidores provienen de los tipos de ingreso: **‘Working’**, **‘Commercial Associate’** y **‘Pensioner’**.
# - Las personas con tipos de ingreso **‘Student’** y **‘Businessman’** no registran pagos tardíos.

# In[83]:


# Gráfico para TARGET=0
sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(15, 30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization Type for Target - 0")
plt.xticks(rotation=90)

# Aplicar orden correcto en base a conteos y sin usar escala logarítmica en el eje x
sns.countplot(
    data=target0_df,
    y='ORGANIZATION_TYPE',
    order=target0_df['ORGANIZATION_TYPE'].value_counts().index,
    palette='bright'
)

plt.xscale('linear')  # Eliminar escala logarítmica en x si no es necesaria
plt.show()

# Gráfico para TARGET=1
sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(15, 30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization Type for Target - 1")
plt.xticks(rotation=90)

# Aplicar orden correcto en base a conteos y evitar escala logarítmica en x
sns.countplot(
    data=target1_df,
    y='ORGANIZATION_TYPE',
    order=target1_df['ORGANIZATION_TYPE'].value_counts().index,
    palette='bright'
)

plt.xscale('linear')  # Escala estándar para barras categóricas
plt.show()


# #### Análisis del Tipo de Organización para Clientes que Solicitan Créditos
# 
# **Puntos Clave:**
# 1. La mayoría de los clientes que han solicitado créditos provienen de los tipos de organización:
#    - **‘Business Entity Type 3’**
#    - **‘Self Employed’**
#    - **‘Other’**
#    - **‘Medicine’**
#    - **‘Government’**
# 
# 2. Hay menos clientes provenientes de los siguientes tipos de industria:
#    - **Industry Type 8**
#    - **Industry Type 6**
#    - **Industry Type 10**
#    - **Religion**
#    - **Trade Type 5**
#    - **Trade Type 4**

# In[84]:


plt.figure(figsize=(12, 10))

# Graficar la distribución de edades para TARGET=0 (préstamos no incumplidos)
sns.kdeplot(
    data.loc[data['TARGET'] == 0, 'DAYS_BIRTH'] / -365,
    label='Clientes sin dificultades para pagar el préstamo (0)',
    shade=True
)

# Graficar la distribución de edades para TARGET=1 (préstamos incumplidos)
sns.kdeplot(
    data.loc[data['TARGET'] == 1, 'DAYS_BIRTH'] / -365,
    label='Clientes con dificultades para pagar el préstamo  (1)',
    shade=True
)

# Etiquetas y título
plt.xlabel('Edad en años', fontsize=14)
plt.ylabel('Densidad', fontsize=14)
plt.title('Distribución de Edad según Estado del Préstamo', fontsize=16)
plt.legend(fontsize=12)
plt.show()


# #### Análisis de la Relación entre Edad y Reembolso de Préstamos
# 
# 1. A medida que los clientes son **más mayores**, tienden a **tener menos dificultades** para cumplir con sus responsabilidades crediticias.
# 2. Los clientes **más jóvenes** son menos fiables en comparación con los clientes mayores.
# 

# In[85]:


# Crear rangos de edad (en años)
data['AGE'] = (data['DAYS_BIRTH'] / -365).astype(int)
age_bins = [20, 30, 40, 50, 60, 70, 80]
age_labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
data['AGE_RANGE'] = pd.cut(data['AGE'], bins=age_bins, labels=age_labels, right=False)

# Calcular el porcentaje de préstamos no pagados por rango de edad
age_data = data.groupby('AGE_RANGE')['TARGET'].mean() * 100

# Graficar
plt.figure(figsize=(10, 6))
plt.bar(age_data.index.astype(str), age_data, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Rango de Edad', fontsize=12)
plt.ylabel('% de Préstamos No Pagados', fontsize=12)
plt.title('Análisis por Edad', fontsize=14)
plt.show()


# #### Análisis del Gráfico de Edad y Tasa de Incumplimiento
# 
# - Este gráfico es consistente con el análisis anterior: 
#    - Los clientes **jóvenes** tienen una tasa de incumplimiento de aproximadamente del **12%**.
#    - Los clientes **mayores** tienen una tasa significativamente menor, de solo **5%**.

# In[86]:


def plot_numerical(feature):
    """
    Función para graficar la distribución de una variable numérica.
    :param feature: Nombre de la columna en el dataset
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Distribución de {feature}", fontsize=16)
    
    # Graficar la distribución
    sns.histplot(data[feature].dropna(), kde=True, bins=30, color='skyblue', edgecolor='black')
    
    # Etiquetas de los ejes
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.show()

# Ejemplo de uso con tu dataset

plot_features('CNT_CHILDREN')


# En un principio pensamos que el número de hijos, sería una variable significativa en el análisis de la calidad crediticia de los clientes. Pero el gráfico de la derecha, demuestra que no es así ya que, aquellos individuos que tienen mas problemas, aun siendo estos muy pocos (menos de 1%), a la hora de pagar los créditos son los individuos con 9 u 11 hijos, los cuales son una minoría (prácticamente insignificante) y no tienen un impacto directo sobre la variable objetivo.  

# In[87]:


def plot_comparisons(features):
    """
    Función para comparar la distribución de múltiples variables numéricas
    entre clientes que pagaron el préstamo (TARGET=0) y los que no lo hicieron (TARGET=1).
    
    :param features: Lista de columnas numéricas a comparar.
    """
    # Filtrar los datos según el valor de TARGET
    t_1 = data.loc[data['TARGET'] == 1]  
    t_0 = data.loc[data['TARGET'] == 0]  
    
    for feature in features:
        plt.figure(figsize=(12, 6))
        
        # Graficar la distribución para TARGET=0 y TARGET=1
        sns.kdeplot(
            t_0[feature].dropna(), 
            bw_adjust=0.5, 
            label="Clientes con dificultades para pagar el préstamo", 
            shade=True, 
            color="skyblue"
        )
        sns.kdeplot(
            t_1[feature].dropna(), 
            bw_adjust=0.5, 
            label="Clientes sin dificultades para pagar el préstamo", 
            shade=True, 
            color="salmon"
        )
        
        # Configuración del gráfico
        plt.title(f"Comparación de {feature}", fontsize=16)
        plt.ylabel('Densidad', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.show()

# Lista de características a comparar
feat = ['AMT_CREDIT', 'AMT_ANNUITY']

# Llamar a la función con el dataset
plot_comparisons(feat)


# #### Análisis de los gráficos 
# 
# - **Gráfico sobre la comparación de la cantidad del crédito**
#    - Los clientes con dificultades para pagar el préstamo son sobre todo aquellos cuya cantidad excede el millón. 
#    - Los clientes cuyo préstamos es inferior a un millón tienen por lo general, menos dificultades para pagarlo.
# 
# - **Gráfico sobre la comparación de la cantidad de la prima anual**
#    - Los valores de primas que se encuentran en los extremos (Mayor anualidad / Menor anualidad) tienen más dificultades para pagar el préstamo
#    - Los clientes con una prima media presentan menor dificultad

# In[88]:


plt.figure(figsize=(10, 12))

# Lista de fuentes externas
externals = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

# Generar gráficos para cada fuente externa
for i, source in enumerate(externals):
    plt.subplot(3, 1, i + 1)
    
    # Distribución para TARGET=0
    sns.kdeplot(
        data.loc[data['TARGET'] == 0, source], 
        label='Clientes sin dificultades para pagar', 
        shade=True, 
        color='skyblue'
    )
    
    # Distribución para TARGET=1
    sns.kdeplot(
        data.loc[data['TARGET'] == 1, source], 
        label='Clientes con dificultades para pagar', 
        shade=True, 
        color='salmon'
    )
    
    # Configuración del gráfico
    plt.title(f'Distribución de {source} por Valor de TARGET', fontsize=14)
    plt.xlabel(f'{source}', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend(fontsize=10)

# Ajustar el espaciado entre los subgráficos
plt.tight_layout(h_pad=2.5)
plt.show()


# Son variables que tienen un impacto significativo a la hora de analizar la variable objetivo, pero desconocemos el tipo de informacion que contienen estos datos, por lo que no se puede llegar a una conclusión concreta acerca de la relevancia en el modelo. 

# ## __Analizar variables numéricas__ ##

# In[42]:


# Describir estadísticas de variables numéricas
print("\nEstadísticas descriptivas de las variables numéricas:")
print(data_cleaned[numerical_vars].describe())


# ## __Preprocesamiento inicial de algunas variables__

# In[43]:


# Listar todas las columnas del dataset
print("Columnas en el dataset:")
print(data_cleaned.columns.tolist())


# In[44]:


# Convertir días a años para facilitar la interpretación
columns_days_to_years = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']

for col in columns_days_to_years:
    if col in data_cleaned.columns:
        data_cleaned[col + '_YEARS'] = (data_cleaned[col] / -365).round(1)  # Dividir entre -365 para años

# Verificar los resultados
print("Transformación de días a años completada:")
print(data_cleaned[[col + '_YEARS' for col in columns_days_to_years]].head())


# In[45]:


# Verificar columnas financieras
columns_financial = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

print("Ejemplo de columnas financieras:")
print(data_cleaned[columns_financial].describe())


# In[46]:


# Asegurar que las columnas categóricas estén correctamente codificadas
categorical_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                       'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'EMERGENCYSTATE_MODE']

for col in categorical_columns:
    if col in data_cleaned.columns:
        data_cleaned[col] = data_cleaned[col].astype('category')

# Verificar los cambios
print("Transformación de columnas categóricas completada:")
print(data_cleaned[categorical_columns].dtypes)


# In[ ]:


# Guardar el dataset actualizado
import os

output_dir = "data_preprocessing"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "data_cleaned_transformed.csv")

data_cleaned.to_csv(output_path, index=False)
print(f"Dataset preprocesado guardado en: {output_path}")

