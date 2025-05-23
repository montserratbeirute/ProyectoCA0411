# -*- coding: utf-8 -*-

"""
Created on Wed Apr  2 11:24:14 2025

@author: Beirute, Campos, Gómez y Hernández
"""
# Descarga de librerias ------------------------------------------------------

# Importamos las librerías y los módulos de algunas librerías
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Lectura de datos -----------------------------------------------------------

# Devuelve el directorio de un nombre de ruta dado
ruta_base = os.path.dirname(os.path.abspath(__file__))

# Cargar el CSV
df_loan = pd.read_csv(os.path.join(ruta_base, "..", "data", "loan_data.csv"))


# Visualización de datos -----------------------------------------------------

# Visualizamos primeras 5 filas de la df 
print(df_loan.head())

# Imprimimos codigo de latex para primeras 5 filas
print(df_loan.round(2).head().to_latex(index = 0))

# Resumen de 5 números -------------------------------------------------------

# Obtener el resumen de 5 números de las variables numéricas
print(df_loan.describe().loc[['min', '25%', '50%', '75%', 'max']])


# Análisis variables cuantitativas -------------------------------------------

# Filtrar solo columnas numéricas
datos_cuantitativos = df_loan.select_dtypes(include=['number']).drop(columns=['credit.policy', 'not.fully.paid'])

# Configurar el tamaño del gráfico
plt.figure(figsize=(12, 6))

# Crear histogramas para cada variable cuantitativa
for columna in datos_cuantitativos.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_loan[columna], kde=True, bins=20, color="skyblue")
    plt.title(f'Distribución de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.show()
    

# Crear boxplots para visualizar la dispersión
plt.figure(figsize=(12, 6))
sns.boxplot(data=datos_cuantitativos, palette="pastel")
plt.title("Boxplot de las Variables Cuantitativas")
plt.xticks(rotation=45)
plt.show()


# Matriz de dispersión
sns.pairplot(datos_cuantitativos, diag_kind="kde", plot_kws={'alpha':0.6, 's':10})
plt.suptitle("Matriz de Dispersión entre Variables Cuantitativas", y=1.02)
plt.show()

# Mapa de calor de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(datos_cuantitativos.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de Calor - Correlación entre Variables")
plt.show()

# Análisis variables cualitativas --------------------------------------------

# Sabemos que hay 3 variables cualitativas
df_loan['credit.policy'].astype("category")
df_loan['purpose'].astype("category")
df_loan['not.fully.paid'].astype("category")


# Queremos saber el # de 0s y 1s para dos de esas variables
print(df_loan['not.fully.paid'].value_counts())
print(df_loan['credit.policy'].value_counts())


# Distribución de variable purpose
df_loan['purpose'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title('Frecuencia de cada categoría en purpose')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribución porcentual
df_loan['purpose'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Distribución porcentual de purpose')
plt.ylabel('')  
plt.show()

# Detección de valores nulos ------------------------------------------------

# Para cada variable, sumamos los valores nulos
valores_faltantes = df_loan.isnull().sum()

# No hay valores nulos
print(df_loan.isnull().sum())

# Detectar outliers ---------------------------------------------------------

# Seleccionamos solo variables numéricas y excluimos las variables cualitativas
numericas = df_loan.select_dtypes(include=['float64', 'int64']).drop(columns=['credit.policy', 'not.fully.paid'])

# Detectar outliers por variable
outliers = {}

for col in numericas.columns:
    Q1 = df_loan[col].quantile(0.25)
    Q3 = df_loan[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outlier_cond = (df_loan[col] < limite_inferior) | (df_loan[col] > limite_superior)
    total_outliers = outlier_cond.sum()

    outliers[col] = {
        'outliers': total_outliers,
        'porcentaje': round(total_outliers / len(df_loan) * 100, 2)
    }

# Mostrar resultados
outliers_df = pd.DataFrame(outliers).T
print(outliers_df)


# Para presentar resultados en tabla

# Crear lista para almacenar los resultados
tabla = []

# Calcular estadísticas para cada variable numérica
for col in numericas.columns:
    serie = numericas[col]
    
    # Cuartiles y extremos
    min_val = serie.min()
    Q1 = serie.quantile(0.25)
    Q2 = serie.quantile(0.5)
    Q3 = serie.quantile(0.75)
    max_val = serie.max()
    
    # Conteo de observaciones en los rangos solicitados
    count_min_q1 = ((serie >= min_val) & (serie < Q1)).sum()
    count_q3_max = ((serie > Q3) & (serie <= max_val)).sum()
    
    # Añadir fila a la tabla
    tabla.append({
        'Variable': col,
        'Mínimo': round(min_val, 2),
        'Q1': round(Q1, 2),
        'Q2 (Mediana)': round(Q2, 2),
        'Q3': round(Q3, 2),
        'Máximo': round(max_val, 2),
        'Valores entre Mínimo y Q1': count_min_q1,
        'Valores entre Q3 y Máximo': count_q3_max
    })

# Convertir a DataFrame
tabla_df = pd.DataFrame(tabla)

# Graficar distribuciones con estadísticos -----------------------------------

def graficar_distribuciones(df, excluir=[]):
    """
    Grafica la distribución de cada variable numérica (excepto las excluidas),
    incluyendo líneas verticales para la mediana y el tercer cuartil (Q3).
    
    Parámetros:
        df (pd.DataFrame): DataFrame original
        excluir (list): lista de columnas a excluir
    """
    df_numericas = df.select_dtypes(include=['float64', 'int64'])

    for col in df_numericas.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue', edgecolor='black')

        # Calcular valores estadísticos
        mediana = df[col].median()
        q3 = df[col].quantile(0.75)

        # Añadir líneas verticales
        plt.axvline(mediana, color='red', linestyle='--', linewidth=2, label=f'Mediana: {round(mediana, 2)}')
        plt.axvline(q3, color='green', linestyle='--', linewidth=2, label=f'Q3: {round(q3, 2)}')

        # Etiquetas
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
excluir = []        
graficar_distribuciones(numericas,excluir)


# Resolver el problema de outliers -------------------------------------------

# 1. Binarizar `pub.rec`: 0 = sin registros, 1 = con al menos uno
df_loan['pub.rec_flag'] = df_loan['pub.rec'].apply(lambda x: 1 if x > 0 else 0)

# 2. Agrupar `delinq.2yrs`: '0' vs '1 o más'
df_loan['delinq.2yrs_binned'] = df_loan['delinq.2yrs'].apply(lambda x: '0' if x == 0 else '1+')

# 3. Transformar `revol.bal` con logaritmo para reducir asimetría
df_loan['log.revol.bal'] = np.log1p(df_loan['revol.bal'])  # log(x + 1) evita log(0)


# Aplicación del modelo AdaBoost  ---------------------------------------------
from sklearn.model_selection import train_test_split # para separar los datos en entrenamiento y prueba
from sklearn.preprocessing import OneHotEncoder # para trabajar con la variable categórica
from sklearn.compose import ColumnTransformer # se usa junto con el OneHotEncoder
from sklearn.pipeline import Pipeline # para procesar datos y crear el modelo como un solo objeto
from sklearn.ensemble import AdaBoostClassifier # clasificador
from sklearn.tree import DecisionTreeClassifier # para el clasificador débil del AdaBoost
from sklearn.metrics import classification_report, confusion_matrix # para mostrar resultados
from mlxtend.plotting import plot_confusion_matrix # para graficar matriz de confusión
from sklearn.ensemble import RandomForestClassifier

# Separar en dos df's: X variables de entrada, Y variable de salida (not.fully.paid)
X = df_loan.drop("not.fully.paid", axis = 1)
Y = df_loan["not.fully.paid"]

variables_categoricas = ['purpose', 'delinq.2yrs_binned']  # solo 'purpose' es categórica
variables_numericas = X.drop(columns=variables_categoricas).columns.tolist()

# Transformador de columnas categóricas en dummies
preprocesar_datos = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), variables_categoricas)], 
                                      remainder = "passthrough")

# Clasificador AdaBoost con árbol de decisión
adaboost = AdaBoostClassifier(
    estimator = DecisionTreeClassifier(max_depth=3, class_weight="balanced"), # modelo débil usando pesos
    n_estimators = 50, # número de modelos débiles 
    learning_rate = 0.005,
    random_state = 123 # semilla del generador aleatorio
)

# Crear el objeto Pipeline
pipeline_adaboost = Pipeline([
    ("preprocesar", preprocesar_datos),
    ("clasificador", adaboost)
])

# Separar datos en train y test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=123)

# Entrenar el modelo
pipeline_adaboost.fit(X_train, Y_train)

# Predecir
Y_pred = pipeline_adaboost.predict(X_test)

# Resultados
## Matriz de confusión
matriz_confusion = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(conf_mat=matriz_confusion)

print("\nReporte de clasificación:\n", classification_report(Y_test, Y_pred))

## Gráfico de Importancias
preprocesador = pipeline_adaboost.named_steps['preprocesar']
modelo = pipeline_adaboost.named_steps['clasificador']
nombres_variables = preprocesador.get_feature_names_out()
importancias = modelo.feature_importances_
resumen_importancias = pd.DataFrame({
    'Variable': nombres_variables,
    'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

# Gráfico de barras
plt.figure(figsize=(10,6))
plt.barh(resumen_importancias['Variable'], resumen_importancias['Importancia'])
plt.gca().invert_yaxis()
plt.title("Importancia de las variables - AdaBoost")
plt.tight_layout()
plt.show()
