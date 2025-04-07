# -*- coding: utf-8 -*-

"""
Created on Wed Apr  2 11:24:14 2025

@author: monts
"""
# Descarga de librerias --------------------------------------

# Importamos las librerías y los módulos de algunas librerías
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns

# Lectura de datos -----------------------------------------

# Devuelve el directorio de un nombre de ruta dado
ruta_base = os.path.dirname(os.path.abspath(__file__))

# Cargar el CSV
df_loan = pd.read_csv(os.path.join(ruta_base, "..", "data", "loan_data.csv"))


# Visualización de datos ----------------------------------

# Visualizamos primeras 5 filas de la df 
print(df_loan.head())

# Imprimimos codigo de latex para primeras 5 filas
print(df_loan.round(2).head().to_latex(index = 0))

# Resumen de 5 números ----------------------------------

# Obtener el resumen de 5 números de las variables numéricas
print(df_loan.describe().loc[['min', '25%', '50%', '75%', 'max']])


# Análisis variables cuantitativas ----------------------

# Filtrar solo columnas numéricas
datos_cuantitativos = df_loan.select_dtypes(include=['number'])


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





