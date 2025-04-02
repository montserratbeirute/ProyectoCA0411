# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:24:14 2025

@author: monts
"""

# Importamos las librerías y los módulos de algunas librerías
import pandas as pd
import os 

ruta_base = os.path.dirname(os.path.abspath(__file__))

# Cargar el CSV
df_loan = pd.read_csv(os.path.join(ruta_base, "..", "data", "loan_data.csv"))

# Tipo categoría
df_loan['not.fully.paid'].astype("category")

print(df_loan.describe())



