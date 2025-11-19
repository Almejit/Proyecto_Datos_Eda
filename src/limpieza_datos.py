import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detectar_outliers(df):
   
    print("--- Generando gráficos individuales de detección de outliers ---")
    
    # 1. Seleccionamos todas las numéricas
    numerical_cols = df.select_dtypes(include=['number']).columns
    
   
    columnas_a_graficar = [col for col in numerical_cols if col not in ['year', 'Unnamed: 0']]

    print(f"Se generarán {len(columnas_a_graficar)} gráficos individuales. Cierra uno para ver el siguiente.")


    for col in columnas_a_graficar:
       
        plt.figure(figsize=(10, 6))
        
     
        sns.boxplot(x=df[col], color='#FF9999') 
        
       
        plt.title(f'Análisis de Outliers: {col}', fontsize=16, fontweight='bold')
        plt.xlabel(col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
       
        plt.tight_layout()
        plt.show()
       

def eliminar_outliers(df, columnas=None):
    
    print("\n--- Tratamiento de Outliers (Método IQR) ---")
    df_limpio = df.copy()
    filas_iniciales = df_limpio.shape[0]
    
    if columnas is None:
       
        columnas = ['AveragePrice', 'Total Volume']

    for col in columnas:
        if col in df_limpio.columns:
            Q1 = df_limpio[col].quantile(0.25)
            Q3 = df_limpio[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            condicion = (df_limpio[col] >= limite_inferior) & (df_limpio[col] <= limite_superior)
            df_limpio = df_limpio[condicion]

    filas_finales = df_limpio.shape[0]
    print(f"Se eliminaron {filas_iniciales - filas_finales} filas por outliers en {columnas}.")
    
    return df_limpio

def tratar_valores_nulos(df):
    
    print("\n--- Tratamiento de Valores Nulos (Eliminación) ---")
    df_tratado = df.copy()
    
    nulos_antes = df_tratado.isnull().sum().sum()
    filas_antes = df_tratado.shape[0]
    
    df_tratado.dropna(inplace=True)
    
    filas_despues = df_tratado.shape[0]
    eliminadas = filas_antes - filas_despues
    
    print(f"Nulos encontrados: {nulos_antes}")
    print(f"Filas eliminadas: {eliminadas}")
    
    return df_tratado