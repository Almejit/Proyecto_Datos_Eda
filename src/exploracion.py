import pandas as pd
import matplotlib.pyplot as plt

def explorar_datos(df):
    print("=" * 25 ,"Primeras filas" ,"=" * 25)
    print(df.head())

    print("=" * 25 ,"Dimensiones" ,"=" * 25)
    print(df.shape)

    print("=" * 25 ,"Tipos de datos" ,"=" * 25)
    print(df.dtypes)

    df.info()
    
    print("=" * 25 ,"Estadísticas" ,"=" * 25)
    print(df.describe())

    print("=" * 25 ,"Valores nulos por columna" ,"=" * 25)
    print(df.isnull().sum())
    
    print("\nValores faltantes por columna:")
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    missing_info = pd.DataFrame({
    'Valores Faltantes': missing_values,
    '% Faltantes': missing_percentage
    })

    missing_info = missing_info[missing_info['Valores Faltantes'] > 0].sort_values(by='% Faltantes', ascending=False)
    print(missing_info)