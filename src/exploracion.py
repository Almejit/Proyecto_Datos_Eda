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