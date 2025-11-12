from pathlib import Path
import pandas as pd

def cargar_datos():
    try:
        
        df = pd.read_csv('data/avocado.csv')
    except FileNotFoundError:
        
        csv_path = Path(__file__).resolve().parent.parent / 'data' / 'avocado.csv'
        df = pd.read_csv(csv_path)
    
    columnas_numericas = [
        "AveragePrice", "Total Volume", "4046", "4225", "4770",
        "Total Bags", "Small Bags", "Large Bags", "XLarge Bags", "year"
    ]
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    return df

df = cargar_datos()
print(df.info())
print(df.head())

