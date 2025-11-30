import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def transformar_preparar_datos(df):
    
    print("="*60)
    print("TRANSFORMACIÓN Y PREPARACIÓN DE DATOS")
    print("="*60)
    
    df_transformed = df.copy()
    
    
    
    print("\n1. ESTANDARIZACIÓN Y NORMALIZACIÓN")
    print("-"*60)
    
    # Columnas numéricas para escalar
    columnas_numericas = [
        'AveragePrice', 'Total Volume', '4046', '4225', '4770',
        'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags'
    ]
    
    # a) StandardScaler (media=0, desviación=1)
    scaler_std = StandardScaler()
    df_transformed[['AveragePrice_std', 'Total Volume_std']] = scaler_std.fit_transform(
        df_transformed[['AveragePrice', 'Total Volume']]
    )
    print("✓ Estandarización aplicada a AveragePrice y Total Volume")
    print(f"  - Media AveragePrice_std: {df_transformed['AveragePrice_std'].mean():.4f}")
    print(f"  - Std AveragePrice_std: {df_transformed['AveragePrice_std'].std():.4f}")
    
    # b) MinMaxScaler (rango 0-1) para otras variables
    scaler_minmax = MinMaxScaler()
    columnas_normalizar = ['4046', '4225', '4770']
    df_transformed[[col + '_norm' for col in columnas_normalizar]] = scaler_minmax.fit_transform(
        df_transformed[columnas_normalizar]
    )
    print(f"\n✓ Normalización (0-1) aplicada a códigos PLU")
    print(f"  - Rango 4046_norm: [{df_transformed['4046_norm'].min():.2f}, {df_transformed['4046_norm'].max():.2f}]")
    
    

    
    print("\n\n2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    print("-"*60)
    
    # a) Label Encoding para 'type' (binaria: conventional/organic)
    le_type = LabelEncoder()
    df_transformed['type_encoded'] = le_type.fit_transform(df_transformed['type'])
    print("✓ Label Encoding aplicado a 'type':")
    print(f"  - conventional = {le_type.transform(['conventional'])[0]}")
    print(f"  - organic = {le_type.transform(['organic'])[0]}")
    
    # b) One-Hot Encoding para 'region' (múltiples categorías)
    df_onehot = pd.get_dummies(df_transformed, columns=['region'], prefix='region', drop_first=True)
    print(f"\n✓ One-Hot Encoding aplicado a 'region'")
    print(f"  - Regiones originales: {df_transformed['region'].nunique()}")
    print(f"  - Columnas creadas: {len([col for col in df_onehot.columns if col.startswith('region_')])}")
    
    # Actualizar dataframe
    df_transformed = df_onehot
    
    
    # 3. FEATURE ENGINEERING (Creación de nuevas variables)
    
    print("\n\n3. FEATURE ENGINEERING - NUEVAS VARIABLES")
    print("-"*60)
    
    # Proporción de bolsas sobre volumen total
    df_transformed['bags_ratio'] = df_transformed['Total Bags'] / (df_transformed['Total Volume'] + 1)
    print("✓ bags_ratio: Proporción de bolsas sobre volumen total")
    
    # Precio por unidad de volumen
    df_transformed['price_per_volume'] = df_transformed['AveragePrice'] / (df_transformed['Total Volume'] + 1)
    print("✓ price_per_volume: Precio por unidad de volumen")
    
    # Dominancia de tamaño de bolsa
    df_transformed['small_bag_dominance'] = df_transformed['Small Bags'] / (df_transformed['Total Bags'] + 1)
    df_transformed['large_bag_dominance'] = df_transformed['Large Bags'] / (df_transformed['Total Bags'] + 1)
    print("✓ small/large_bag_dominance: Proporción de cada tipo de bolsa")
    
    # Variables temporales desde Date
    if 'Date' in df_transformed.columns:
        df_transformed['month'] = pd.to_datetime(df_transformed['Date']).dt.month
        df_transformed['quarter'] = pd.to_datetime(df_transformed['Date']).dt.quarter
        df_transformed['week_of_year'] = pd.to_datetime(df_transformed['Date']).dt.isocalendar().week
        print("✓ Variables temporales: month, quarter, week_of_year")
    
    # Volumen total de códigos PLU
    df_transformed['total_plu_volume'] = df_transformed['4046'] + df_transformed['4225'] + df_transformed['4770']
    print("✓ total_plu_volume: Suma de todos los códigos PLU")
    
    # Categoría de precio
    df_transformed['price_category'] = pd.cut(
        df_transformed['AveragePrice'],
        bins=[0, 1.0, 1.5, 2.0, float('inf')],
        labels=['Bajo', 'Medio', 'Alto', 'Premium']
    )
    print("✓ price_category: Categorización del precio en rangos")
  
    df_transformed['type_price_interaction'] = df_transformed['type_encoded'] * df_transformed['AveragePrice']
    print("✓ type_price_interaction: Interacción entre tipo y precio")
    
    
    # RESUMEN FINAL
   
    print("\n\n" + "="*60)
    print("RESUMEN DE TRANSFORMACIONES")
    print("="*60)
    print(f"Dimensiones originales: {df.shape}")
    print(f"Dimensiones transformadas: {df_transformed.shape}")
    print(f"Nuevas columnas agregadas: {df_transformed.shape[1] - df.shape[1]}")
    
    
    print("\n📊 Muestra de variables transformadas:")
    columnas_muestra = ['AveragePrice', 'AveragePrice_std', 'type', 'type_encoded', 
                        'bags_ratio', 'price_per_volume', 'price_category']
    print(df_transformed[columnas_muestra].head())
    
    print("\n✅ Transformación completada exitosamente!")
    
    return df_transformed, scaler_std, scaler_minmax, le_type


def preparar_para_ml(df_transformed, target_column='AveragePrice'):
    """
    Prepara el dataset para algoritmos de Machine Learning
    """
    print("\n\n" + "="*60)
    print("PREPARACIÓN PARA MACHINE LEARNING")
    print("="*60)
    
    # Eliminar columnas no necesarias
    columnas_drop = ['Date', 'type', 'price_category']  # Categóricas originales
    df_ml = df_transformed.drop(columns=[col for col in columnas_drop if col in df_transformed.columns])
    
    # Separar features y target
    if target_column in df_ml.columns:
        X = df_ml.drop(columns=[target_column])
        y = df_ml[target_column]
        
        print(f"✓ Target: {target_column}")
        print(f"✓ Features: {X.shape[1]} variables")
        
       
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📦 División de datos:")
        print(f"  - Train: {X_train.shape[0]} muestras")
        print(f"  - Test: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test
    else:
        print(f"⚠️ Columna target '{target_column}' no encontrada")
        return None


