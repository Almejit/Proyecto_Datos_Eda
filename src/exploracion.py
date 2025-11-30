import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# Importaciones necesarias para Detección de Patrones/Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 

# -----------------------------------------------------------
# 1. EXPLORACIÓN BÁSICA (PUNTOS 1-4)
# -----------------------------------------------------------

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
    
# -----------------------------------------------------------
# 2. ANÁLISIS EDA AVANZADO (PUNTO 5)
# -----------------------------------------------------------

def analisis_eda(df):
    """
    Realiza el Análisis Exploratorio de Datos (EDA) avanzado:
    1. Matriz de Correlación (Heatmap).
    2. Comparación de distribuciones (Violin Plot).
    3. Relaciones entre variables clave (Pairplot).
    """
    print("\n\n" + "=" * 25 , "5. ANÁLISIS EXPLORATORIO DE DATOS (EDA) AVANZADO" , "=" * 25)

    # 1. Matriz de Correlación (Heatmap)
    print("\n--- 1. Matriz de Correlación y Heatmap ---")
    
    numerical_cols = df.select_dtypes(include=['number']).columns.drop('Unnamed: 0', errors='ignore')
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        linewidths=.5
    )
    plt.title('Mapa de Calor (Heatmap) de la Matriz de Correlación', fontsize=16, fontweight='bold')
    plt.show()
    # 



    # 2. Comparar Distribuciones entre grupos (Violin Plot: Precio vs Tipo)
    print("\n--- 2. Comparación de Distribuciones: Precio Promedio vs Tipo ---")
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='type', y='AveragePrice', data=df, palette={'conventional': 'skyblue', 'organic': 'lightcoral'})
    plt.title('Distribución del Precio Promedio por Tipo de Aguacate', fontsize=16, fontweight='bold')
    plt.xlabel('Tipo', fontsize=12)
    plt.ylabel('Precio Promedio', fontsize=12)
    plt.show()

    # 3. Tendencias y Relaciones causa-efecto (Pairplot)
    print("\n--- 3. Relaciones Clave: Pairplot ---")
    
    cols_for_pairplot = ['AveragePrice', 'Total Volume', 'Small Bags', 'year']
    
    sns.pairplot(df, vars=cols_for_pairplot, hue='type', diag_kind='kde')
    plt.suptitle('Pairplot de Variables Clave Segmentado por Tipo', y=1.02, fontsize=16, fontweight='bold')
    plt.show()
    
    print("\n✅ EDA Avanzado completado. Gráficos generados.")

# -----------------------------------------------------------
# 3. DETECCIÓN DE PATRONES Y AGRUPACIONES (PUNTO 5.2/6)
# -----------------------------------------------------------

def deteccion_patrones(df):
    """
    Realiza la detección de patrones avanzados:
    1. Clustering visual (KMeans + PCA) para detectar grupos de regiones.
    2. FacetGrid para comparar distribuciones de precio en distintas categorías.
    """
    print("\n\n" + "=" * 25 , "6. DETECCIÓN DE PATRONES Y AGRUPACIONES (CLUSTERING)" , "=" * 25)

    # 1. Preparación de datos para Clustering (Buscando grupos de regiones)
    df_agrupado = df.groupby('region')[['AveragePrice', 'Total Volume']].mean().reset_index()
    
    X_cluster = df_agrupado[['AveragePrice', 'Total Volume']]
    
    # Escalado de datos para que las variables tengan peso similar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Aplicar K-Means (Se prueba con K=3: regiones caras, medias y baratas)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_agrupado['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print("\n--- 1. Clustering Visual (Grupos de Regiones) ---")
    print(f"✅ Se detectaron {df_agrupado['Cluster'].nunique()} grupos de regiones. Distribución:")
    print(df_agrupado.groupby('Cluster')['region'].count())
    
    # Visualización 2D usando PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_agrupado['PCA1'] = components[:, 0]
    df_agrupado['PCA2'] = components[:, 1]
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA1', 
        y='PCA2', 
        hue='Cluster', 
        data=df_agrupado, 
        palette='viridis', 
        style='Cluster', 
        s=100
    )
    plt.title('Clustering de Regiones por Precio y Volumen (Visualización PCA)', fontsize=16, fontweight='bold')
    plt.xlabel(f'PCA Componente 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'PCA Componente 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.show()
    # 

    # 4. FacetGrid: Comparar distribuciones de precio en grupos clave
    print("\n--- 2. FacetGrid: Comparación de Precios por Región y Tipo ---")
    
    # Seleccionamos un subconjunto de regiones para una visualización clara
    regiones_clave = ['California', 'NewYork', 'TotalUS', 'Boston', 'Seattle']
    df_sub = df[df['region'].isin(regiones_clave)]
    
    # Usamos FacetGrid para ver el histograma de AveragePrice segmentado por 'region' y 'type'
    g = sns.FacetGrid(df_sub, col="region", row="type", margin_titles=True, height=3)
    
    g.map_dataframe(sns.histplot, x="AveragePrice", bins=15, kde=True)
    
    g.set_axis_labels("Precio Promedio", "Frecuencia")
    g.set_titles(col_template="{col_name} Región", row_template="{row_name} Tipo")
    plt.suptitle('FacetGrid: Distribución del Precio Promedio en Regiones Clave por Tipo', y=1.05, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    # 

    print("\n✅ Detección de patrones y FacetGrid completados.")