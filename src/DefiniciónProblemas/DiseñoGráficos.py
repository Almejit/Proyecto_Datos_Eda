# ==============================================================================
# IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==============================================================================
# I. FUNCIONES DE UTILIDAD
# ==============================================================================
def crear_carpeta_graficos():
    """Crea la carpeta 'graficos' si no existe y devuelve su ruta."""
    graficos_dir = Path('graficos')
    graficos_dir.mkdir(exist_ok=True)
    return graficos_dir



# ==============================================================================
# II. GRÁFICOS DEL 1 AL 13 CON COMENTARIOS
# ==============================================================================

# ---------------------------------------------------------------------------------------
# 1. ¿Cómo se distribuyen los precios promedio de los aguacates?
# Este gráfico muestra un histograma con KDE para ver la forma de la distribución
# del precio medio: si es simétrica, sesgada, concentrada o dispersa.
# ---------------------------------------------------------------------------------------
def grafico_1(df, graficos_dir=None, guardar=False):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AveragePrice'], kde=True, bins=30, color='skyblue')
    plt.title('1. Distribución del Precio Promedio (AveragePrice)', fontsize=14, fontweight='bold')
    plt.xlabel('Precio Promedio ($)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '01_histograma_precio.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 2. ¿Existen valores atípicos (outliers) en el volumen total vendido?
# Con un boxplot es fácil identificar si hay ventas extremadamente altas o bajas que
# no siguen la tendencia general.
# ---------------------------------------------------------------------------------------
def grafico_2(df, graficos_dir=None, guardar=False):
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Total Volume'], color='salmon')
    plt.title('2. Boxplot de Volumen Total (Detección de Outliers)', fontsize=14, fontweight='bold')
    plt.ylabel('Volumen Total')
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '02_boxplot_volumen.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 3. ¿Qué tipo de aguacate (orgánico o convencional) es más caro?
# Este boxplot compara la distribución de precios entre los dos tipos de aguacate
# para identificar diferencias de costo.
# ---------------------------------------------------------------------------------------
def grafico_3(df, graficos_dir=None, guardar=False):
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='type',
        y='AveragePrice',
        data=df,
        palette={'conventional': 'orange', 'organic': 'green'}
    )
    plt.title('3. AveragePrice por Tipo de Aguacate', fontsize=14, fontweight='bold')
    plt.xlabel('Tipo')
    plt.ylabel('Precio Promedio')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '03_boxplot_precio_por_tipo.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 4. ¿Cómo ha cambiado el precio promedio a lo largo del tiempo?
# Esta gráfica de línea permite ver la tendencia temporal (subidas, bajadas y ciclos).
# ---------------------------------------------------------------------------------------
def grafico_4(df, graficos_dir=None, guardar=False):
    df_time = df.groupby('Date')['AveragePrice'].mean().reset_index()
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date', y='AveragePrice', data=df_time, color='purple', linewidth=2)
    plt.title('4. Tendencia del Precio Promedio a lo largo del Tiempo', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '04_linea_precio_temporal.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 5. ¿Cómo evoluciona el volumen de ventas según el tipo de aguacate?
# Gráfico temporal comparando ventas de aguacate orgánico vs convencional.
# ---------------------------------------------------------------------------------------
def grafico_5(df, graficos_dir=None, guardar=False):
    df_vol_tipo = df.groupby(['Date', 'type'])['Total Volume'].sum().reset_index()
    plt.figure(figsize=(14, 6))

    for tipo in df_vol_tipo['type'].unique():
        subset = df_vol_tipo[df_vol_tipo['type'] == tipo]
        plt.plot(subset['Date'], subset['Total Volume'], label=tipo, linewidth=2)

    plt.title('5. Evolución del Volumen Total por Tipo', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '05_volumen_por_tipo.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 6. ¿Qué tipo de bolsa es la más utilizada en la venta de aguacates?
# Compara bolsas pequeñas, grandes y extra grandes para identificar cuál se usa más.
# ---------------------------------------------------------------------------------------
def grafico_6(df, graficos_dir=None, guardar=False):
    bags_cols = ['Small Bags', 'Large Bags', 'XLarge Bags']
    bags_data = df[bags_cols].sum()

    plt.figure(figsize=(10, 6))
    plt.bar(bags_data.index, bags_data.values)
    plt.title('6. Distribución Total de Tipos de Bolsas', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '06_distribucion_bolsas.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 7. ¿Qué tan relacionadas están las variables numéricas del dataset?
# El heatmap permite identificar correlaciones fuertes y relaciones importantes.
# ---------------------------------------------------------------------------------------
def grafico_7(df, graficos_dir=None, guardar=False):
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('7. Matriz de Correlación General', fontsize=14, fontweight='bold')

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '07_heatmap_correlacion_general.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 8. ¿Cuáles son las regiones donde los aguacates son más caros?
# Ordena las 15 regiones con mayor precio promedio.
# ---------------------------------------------------------------------------------------
def grafico_8(df, graficos_dir=None, guardar=False):
    top_regions = df.groupby('region')['AveragePrice'].mean().sort_values(ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_regions.values, y=top_regions.index)
    plt.title('8. Top 15 Regiones con Mayor Precio Promedio', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '08_top_regiones_precio.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 9. ¿Qué código PLU (tamaño del aguacate) se vende más?
# Compara la cantidad total vendida de los PLUs 4046, 4225 y 4770.
# ---------------------------------------------------------------------------------------
def grafico_9(df, graficos_dir=None, guardar=False):
    plu_cols = ['4046', '4225', '4770']
    plu_data = df[plu_cols].sum()

    plt.figure(figsize=(10, 6))
    plt.bar(plu_cols, plu_data.values)
    plt.title('9. Distribución Total de Códigos PLU', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '09_distribucion_plu.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 10. ¿Cuántos registros tiene cada región?
# Muestra la cantidad de observaciones disponibles por región.
# ---------------------------------------------------------------------------------------
def grafico_10(df, graficos_dir=None, guardar=False):
    plt.figure(figsize=(10, 8))
    sns.countplot(y='region', data=df, order=df['region'].value_counts().index)
    plt.title('10. Conteo de Observaciones por Región', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '10_conteo_regiones.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 11. ¿Qué variables numéricas están más fuertemente relacionadas?
# Heatmap avanzado solo con variables relevantes elegidas manualmente.
# ---------------------------------------------------------------------------------------
def grafico_11(df, graficos_dir=None, guardar=False):
    correlation_cols = [
        'AveragePrice', 'Total Volume', '4046', '4225', '4770',
        'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year'
    ]
    cols = [col for col in correlation_cols if col in df.columns]

    corr_matrix = df[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('11. Heatmap Avanzado', fontsize=14, fontweight='bold')

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '11_heatmap_avanzado.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 12. ¿Cómo cambia la variabilidad de precios por año y tipo de aguacate?
# El violin plot permite ver la dispersión y forma de distribución por año.
# ---------------------------------------------------------------------------------------
def grafico_12(df, graficos_dir=None, guardar=False):
    if 'year' not in df.columns or 'type' not in df.columns:
        return

    df['year_str'] = df['year'].astype(str)

    plt.figure(figsize=(14, 7))
    sns.violinplot(x='year_str', y='AveragePrice', hue='type', data=df, split=True)
    plt.title('12. Volatilidad de Precios por Año y Tipo', fontsize=14, fontweight='bold')

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '12_violin_plot.png', dpi=300)
    plt.show()



# ---------------------------------------------------------------------------------------
# 13. ¿Qué regiones tienen mayor variabilidad en el precio?
# Se calcula el IQR (rango intercuartílico) del precio por región.
# ---------------------------------------------------------------------------------------
def grafico_13(df, graficos_dir=None, guardar=False):
    if 'region' not in df.columns:
        return

    region_stats = df.groupby('region')['AveragePrice'].agg(
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index().sort_values(by='IQR', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='IQR', y='region', data=region_stats.head(20))
    plt.title('13. Top 20 Regiones por IQR del Precio', fontsize=14, fontweight='bold')

    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '13_iqr_regional.png', dpi=300)
    plt.show()
