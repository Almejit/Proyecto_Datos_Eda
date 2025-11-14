import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def crear_carpeta_graficos():
    """Crea la carpeta para guardar los gráficos si no existe."""
    graficos_dir = Path('graficos')
    graficos_dir.mkdir(exist_ok=True)
    return graficos_dir


def visualizar_exploracion(df: pd.DataFrame, guardar: bool = True):
    """
    Realiza las visualizaciones clave del Análisis Exploratorio de Datos.
    
    Args:
        df: DataFrame con los datos de aguacates
        guardar: Si True, guarda los gráficos en la carpeta 'graficos/'
    """
    # Configuración básica
    sns.set_style("whitegrid")
    
    # Crear carpeta para guardar gráficos
    if guardar:
        graficos_dir = crear_carpeta_graficos()
    
    # --- Preparación necesaria para la visualización ---
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Asegurar que 'Date' es datetime para la Serie de Tiempo
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # --- 1. Histograma de AveragePrice ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AveragePrice'], kde=True, bins=30, color='skyblue')
    plt.title('1. Distribución del Precio Promedio (AveragePrice)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Precio Promedio ($)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '01_histograma_precio.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 2. Boxplot de Variables de Volumen: Total Volume ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Total Volume'], color='salmon')
    plt.title('2. Boxplot de Volumen Total (Detección de Outliers)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Volumen Total', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '02_boxplot_volumen.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 3. Boxplot por Categoría: AveragePrice vs. type ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='type', y='AveragePrice', data=df, 
                palette={'conventional': 'orange', 'organic': 'green'})
    plt.title('3. AveragePrice por Tipo de Aguacate', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Tipo de Aguacate', fontsize=11)
    plt.ylabel('Precio Promedio ($)', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '03_boxplot_precio_por_tipo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 4. Gráfico de Líneas: AveragePrice vs. Date ---
    df_time = df.groupby('Date')['AveragePrice'].mean().reset_index()
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date', y='AveragePrice', data=df_time, color='purple', linewidth=2)
    plt.title('4. Tendencia del Precio Promedio a lo largo del Tiempo', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=11)
    plt.ylabel('Precio Promedio ($)', fontsize=11)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '04_linea_precio_temporal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # --- 6. Distribución de Tipos de Bolsas ---
    plt.figure(figsize=(10, 6))
    bags_cols = ['Small Bags', 'Large Bags', 'XLarge Bags']
    bags_data = df[bags_cols].sum()
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    plt.bar(bags_data.index, bags_data.values, color=colors)
    plt.title('6. Distribución Total de Tipos de Bolsas', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Tipo de Bolsa', fontsize=11)
    plt.ylabel('Cantidad Total', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '06_distribucion_bolsas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 7. Heatmap de Correlación ---
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, center=0, square=True)
    plt.title('7. Matriz de Correlación de Variables Numéricas', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '07_heatmap_correlacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 8. Precio por Región (Top 15) ---
    plt.figure(figsize=(12, 8))
    top_regions = df.groupby('region')['AveragePrice'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=top_regions.values, y=top_regions.index, palette='viridis')
    plt.title('8. Top 15 Regiones con Mayor Precio Promedio', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Precio Promedio ($)', fontsize=11)
    plt.ylabel('Región', fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '08_top_regiones_precio.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 9. Evolución de Volumen por Tipo ---
    plt.figure(figsize=(14, 6))
    df_vol_tipo = df.groupby(['Date', 'type'])['Total Volume'].sum().reset_index()
    for tipo in df_vol_tipo['type'].unique():
        data_tipo = df_vol_tipo[df_vol_tipo['type'] == tipo]
        plt.plot(data_tipo['Date'], data_tipo['Total Volume'], 
                label=tipo, linewidth=2)
    plt.title('9. Evolución del Volumen Total por Tipo de Aguacate', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=11)
    plt.ylabel('Volumen Total', fontsize=11)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '09_volumen_por_tipo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 10. Distribución de PLU Codes (4046, 4225, 4770) ---
    plt.figure(figsize=(10, 6))
    plu_cols = ['4046', '4225', '4770']
    plu_data = df[plu_cols].sum()
    plt.bar(plu_cols, plu_data.values, color=['#FFB366', '#66FFB2', '#B266FF'])
    plt.title('10. Distribución Total de Códigos PLU', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Código PLU', fontsize=11)
    plt.ylabel('Cantidad Total', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if guardar:
        plt.savefig(graficos_dir / '10_distribucion_plu.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    if guardar:
        print(f"\n✓ Todas las visualizaciones guardadas en '{graficos_dir}/'")
    
    print("\n✓ Visualizaciones de Exploración Inicial completadas.")


# Función adicional para generar solo algunos gráficos específicos
def visualizar_especificos(df: pd.DataFrame, graficos: list, guardar: bool = True):
    """
    Genera solo los gráficos especificados.
    
    Args:
        df: DataFrame con los datos
        graficos: Lista con números de gráficos a generar (1-10)
        guardar: Si True, guarda los gráficos
    
    Ejemplo:
        visualizar_especificos(df, [1, 3, 4])
    """
    sns.set_style("whitegrid")
    
    if guardar:
        graficos_dir = crear_carpeta_graficos()
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    for num in graficos:
        if num == 1:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['AveragePrice'], kde=True, bins=30, color='skyblue')
            plt.title('Distribución del Precio Promedio', fontweight='bold')
            plt.tight_layout()
            if guardar:
                plt.savefig(graficos_dir / '01_histograma_precio.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        elif num == 2:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df['Total Volume'], color='salmon')
            plt.title('Boxplot de Volumen Total', fontweight='bold')
            plt.tight_layout()
            if guardar:
                plt.savefig(graficos_dir / '02_boxplot_volumen.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Agregar más casos según necesites...
    
    print(f"\n✓ Gráficos {graficos} generados.")