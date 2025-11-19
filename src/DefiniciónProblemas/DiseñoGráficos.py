# ==============================================================================
# IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.widgets import Button

# ==============================================================================
# I. CLASE NAVEGADOR (VISOR INTERACTIVO)
# ==============================================================================
class GraficosNavegador:
    def __init__(self, df, lista_funciones):
        self.df = df
        self.funciones = lista_funciones
        self.indice = 0
        self.total = len(lista_funciones)
        self.fig = plt.figure(figsize=(14, 8))
        
        # Iniciar visualización
        self.actualizar_grafico()

    def anterior(self, event):
        self.indice = (self.indice - 1) % self.total
        self.actualizar_grafico()

    def siguiente(self, event):
        self.indice = (self.indice + 1) % self.total
        self.actualizar_grafico()

    def actualizar_grafico(self):
        # 1. Limpiar la figura completa para borrar el gráfico anterior
        self.fig.clear()
        
        # 2. Ejecutar la función de gráfico actual
        # NOTA: Pasamos guardar=False para que solo muestre en pantalla
        try:
            self.funciones[self.indice](self.df, guardar=False)
        except Exception as e:
            plt.text(0.5, 0.5, f"Error mostrando gráfico: {e}", ha='center')

        # 3. Ajustar márgenes para dejar espacio a los botones abajo
        plt.subplots_adjust(bottom=0.15)

        # 4. Añadir texto de paginación
        plt.figtext(0.5, 0.05, f'Gráfico {self.indice + 1} de {self.total}', 
                    ha='center', fontsize=12, fontweight='bold')

        # 5. Re-crear los botones (se borran con fig.clear(), hay que ponerlos de nuevo)
        ax_prev = plt.axes([0.3, 0.02, 0.15, 0.05])
        ax_next = plt.axes([0.55, 0.02, 0.15, 0.05])
        
        self.btn_prev = Button(ax_prev, '<< Anterior', color='lightblue', hovercolor='skyblue')
        self.btn_next = Button(ax_next, 'Siguiente >>', color='lightblue', hovercolor='skyblue')
        
        # Conectar eventos
        self.btn_prev.on_clicked(self.anterior)
        self.btn_next.on_clicked(self.siguiente)
        
        # Refrescar
        plt.draw()

# ==============================================================================
# II. FUNCIONES DE UTILIDAD
# ==============================================================================
def crear_carpeta_graficos():
    """Crea la carpeta 'graficos' si no existe y devuelve su ruta."""
    graficos_dir = Path('graficos')
    graficos_dir.mkdir(exist_ok=True)
    return graficos_dir

# ==============================================================================
# III. GRÁFICOS DEL 1 AL 13 (MODIFICADOS PARA EL VISOR)
# Nota: Se eliminaron plt.figure() y plt.show() interno para que funcionen en el visor.
# ==============================================================================

def grafico_1(df, graficos_dir=None, guardar=False):
    # No usamos plt.figure() aquí para que use la ventana del navegador
    sns.histplot(df['AveragePrice'], kde=True, bins=30, color='skyblue')
    plt.title('1. Distribución del Precio Promedio (AveragePrice)', fontsize=14, fontweight='bold')
    plt.xlabel('Precio Promedio ($)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '01_histograma_precio.png', dpi=300)

def grafico_2(df, graficos_dir=None, guardar=False):
    sns.boxplot(y=df['Total Volume'], color='salmon')
    plt.title('2. Boxplot de Volumen Total (Detección de Outliers)', fontsize=14, fontweight='bold')
    plt.ylabel('Volumen Total')
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '02_boxplot_volumen.png', dpi=300)

def grafico_3(df, graficos_dir=None, guardar=False):
    sns.boxplot(x='type', y='AveragePrice', data=df, palette={'conventional': 'orange', 'organic': 'green'})
    plt.title('3. AveragePrice por Tipo de Aguacate', fontsize=14, fontweight='bold')
    plt.xlabel('Tipo')
    plt.ylabel('Precio Promedio')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '03_boxplot_precio_por_tipo.png', dpi=300)

def grafico_4(df, graficos_dir=None, guardar=False):
    df_time = df.groupby('Date')['AveragePrice'].mean().reset_index()
    sns.lineplot(x='Date', y='AveragePrice', data=df_time, color='purple', linewidth=2)
    plt.title('4. Tendencia del Precio Promedio a lo largo del Tiempo', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '04_linea_precio_temporal.png', dpi=300)

def grafico_5(df, graficos_dir=None, guardar=False):
    df_vol_tipo = df.groupby(['Date', 'type'])['Total Volume'].sum().reset_index()
    for tipo in df_vol_tipo['type'].unique():
        subset = df_vol_tipo[df_vol_tipo['type'] == tipo]
        plt.plot(subset['Date'], subset['Total Volume'], label=tipo, linewidth=2)
    plt.title('5. Evolución del Volumen Total por Tipo', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '05_volumen_por_tipo.png', dpi=300)

def grafico_6(df, graficos_dir=None, guardar=False):
    bags_cols = ['Small Bags', 'Large Bags', 'XLarge Bags']
    bags_data = df[bags_cols].sum()
    plt.bar(bags_data.index, bags_data.values)
    plt.title('6. Distribución Total de Tipos de Bolsas', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '06_distribucion_bolsas.png', dpi=300)

def grafico_7(df, graficos_dir=None, guardar=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('7. Matriz de Correlación General', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '07_heatmap_correlacion_general.png', dpi=300)

def grafico_8(df, graficos_dir=None, guardar=False):
    top_regions = df.groupby('region')['AveragePrice'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=top_regions.values, y=top_regions.index)
    plt.title('8. Top 15 Regiones con Mayor Precio Promedio', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '08_top_regiones_precio.png', dpi=300)

def grafico_9(df, graficos_dir=None, guardar=False):
    plu_cols = ['4046', '4225', '4770']
    plu_data = df[plu_cols].sum()
    plt.bar(plu_cols, plu_data.values)
    plt.title('9. Distribución Total de Códigos PLU', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '09_distribucion_plu.png', dpi=300)

def grafico_10(df, graficos_dir=None, guardar=False):
    sns.countplot(y='region', data=df, order=df['region'].value_counts().index)
    plt.title('10. Conteo de Observaciones por Región', fontsize=14, fontweight='bold')
    # Ajustamos el tamaño de la fuente para que quepan las etiquetas
    plt.tick_params(axis='y', labelsize=8) 
    plt.grid(axis='x', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '10_conteo_regiones.png', dpi=300)

def grafico_11(df, graficos_dir=None, guardar=False):
    correlation_cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 
                        'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
    cols = [col for col in correlation_cols if col in df.columns]
    corr_matrix = df[cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('11. Heatmap Avanzado', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '11_heatmap_avanzado.png', dpi=300)

def grafico_12(df, graficos_dir=None, guardar=False):
    if 'year' not in df.columns or 'type' not in df.columns: return
    df['year_str'] = df['year'].astype(str)
    sns.violinplot(x='year_str', y='AveragePrice', hue='type', data=df, split=True)
    plt.title('12. Volatilidad de Precios por Año y Tipo', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '12_violin_plot.png', dpi=300)

def grafico_13(df, graficos_dir=None, guardar=False):
    if 'region' not in df.columns: return
    region_stats = df.groupby('region')['AveragePrice'].agg(
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index().sort_values(by='IQR', ascending=False)
    sns.barplot(x='IQR', y='region', data=region_stats.head(20))
    plt.title('13. Top 20 Regiones por IQR del Precio', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '13_iqr_regional.png', dpi=300)

# ==============================================================================
# FUNCIÓN PARA INICIAR EL VISOR (LLÁMALA DESDE EL MAIN)
# ==============================================================================
def iniciar_navegador(df):
    # Lista ordenada de todas las funciones que quieres mostrar
    lista_graficos = [
        grafico_1, grafico_2, grafico_3, grafico_4, grafico_5,
        grafico_6, grafico_7, grafico_8, grafico_9, grafico_10,
        grafico_11, grafico_12, grafico_13
    ]
    
    # Instanciar el navegador
    # IMPORTANTE: Guardamos la referencia en una variable (visor) para que no se borre
    visor = GraficosNavegador(df, lista_graficos)
    plt.show()