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
    def __init__(self, df, lista_funciones, guardar_automatico=True):
        self.df = df
        self.funciones = lista_funciones
        self.indice = 0
        self.total = len(lista_funciones)
        self.fig = plt.figure(figsize=(14, 8))
        self.guardar_automatico = guardar_automatico
        
        # Crear carpeta de gráficos si se va a guardar
        if self.guardar_automatico:
            self.graficos_dir = crear_carpeta_graficos()
        else:
            self.graficos_dir = None
        
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
        try:
            # Ahora pasamos guardar=True y graficos_dir para que se guarde
            self.funciones[self.indice](
                self.df, 
                graficos_dir=self.graficos_dir, 
                guardar=self.guardar_automatico
            )
        except Exception as e:
            plt.text(0.5, 0.5, f"Error mostrando gráfico: {e}", ha='center')

        # 3. Ajustar márgenes para dejar espacio a los botones abajo
        plt.subplots_adjust(bottom=0.15)

        # 4. Añadir texto de paginación
        status_text = f'Gráfico {self.indice + 1} de {self.total}'
        if self.guardar_automatico:
            status_text += ' ✓ Guardado'
        
        plt.figtext(0.5, 0.05, status_text, 
                    ha='center', fontsize=12, fontweight='bold')

        # 5. Re-crear los botones
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
# III. GRÁFICOS DEL 1 AL 13
# ==============================================================================

def grafico_1(df, graficos_dir=None, guardar=False):
    sns.histplot(df['AveragePrice'], kde=True, bins=30, color='skyblue')
    plt.title('1. Distribución del Precio Promedio (AveragePrice)', fontsize=14, fontweight='bold')
    plt.xlabel('Precio Promedio ($)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '01_histograma_precio.png', dpi=300, bbox_inches='tight')

def grafico_2(df, graficos_dir=None, guardar=False):
    sns.boxplot(y=df['Total Volume'], color='salmon')
    plt.title('2. Boxplot de Volumen Total (Detección de Outliers)', fontsize=14, fontweight='bold')
    plt.ylabel('Volumen Total')
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '02_boxplot_volumen.png', dpi=300, bbox_inches='tight')

def grafico_3(df, graficos_dir=None, guardar=False):
    sns.boxplot(x='type', y='AveragePrice', data=df, palette={'conventional': 'orange', 'organic': 'green'})
    plt.title('3. AveragePrice por Tipo de Aguacate', fontsize=14, fontweight='bold')
    plt.xlabel('Tipo')
    plt.ylabel('Precio Promedio')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '03_boxplot_precio_por_tipo.png', dpi=300, bbox_inches='tight')

def grafico_4(df, graficos_dir=None, guardar=False):
    df_time = df.groupby('Date')['AveragePrice'].mean().reset_index()
    sns.lineplot(x='Date', y='AveragePrice', data=df_time, color='purple', linewidth=2)
    plt.title('4. Tendencia del Precio Promedio a lo largo del Tiempo', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '04_linea_precio_temporal.png', dpi=300, bbox_inches='tight')

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
        plt.savefig(graficos_dir / '05_volumen_por_tipo.png', dpi=300, bbox_inches='tight')

def grafico_6(df, graficos_dir=None, guardar=False):
    bags_cols = ['Small Bags', 'Large Bags', 'XLarge Bags']
    bags_data = df[bags_cols].sum()
    plt.bar(bags_data.index, bags_data.values)
    plt.title('6. Distribución Total de Tipos de Bolsas', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '06_distribucion_bolsas.png', dpi=300, bbox_inches='tight')

def grafico_7(df, graficos_dir=None, guardar=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('7. Matriz de Correlación General', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '07_heatmap_correlacion_general.png', dpi=300, bbox_inches='tight')

def grafico_8(df, graficos_dir=None, guardar=False):
    top_regions = df.groupby('region')['AveragePrice'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=top_regions.values, y=top_regions.index)
    plt.title('8. Top 15 Regiones con Mayor Precio Promedio', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '08_top_regiones_precio.png', dpi=300, bbox_inches='tight')

def grafico_9(df, graficos_dir=None, guardar=False):
    plu_cols = ['4046', '4225', '4770']
    plu_data = df[plu_cols].sum()
    plt.bar(plu_cols, plu_data.values)
    plt.title('9. Distribución Total de Códigos PLU', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '09_distribucion_plu.png', dpi=300, bbox_inches='tight')

def grafico_10(df, graficos_dir=None, guardar=False):
    sns.countplot(y='region', data=df, order=df['region'].value_counts().index)
    plt.title('10. Conteo de Observaciones por Región', fontsize=14, fontweight='bold')
    plt.tick_params(axis='y', labelsize=8) 
    plt.grid(axis='x', alpha=0.3)
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '10_conteo_regiones.png', dpi=300, bbox_inches='tight')

def grafico_11(df, graficos_dir=None, guardar=False):
    correlation_cols = ['AveragePrice', 'Total Volume', '4046', '4225', '4770', 
                        'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
    cols = [col for col in correlation_cols if col in df.columns]
    corr_matrix = df[cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('11. Heatmap Avanzado', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '11_heatmap_avanzado.png', dpi=300, bbox_inches='tight')

def grafico_12(df, graficos_dir=None, guardar=False):
    if 'year' not in df.columns or 'type' not in df.columns: return
    df['year_str'] = df['year'].astype(str)
    sns.violinplot(x='year_str', y='AveragePrice', hue='type', data=df, split=True)
    plt.title('12. Volatilidad de Precios por Año y Tipo', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '12_violin_plot.png', dpi=300, bbox_inches='tight')

def grafico_13(df, graficos_dir=None, guardar=False):
    if 'region' not in df.columns: return
    region_stats = df.groupby('region')['AveragePrice'].agg(
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index().sort_values(by='IQR', ascending=False)
    sns.barplot(x='IQR', y='region', data=region_stats.head(20))
    plt.title('13. Top 20 Regiones por IQR del Precio', fontsize=14, fontweight='bold')
    if guardar and graficos_dir:
        plt.savefig(graficos_dir / '13_iqr_regional.png', dpi=300, bbox_inches='tight')

# ==============================================================================
# FUNCIÓN PARA INICIAR EL VISOR
# ==============================================================================
def iniciar_navegador(df, guardar_graficos=True):
    """
    Inicia el navegador interactivo de gráficos.
    
    Parámetros:
    - df: DataFrame con los datos
    - guardar_graficos: Si es True, guarda cada gráfico en la carpeta 'graficos'
    """
    lista_graficos = [
        grafico_1, grafico_2, grafico_3, grafico_4, grafico_5,
        grafico_6, grafico_7, grafico_8, grafico_9, grafico_10,
        grafico_11, grafico_12, grafico_13
    ]
    
    if guardar_graficos:
        print("📁 Los gráficos se guardarán automáticamente en la carpeta 'graficos/'")
    
    visor = GraficosNavegador(df, lista_graficos, guardar_automatico=guardar_graficos)
    plt.show()

# ==============================================================================
# FUNCIÓN ALTERNATIVA: GUARDAR TODOS LOS GRÁFICOS SIN VISOR
# ==============================================================================
def guardar_todos_los_graficos(df):
    """
    Guarda todos los gráficos directamente en la carpeta 'graficos' sin mostrar el visor.
    """
    print("\n📊 Generando y guardando todos los gráficos...")
    graficos_dir = crear_carpeta_graficos()
    
    lista_graficos = [
        grafico_1, grafico_2, grafico_3, grafico_4, grafico_5,
        grafico_6, grafico_7, grafico_8, grafico_9, grafico_10,
        grafico_11, grafico_12, grafico_13
    ]
    
    for i, func_grafico in enumerate(lista_graficos, 1):
        plt.figure(figsize=(14, 8))
        try:
            func_grafico(df, graficos_dir=graficos_dir, guardar=True)
            print(f"  ✓ Gráfico {i}/13 guardado")
        except Exception as e:
            print(f"  ✗ Error en gráfico {i}: {e}")
        plt.close()
    
    print(f"\n✅ Todos los gráficos guardados en: {graficos_dir.absolute()}")