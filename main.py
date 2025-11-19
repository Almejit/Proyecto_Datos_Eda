from src.carga_datos import cargar_datos
from src.exploracion import explorar_datos
from src.limpieza_datos import detectar_outliers, eliminar_outliers, tratar_valores_nulos

# --- 🎯 Nuevo Importe del Módulo de Gráficos ---
# Importamos el módulo que contiene 'iniciar_navegador' y le damos un alias (dg)
from src.DefiniciónProblemas import DiseñoGráficos as dg 

def main():
    print("--- 🚀 Iniciando Proceso EDA Modular ---")

    # 1. Carga de Datos
    print("\n[1/5] 📥 Cargando datos...")
    df = cargar_datos()

    # Si la carga falla (df es None o vacío), detenemos el script
    if df is None or df.empty:
        print("❌ ERROR: No se pudo cargar el DataFrame. Terminando proceso.")
        return

    # 2. Exploración Inicial
    print("\n[2/5] 🔎 Explorando datos iniciales...")
    explorar_datos(df)

    # 3. Limpieza de Outliers
    print("\n[3/5] 🧼 Detectando y tratando Outliers...")
    # Detección
    detectar_outliers(df)
    # Eliminación de Outliers en columnas clave
    df = eliminar_outliers(df, columnas=['AveragePrice', 'Total Volume'])

    # 4. Tratamiento de Valores Nulos
    print("\n[4/5] 🧼 Tratando valores nulos...")
    tratar_valores_nulos(df)
    
    # 5. Visualización Interactiva
    print("\n[5/5] 🎨 Lanzando Visor de Gráficos Interactivo...")
    
    try:
        # Llamamos a la función que inicia el visor con botones Anterior/Siguiente
        dg.iniciar_navegador(df)
        
        print("\n✅ Proceso completado. Visor cerrado.")
    except AttributeError:
        print("❌ Error: Verifica que 'DiseñoGráficos.py' contenga la función 'iniciar_navegador'.")
    except Exception as e:
        print(f"❌ Error al lanzar el visor: {e}")


if __name__ == "__main__":
    main()