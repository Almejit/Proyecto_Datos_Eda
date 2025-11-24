from src.carga_datos import cargar_datos
from src.exploracion import explorar_datos
from src.limpieza_datos import detectar_outliers, eliminar_outliers, tratar_valores_nulos
from src.DefiniciónProblemas.DiseñoGráficos import iniciar_navegador


def main():
   
    df = cargar_datos()

    if df is None or df.empty:
        print("❌ ERROR: No se pudo cargar el DataFrame. Terminando proceso.")
        return

    explorar_datos(df)

  
    print("\n Detectando y tratando Outliers...")
    detectar_outliers(df)
    df = eliminar_outliers(df, columnas=['AveragePrice', 'Total Volume'])


    print("\n Tratando valores nulos...")
    df = tratar_valores_nulos(df)  
    
    print("\nLanzando Visor de Gráficos Interactivo...")
    
    try:
        iniciar_navegador(df)
        print("\n✅ Proceso completado. Visor cerrado.")
    except Exception as e:
        print(f"❌ Error al lanzar el visor: {e}")


if __name__ == "__main__":
    main()