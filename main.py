from src.carga_datos import cargar_datos
# ¡Importaciones Actualizadas para incluir las dos funciones del EDA avanzado!
from src.exploracion import explorar_datos, analisis_eda, deteccion_patrones
from src.limpieza_datos import detectar_outliers, eliminar_outliers, tratar_valores_nulos
from src.DefiniciónProblemas.DiseñoGráficos import iniciar_navegador
from src.transformacion_datos import transformar_preparar_datos, preparar_para_ml

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
    
    # 5. LLAMADA AL ANÁLISIS EXPLORATORIO AVANZADO (EDA)
    analisis_eda(df) 
    
    # 6. LLAMADA A LA DETECCIÓN DE PATRONES/CLUSTERING
    deteccion_patrones(df) # <-- ¡Nuevo paso!
    
    print("\nLanzando Visor de Gráficos Interactivo...")
    
    try:
        iniciar_navegador(df)
        print("\n✅ Proceso completado. Visor cerrado.")
    except Exception as e:
        print(f"❌ Error al lanzar el visor: {e}")
        
    
    df_transformed, scaler_std, scaler_minmax, le_type = transformar_preparar_datos(df)
    
    X_train, X_test, y_train, y_test = preparar_para_ml(df_transformed)
    
    print("\n\n🎯 Dataset listo para aplicar algoritmos de ML!")
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de y_train: {y_train.shape}")


if __name__ == "__main__":
    main()