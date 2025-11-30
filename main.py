from src.carga_datos import cargar_datos
from src.exploracion import explorar_datos
from src.limpieza_datos import preparar_datos_inicial, detectar_outliers, eliminar_outliers, tratar_valores_nulos
from src.DefiniciónProblemas.DiseñoGráficos import iniciar_navegador
from src.transformacion_datos import transformar_preparar_datos, preparar_para_ml


def main():
    print("\n" + "="*70)
    print("🥑 ANÁLISIS DE DATOS - AGUACATES")
    print("="*70)
    
  
    print("\n📂 Paso 1: Cargando datos...")
    df = cargar_datos()

    if df is None or df.empty:
        print("❌ ERROR: No se pudo cargar el DataFrame. Terminando proceso.")
        return
    
    print(f"✓ Datos cargados: {df.shape[0]} filas × {df.shape[1]} columnas")

    print("\n🔍 Paso 2: Exploración inicial de datos...")
    explorar_datos(df)

   
    print("\n🧹 Paso 3: Limpieza de datos...")
    
    df = preparar_datos_inicial(df)
    
  
    print("\n Detectando y tratando Outliers...")
    detectar_outliers(df)
  
    df = eliminar_outliers(df, columnas=['AveragePrice', 'Total Volume'])

   
    print("\n Tratando valores nulos...")
    df = tratar_valores_nulos(df)  
    
    print(f"\n✅ Limpieza completada: {df.shape[0]} filas × {df.shape[1]} columnas")

    df.to_csv('data/avocado_limpio.csv', index=False)
    print("💾 Archivo 'data/avocado_limpio.csv' guardado correctamente")

    print("\n📊 Paso 4: Lanzando Visor de Gráficos Interactivo...")
    
    try:
        iniciar_navegador(df)
        print("\n✅ Visor cerrado.")
    except Exception as e:
        print(f"❌ Error al lanzar el visor: {e}")

    print("\n🔧 Paso 5: Transformación de datos para Machine Learning...")
    
    try:
        df_transformed, scaler_std, scaler_minmax, le_type = transformar_preparar_datos(df)
        
        X_train, X_test, y_train, y_test = preparar_para_ml(df_transformed)
        
        print("\n🎯 Dataset listo para Machine Learning:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - y_test: {y_test.shape}")
        
    except Exception as e:
        print(f"⚠️  Error en transformación: {e}")

    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"📊 Variables originales: {df.shape[1] - 2} columnas")
    print(f"📊 Variables creadas: total_bags, total_volume")
    print(f"📁 Archivo generado: data/avocado_limpio.csv")
    print("\n🚀 ¡Dataset listo para análisis y modelado!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()