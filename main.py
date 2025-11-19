from src.carga_datos import cargar_datos
from src.exploracion import explorar_datos
#from src.DefiniciónProblemas.DiseñoGráficos import visualizar_exploracion

from src.limpieza_datos import detectar_outliers, eliminar_outliers, tratar_valores_nulos

def main():

    df = cargar_datos()

    explorar_datos(df)

    detectar_outliers(df)

    eliminar_outliers(df, columnas=['AveragePrice', 'Total Volume'])

    tratar_valores_nulos(df)

    #visualizar_exploracion(df, guardar=True)


if __name__ == "__main__":
    main()