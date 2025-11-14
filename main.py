from src.carga_datos import cargar_datos
from src.exploracion import explorar_datos
from src.DefiniciónProblemas.DiseñoGráficos import visualizar_exploracion

def main():

    df = cargar_datos()
    explorar_datos(df)

    visualizar_exploracion(df, guardar=True)


if __name__ == "__main__":
    main()