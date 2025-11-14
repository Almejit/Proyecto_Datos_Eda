from src.carga_datos import cargar_datos
from src.exploracion import explorar_datos

def main():

    df = cargar_datos()
    explorar_datos(df)


if __name__ == "__main__":
    main()