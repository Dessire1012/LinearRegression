import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def remover_valores_atipicos(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Uso: py LinearRegressionGrapherScript.py <archivo_csv>")
        sys.exit(1)

    archivo_csv = sys.argv[1]

    # Cargar los datos desde el archivo CSV proporcionado en línea de comandos
    try:
        housing = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo '{archivo_csv}'")
        sys.exit(1)

    # Filtrar las columnas relevantes
    housing = housing.filter(["price", "bed", "bath", "acre_lot", "house_size"])

    # Eliminar filas con valores faltantes
    housing = housing.dropna()

    # Remover valores atípicos en varias columnas
    for col in ['bath', 'price', 'acre_lot', 'bed', 'house_size']:
        housing = remover_valores_atipicos(housing, col)

    # Definir variable objetivo
    train_target_label = housing['price']

    # Seleccionar columnas de características
    columnas_caracteristicas = ["bed", "bath", "acre_lot", "house_size"]
    training_sample_df = housing[columnas_caracteristicas]

    # Escalar características
    scaler = StandardScaler()
    training_sample_df = scaler.fit_transform(training_sample_df)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(training_sample_df, train_target_label, random_state=101,
                                                        train_size=0.8)

    # Instanciar y entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Guardar el modelo entrenado en un archivo
    ruta_modelo_guardado = 'modelo_entrenado.pkl'
    joblib.dump(modelo, ruta_modelo_guardado)
    print(f"Modelo entrenado guardado en '{ruta_modelo_guardado}'\n")
    print(f"Modelo entrenado guardado en '{ruta_modelo_guardado}'")

