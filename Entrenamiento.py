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

    try:
        housing = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo '{archivo_csv}'")
        sys.exit(1)

    #PRE-PROCESAMIENTO

    # Filtrar las columnas relevantes
    housing = housing.filter(["price", "bed", "bath", "acre_lot", "house_size"])

    # Eliminar filas con valores faltantes
    housing = housing.dropna()

    # Remover valores atípicos en varias columnas
    for col in ['price', 'bed', 'bath', 'acre_lot', 'house_size']:
        housing = remover_valores_atipicos(housing, col)

    # Seleccionar las columnas necesarias
    columnas_caracteristicas = ["bed", "bath", "acre_lot", "house_size"]
    X = housing[columnas_caracteristicas]

    y = housing['price']

    # Escalar características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # DIVISIÓN DE DATOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=101)

    #ENTRENAMIENTO DEL MODELOjot
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    ruta_modelo_guardado = 'modelo_entrenado.pkl'
    joblib.dump(modelo, ruta_modelo_guardado)
    print(f"Modelo entrenado guardado en '{ruta_modelo_guardado}'\n")
    print(f"Modelo entrenado guardado en '{ruta_modelo_guardado}'")

