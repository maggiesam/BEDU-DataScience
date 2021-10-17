import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#función de mapa de correlaciones
def mapa():
    df = pd.read_csv("https://raw.githubusercontent.com/maggiesam/BEDU-DataScience/main/Datasets/dataframe-junto.csv")
    correlation_mat = df.corr()
    fig = plt.figure(figsize=(10,10))
    sns.heatmap(correlation_mat, annot = True)
    plt.show()


#función de regresion
def regresion_lineal(planta=0, grado=1, meses=False, n_mes=1,  prueba=0.2, semilla=50):

    #carga los datos directamente del repositorio
    df = pd.read_csv("https://raw.githubusercontent.com/maggiesam/BEDU-DataScience/main/Datasets/dataframe-junto.csv")

    #diferencia entre las plantas
    lista=["maiz","frijol","trigo"]
    producto = df[df["producto"] == lista[planta]]

    #añade una columna de porcentaje de cosecha
    producto["porcentaje"] = producto["Cosechada_ha"]*100/producto["Sembrada_ha"]

    #activa el mes que queremos explorar
    if meses:
        producto = producto[producto["Mes"] == n_mes]

    #elimina las columnas no relevantes para la regresion
    nuevo = producto.drop(["producto", "ENTIDAD", "Año", "Mes", "Tipo_sequia", "Unnamed: 0", "perdida_ha", "Cosechada_ha"], axis=1)
    nuevo = nuevo.reset_index(drop=True)
    X = nuevo.drop("porcentaje",axis=1)
    Y = nuevo["porcentaje"]

    #selecciona los tamaños de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = prueba, random_state=semilla)

    #regresión lineal
    if grado == 1:
      print("función de regresion lineal de "+lista[planta])
      lin_model = LinearRegression()
      lin_model.fit(X_train, Y_train)
      y_train_predict = lin_model.predict(X_train)
      MSE = mean_squared_error(Y_train,y_train_predict)
      print("Entrenamiento: MSE ="+str(MSE))

      y_test_predict = lin_model.predict(X_test)
      MSE = (mean_squared_error(Y_test, y_test_predict))
      print("Pruebas: MSE ="+str(MSE))
      df_predicciones = pd.DataFrame({'valor_real':Y_test, 'prediccion':y_test_predict})
      df_predicciones = df_predicciones.reset_index(drop = True)
      return df_predicciones

    #regresión polinomica
    if grado >= 2:
        print("función de regresion polinomica de grado " + str(grado) + " de " + lista[planta])
        poly_model = LinearRegression()
        poly = PolynomialFeatures(degree=grado)

        Xpolytrain = poly.fit_transform(X_train)
        Xpolytest = poly.fit_transform(X_test)

        poly_model.fit(Xpolytrain, Y_train)
        y_train_predict = poly_model.predict(Xpolytrain)

        MSE = mean_squared_error(Y_train,y_train_predict)
        print("Entrenamiento: MSE ="+str(MSE))

        y_test_predict = poly_model.predict(Xpolytest)
        MSE = (mean_squared_error(Y_test, y_test_predict))
        print("Pruebas: MSE ="+str(MSE))

        df_predicciones = pd.DataFrame({'valor_real':Y_test, 'prediccion':y_test_predict})
        df_predicciones = df_predicciones.reset_index(drop = True)
        df_predicciones.head(10)
        return df_predicciones
