import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#librerias de regresion lineal

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


#Funciones de separación de entrenamiento, validación y prueba.
def particionar(entradas,salidas,porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
  temp_size = porcentaje_validacion + porcentaje_prueba
  x_train, x_temp, y_train, y_temp = train_test_split(entradas,salidas,test_size=temp_size)
  if(porcentaje_validacion > 0):
    test_size = porcentaje_prueba/temp_size
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_size)
  else:
    return [x_train, None, x_temp, y_train, None, y_temp]
  return [x_train, x_val, x_temp, y_train, y_val, y_temp]

#Funciones de separación de datasets con K-Fold (el usuario debe poner el K, si K = 1 debe generar un Leave-One-Out Cross Validation).
def kfold(k):
  kfold = KFold(k,True,random_seed=48)
  return kfold

#Funciones de evaluación con matriz de confusión.
def conf_matrix(y_esperados, y_predichos):
  matrix = confusion_matrix(y_esperados, y_predichos)
  return matrix

#Funciones de obtención de Precisión (Accuracy), Sensibilidad y Especificidad.
def conf_matrix(y_esperados, y_predichos): 
  matrix = confusion_matrix(y_esperados, y_predichos)
  return matrix 

def parameters(matrix):
  (TP, FN, FP, TN) = np.ravel(matrix, order = 'C')
  return TP, FN, FP, TN

def accuracy(TP, FN, FP, TN):  #exactitud
  a = (TP + TN)/(TP+TN+FP+FN)
  return a

def sensitivity(TP, FN, FP, TN): #sensibilidad
  s = TP/(TP +FN)
  return s

def specificity(TP, FN, FP, TN): #especificidad
  sp = TN/(TN + FP)
  return sp

def precision(TP, FN, FP, TN): #precisión
  p = TP/(TP+FP)
  return p

#Funciones que comparen dos clasificadores:
#Obtengas precisión, sensibilidad y especificidad del clasificador 1
#Obtengas precisión, sensibilidad y especificidad del clasificador 2
def comparison(matrix1, matrix2): #la función toma dos matrices de confusión
  TP, FN, FP, TN = parameters(matrix1)
  TP2, FN2, FP2, TN2 = parameters(matrix2)

  #valores para la matriz del modelo 1
  a1 = accuracy(TP, FN, FP, TN)
  s1 = sensitivity(TP, FN, FP, TN)
  sp1 = specificity(TP, FN, FP, TN)
  p1 = precision(TP, FN, FP, TN)
  
  print("Modelo 1:") 
  print(f"Exactitud: {a1}") 
  print(f"Sensibilidad: {s1}") 
  print(f"Especificidad: {sp1}") 
  print(f"Precisión: {p1}") 
  
  print("\n") 

  print("Modelo 2:") 
  print(f"Exactitud: {a2}") 
  print(f"Sensibilidad: {s2}") 
  print(f"Especificidad: {sp2}") 
  print(f"Precisión: {p2}") 
  
  print("\n") 
  
  #valores para la matriz del modelo 2
  a2 = accuracy(TP2, FN2, FP2, TN2)
  s2 = sensitivity(TP2, FN2, FP2, TN2)
  sp2 = specificity(TP2, FN2, FP2, TN2)
  p2 = precision(TP2, FN2, FP2, TN2)
  
  #comparacion entre parámetros
  if a1 > a2: #exactitud
    print("El clasificador 1 es mejor que el clasificador 2 en términos de exactitud \n")
  else:
    print("El clasificador 2 es mejor que el clasificador 2 en términos de exactitud \n")

  if s1 > s2: #sensibilidad
    print("El clasificador 1 es mejor que el clasificador 2 en términos de sensibilidad \n")
  else:
    print("El clasificador 2 es mejor que el clasificador 2 en términos de sensibilidad \n")

  if sp1 > sp2: #especificidad
    print("El clasificador 1 es mejor que el clasificador 2 en términos de especificidad \n")
  else:
    print("El clasificador 2 es mejor que el clasificador 2 en términos de especificidad \n") 

  if p1 > p2: #precisión
    print("El clasificador 1 es mejor que el clasificador 2 en términos de precisión \n")
  else:
    print("El clasificador 2 es mejor que el clasificador 2 en términos de precisión \n")

#Funciones de evaluación multiclase.

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

      
      
      
      
 #Forecasting - Efrain





################################################################################
#                               skforecast                                     #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8

import typing
from typing import Union, Dict, List, Tuple
import warnings
import logging
import numpy as np
import pandas as pd
import sklearn
import tqdm


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_absolute_percentage_error


logging.basicConfig(
    format = '%(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


################################################################################
#                             ForecasterAutoreg                                #
################################################################################

class ForecasterAutoreg():
    '''
    This class turns any regressor compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster.
    
    Parameters
    ----------
    regressor : regressor compatible with the scikit-learn API
        An instance of a regressor compatible with the scikit-learn API.
        
    lags : int, list, 1D np.array, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags` (included).
            `list` or `np.array`: include only lags present in `lags`.
    
    Attributes
    ----------
    regressor : regressor compatible with the scikit-learn API
        An instance of a regressor compatible with the scikit-learn API.
        
    lags : 1D np.array
        Lags used as predictors.
        
    max_lag : int
        Maximum value of lag included in lags.
        
    window_size: int
        Size of the window needed to create the predictors. It is equal to
        `max_lag`.
        
    last_window : 1D np.ndarray
        Last time window the forecaster has seen when trained. It stores the
        values needed to calculate the lags used to predict the next `step`
        after the training data.
        
    included_exog : bool
        If the forecaster has been trained using exogenous variable/s.
        
    exog_type : type
        Type used for the exogenous variable/s: pd.Series, pd.DataFrame or np.ndarray.
            
    exog_shape : tuple
        Shape of exog used in training.
        
    in_sample_residuals: np.ndarray
        Residuals of the model when predicting training data. Only stored up to
        1000 values.
        
    out_sample_residuals: np.ndarray
        Residuals of the model when predicting non training data. Only stored
        up to 1000 values.
    fitted: Bool
        Tag to identify if the estimator is fitted.
     
    '''
    
    def __init__(self, regressor, lags: Union[int, np.ndarray, list]) -> None:
        
        self.regressor            = regressor
        self.last_window          = None
        self.included_exog        = False
        self.exog_type            = None
        self.exog_shape           = None
        self.in_sample_residuals  = None
        self.out_sample_residuals = None
        self.fitted               = False
        
        if isinstance(lags, int) and lags < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags)
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                f"`lags` argument must be `int`, `1D np.ndarray`, `range` or `list`. "
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags)
        self.window_size = self.max_lag
                
        
    def __repr__(self) -> str:
        '''
        Information displayed when a ForecasterAutoreg object is printed.
        '''

        info =    "=======================" \
                + "ForecasterAutoreg" \
                + "=======================" \
                + "\n" \
                + "Regressor: " + str(self.regressor) \
                + "\n" \
                + "Lags: " + str(self.lags) \
                + "\n" \
                + "Window size: " + str(self.window_size) \
                + "\n" \
                + "Exogenous variable: " + str(self.included_exog) + ', ' + str(self.exog_type) \
                + "\n" \
                + "Parameters: " + str(self.regressor.get_params())

        return info

    
    
    def create_lags(self, y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        '''       
        Transforms a time series into a 2D array and a 1D array where each value
        of `y` is associated with the lags that precede it.
        
        Notice that the returned matrix X_data, contains the lag 1 in the
        first column, the lag 2 in the second column and so on.
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.
        Returns 
        -------
        X_data : 2D np.ndarray, shape (samples, len(self.lags))
            2D array with the lag values (predictors).
        
        y_data : 1D np.ndarray, shape (nº observaciones - max(seld.lags),)
            Values of the time series related to each row of `X_data`.
            
        '''
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)        
        
        if self.max_lag > len(y):
            raise Exception(
                f"Maximum lag can't be higher than `y` length. "
                f"Got maximum lag={self.max_lag} and `y` length={len(y)}."
            )
            
        n_splits = len(y) - self.max_lag
        X_data  = np.full(shape=(n_splits, self.max_lag), fill_value=np.nan, dtype=float)
        y_data  = np.full(shape=(n_splits, 1), fill_value=np.nan, dtype= float)

        for i in range(n_splits):
            X_index = np.arange(i, self.max_lag + i)
            y_index = [self.max_lag + i]

            X_data[i, :] = y[X_index]
            y_data[i]    = y[y_index]
            
        X_data = X_data[:, -self.lags]
        y_data = y_data.ravel()
            
        return X_data, y_data


    def create_train_X_y(self, y: Union[np.ndarray, pd.Series],
                         exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None
                         ) -> Tuple[np.array, np.array]:
        '''
        Create training matrices X, y
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.
            
        exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and should be aligned so that y[i] is
            regressed on exog[i].
        Returns 
        -------
        X_train : 2D np.ndarray, shape (len(y) - self.max_lag, len(self.lags))
            2D array with the training values (predictors).
            
        y_train : 1D np.ndarray, shape (len(y) - self.max_lag,)
            Values (target) of the time series related to each row of `X_train`.
        
        '''
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is not None:
            self._check_exog(exog=exog)
            exog = self._preproces_exog(exog=exog)
            self.included_exog = True
            self.exog_shape = exog.shape
            
            if exog.shape[0] != len(y):
                raise Exception(
                    f"`exog` must have same number of samples as `y`"
                )
                
        X_train, y_train = self.create_lags(y=y)
    
        if exog is not None:
            # The first `self.max_lag` positions have to be removed from exog
            # since they are not in X_train.
            X_train = np.column_stack((X_train, exog[self.max_lag:,]))
                        
        return X_train, y_train

        
    def fit(self, y: Union[np.ndarray, pd.Series],
            exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None) -> None:
        '''
        Training ForecasterAutoreg
        
        Parameters
        ----------        
        y : 1D np.ndarray, pd.Series
            Training time series.
            
        exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s. Must have the same
            number of observations as `y` and should be aligned so that y[i] is
            regressed on exog[i].
        Returns 
        -------
        self : ForecasterAutoreg
            Trained ForecasterAutoreg
        
        '''
        
        # Reset values in case the forecaster has already been fitted before.
        self.included_exog = False
        self.exog_type     = None
        self.exog_shape    = None
        
        self._check_y(y=y)
        y = self._preproces_y(y=y)
        
        if exog is not None:
            self._check_exog(exog=exog)
            self.exog_type = type(exog)
            exog = self._preproces_exog(exog=exog)
            self.included_exog = True
            self.exog_shape = exog.shape
            
            if exog.shape[0] != len(y):
                raise Exception(
                    f"`exog` must have same number of samples as `y`"
                )
                
        
        X_train, y_train = self.create_train_X_y(y=y, exog=exog)
        
        self.regressor.fit(X=X_train, y=y_train)
        self.fitted = True            
        residuals = y_train - self.regressor.predict(X_train)
            
        if len(residuals) > 1000:
            # Only up to 1000 residuals are stored
            residuals = np.random.choice(a=residuals, size=1000, replace=False)                                              
        self.in_sample_residuals = residuals
        
        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        self.last_window = y_train[-self.max_lag:].copy()
        
            
    def predict(self, steps: int, last_window: Union[np.ndarray, pd.Series]=None,
                exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None) -> np.ndarray:
        '''
        Iterative process in which, each prediction, is used as a predictor
        for the next step.
        
        Parameters
        ----------
        steps : int
            Number of future steps predicted.
            
        last_window : 1D np.ndarray, pd.Series, shape (, max_lag), default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
        Returns 
        -------
        predictions : 1D np.array, shape (steps,)
            Values predicted.
            
        '''

        if not self.fitted:
            raise Exception(
                'This Forecaster instance is not fitted yet. Call `fit` with appropriate arguments before using this it.'
            )
        
        if steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )
        
        if exog is None and self.included_exog:
            raise Exception(
                f"Forecaster trained with exogenous variable/s. "
                f"Same variable/s must be provided in `predict()`."
            )
            
        if exog is not None and not self.included_exog:
            raise Exception(
                f"Forecaster trained without exogenous variable/s. "
                f"`exog` must be `None` in `predict()`."
            )
        
        if exog is not None:
            self._check_exog(
                exog=exog, ref_type = self.exog_type, ref_shape=self.exog_shape
            )
            exog = self._preproces_exog(exog=exog)
            if exog.shape[0] < steps:
                raise Exception(
                    f"`exog` must have at least as many values as `steps` predicted."
                )
     
        if last_window is not None:
            self._check_last_window(last_window=last_window)
            last_window = self._preproces_last_window(last_window=last_window)
            if last_window.shape[0] < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
        else:
            last_window = self.last_window.copy()
            
        predictions = np.full(shape=steps, fill_value=np.nan)

        for i in range(steps):
            X = last_window[-self.lags].reshape(1, -1)
            if exog is None:
                prediction = self.regressor.predict(X)
            else:
                prediction = self.regressor.predict(
                                np.column_stack((X, exog[i,].reshape(1, -1)))
                             )
            predictions[i] = prediction.ravel()[0]

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window = np.append(last_window[1:], prediction)

        return predictions
    
    
    def _estimate_boot_interval(self, steps: int,
                                last_window: Union[np.ndarray, pd.Series]=None,
                                exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
                                interval: list=[5, 95], n_boot: int=500,
                                in_sample_residuals: bool=True) -> np.ndarray:
        '''
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. This method only returns prediction intervals.
        See predict_intervals() to calculate both, predictions and intervals.
        
        Parameters
        ----------   
        steps : int
            Number of future steps predicted.
            
        last_window : 1D np.ndarray, pd.Series, shape (, max_lag), default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        n_boot: int, default `100`
            Number of bootstrapping iterations used to estimate prediction
            intervals.
            
        interval: list, default `[5, 100]`
            Confidence of the prediction interval estimated. Sequence of percentiles
            to compute, which must be between 0 and 100 inclusive.
            
        in_sample_residuals: bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction intervals. If `False`, out of
            sample residuals are used. In the latter case, the user shoud have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
            
        Returns 
        -------
        predicction_interval : np.array, shape (steps, 2)
            Interval estimated for each prediction by bootstrapping.
        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
            
        '''
        
        if steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )
            
        if not in_sample_residuals and self.out_sample_residuals is None:
            raise Exception(
                ('out_sample_residuals is empty. In order to estimate prediction '
                'intervals using out of sample residuals, the user shoud have '
                'calculated and stored the residuals within the forecaster (see'
                '`set_out_sample_residuals()`.')
            )

        if exog is None and self.included_exog:
            raise Exception(
                f"Forecaster trained with exogenous variable/s. "
                f"Same variable/s must be provided in `predict()`."
            )

        if exog is not None and not self.included_exog:
            raise Exception(
                f"Forecaster trained without exogenous variable/s. "
                f"`exog` must be `None` in `predict()`."
            )

        if exog is not None:
            self._check_exog(
                exog=exog, ref_type = self.exog_type, ref_shape=self.exog_shape
            )
            exog = self._preproces_exog(exog=exog)
            if exog.shape[0] < steps:
                raise Exception(
                    f"`exog` must have at least as many values as `steps` predicted."
                )

        if last_window is not None:
            self._check_last_window(last_window=last_window)
            last_window = self._preproces_last_window(last_window=last_window)
            if last_window.shape[0] < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
        else:
            last_window = self.last_window.copy()

        boot_predictions = np.full(
                                shape      = (steps, n_boot),
                                fill_value = np.nan,
                                dtype      = float
                           )

        for i in range(n_boot):

            # In each bootstraping iteration the initial last_window and exog 
            # need to be restored.
            last_window_boot = last_window.copy()
            if exog is not None:
                exog_boot = exog.copy()
            else:
                exog_boot = None
                
            if in_sample_residuals:
                residuals = self.in_sample_residuals
            else:
                residuals = self.out_sample_residuals

            sample_residuals = np.random.choice(
                                    a       = residuals,
                                    size    = steps,
                                    replace = True
                               )

            for step in range(steps):  
                prediction = self.predict(
                                steps       = 1,
                                last_window = last_window_boot,
                                exog        = exog_boot
                             )
                
                prediction_with_residual  = prediction + sample_residuals[step]
                boot_predictions[step, i] = prediction_with_residual

                last_window_boot = np.append(
                                    last_window_boot[1:],
                                    prediction_with_residual
                                   )
                
                if exog is not None:
                    exog_boot = exog_boot[1:]
                            
        prediction_interval = np.percentile(boot_predictions, q=interval, axis=1)
        prediction_interval = prediction_interval.transpose()
        
        return prediction_interval
    
        
    def predict_interval(self, steps: int, last_window: Union[np.ndarray, pd.Series]=None,
                         exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
                         interval: list=[5, 95], n_boot: int=500,
                         in_sample_residuals: bool=True) -> np.ndarray:
        '''
        Iterative process in which, each prediction, is used as a predictor
        for the next step and bootstrapping is used to estimate prediction
        intervals. Both, predictions and intervals, are returned.
        
        Parameters
        ---------- 
        steps : int
            Number of future steps predicted.
            
        last_window : 1D np.ndarray, pd.Series, shape (, max_lag), default `None`
            Values of the series used to create the predictors (lags) need in the 
            first iteration of predictiont (t + 1).
    
            If `last_window = None`, the values stored in` self.last_window` are
            used to calculate the initial predictors, and the predictions start
            right after training data.
            
        exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
            Exogenous variable/s included as predictor/s.
            
        interval: list, default `[5, 100]`
            Confidence of the prediction interval estimated. Sequence of percentiles
            to compute, which must be between 0 and 100 inclusive.
            
        n_boot: int, default `500`
            Number of bootstrapping iterations used to estimate prediction
            intervals.
            
        in_sample_residuals: bool, default `True`
            If `True`, residuals from the training data are used as proxy of
            prediction error to create prediction intervals. If `False`, out of
            sample residuals are used. In the latter case, the user shoud have
            calculated and stored the residuals within the forecaster (see
            `set_out_sample_residuals()`).
        Returns 
        -------
        predictions : np.array, shape (steps, 3)
            Values predicted by the forecaster and their estimated interval.
            Column 0 = predictions
            Column 1 = lower bound interval
            Column 2 = upper bound interval
        Notes
        -----
        More information about prediction intervals in forecasting:
        https://otexts.com/fpp2/prediction-intervals.html
        Forecasting: Principles and Practice (2nd ed) Rob J Hyndman and
        George Athanasopoulos.
            
        '''
        
        if steps < 1:
            raise Exception(
                f"`steps` must be integer greater than 0. Got {steps}."
            )
            
        if not in_sample_residuals and self.out_sample_residuals is None:
            raise Exception(
                ('out_sample_residuals is empty. In order to estimate prediction '
                'intervals using out of sample residuals, the user shoud have '
                'calculated and stored the residuals within the forecaster (see'
                '`set_out_sample_residuals()`.')
            )
        
        if exog is None and self.included_exog:
            raise Exception(
                f"Forecaster trained with exogenous variable/s. "
                f"Same variable/s must be provided in `predict()`."
            )
            
        if exog is not None and not self.included_exog:
            raise Exception(
                f"Forecaster trained without exogenous variable/s. "
                f"`exog` must be `None` in `predict()`."
            )
        
        if exog is not None:
            self._check_exog(
                exog=exog, ref_type = self.exog_type, ref_shape=self.exog_shape
            )
            exog = self._preproces_exog(exog=exog)
            if exog.shape[0] < steps:
                raise Exception(
                    f"`exog` must have as many values as `steps` predicted."
                )
     
        if last_window is not None:
            self._check_last_window(last_window=last_window)
            last_window = self._preproces_last_window(last_window=last_window)
            if last_window.shape[0] < self.max_lag:
                raise Exception(
                    f"`last_window` must have as many values as as needed to "
                    f"calculate the maximum lag ({self.max_lag})."
                )
        else:
            last_window = self.last_window.copy()
        
        # Since during predict() `last_window` and `exog` are modified, the
        # originals are stored to be used later
        last_window_original = last_window.copy()
        if exog is not None:
            exog_original = exog.copy()
        else:
            exog_original = exog
            
        predictions = self.predict(
                            steps       = steps,
                            last_window = last_window,
                            exog        = exog
                      )

        predictions_interval = self._estimate_boot_interval(
                                    steps       = steps,
                                    last_window = last_window_original,
                                    exog        = exog_original,
                                    interval    = interval,
                                    n_boot      = n_boot,
                                    in_sample_residuals = in_sample_residuals
                                )
        
        predictions = np.column_stack((predictions, predictions_interval))

        return predictions

    
    def _check_y(self, y: Union[np.ndarray, pd.Series]) -> None:
        '''
        Raise Exception if `y` is not 1D `np.ndarray` or `pd.Series`.
        
        Parameters
        ----------        
        y : np.ndarray, pd.Series
            Time series values
        '''
        
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise Exception('`y` must be `1D np.ndarray` or `pd.Series`.')
        elif isinstance(y, np.ndarray) and y.ndim != 1:
            raise Exception(
                f"`y` must be `1D np.ndarray` o `pd.Series`, "
                f"got `np.ndarray` with {y.ndim} dimensions."
            )
            
        return
    
    
    def _check_last_window(self, last_window: Union[np.ndarray, pd.Series]) -> None:
        '''
        Raise Exception if `last_window` is not 1D `np.ndarray` or `pd.Series`.
        
        Parameters
        ----------        
        last_window : np.ndarray, pd.Series
            Time series values
        '''
        
        if not isinstance(last_window, (np.ndarray, pd.Series)):
            raise Exception('`last_window` must be `1D np.ndarray` or `pd.Series`.')
        elif isinstance(last_window, np.ndarray) and last_window.ndim != 1:
            raise Exception(
                f"`last_window` must be `1D np.ndarray` o `pd.Series`, "
                f"got `np.ndarray` with {last_window.ndim} dimensions."
            )
            
        return
        
        
    def _check_exog(self, exog: Union[np.ndarray, pd.Series, pd.DataFrame], 
                    ref_type: type=None, ref_shape: tuple=None) -> None:
        '''
        Raise Exception if `exog` is not `np.ndarray`, `pd.Series` or `pd.DataFrame`.
        If `ref_shape` is provided, raise Exception if `ref_shape[1]` do not match
        `exog.shape[1]` (number of columns).
        
        Parameters
        ----------        
        exog : np.ndarray, pd.Series, pd.DataFrame
            Exogenous variable/s included as predictor/s.
        exog_type : type, default `None`
            Type of reference for exog.
            
        exog_shape : tuple, default `None`
            Shape of reference for exog.
        '''
            
        if not isinstance(exog, (np.ndarray, pd.Series, pd.DataFrame)):
            raise Exception('`exog` must be `np.ndarray`, `pd.Series` or `pd.DataFrame`.')
            
        if isinstance(exog, np.ndarray) and exog.ndim > 2:
            raise Exception(
                    f" If `exog` is `np.ndarray`, maximum allowed dim=2. "
                    f"Got {exog.ndim}."
                )
            
        if ref_type is not None:
            
            if ref_type == pd.Series:
                if isinstance(exog, pd.Series):
                    return
                elif isinstance(exog, np.ndarray) and exog.ndim == 1:
                    return
                elif isinstance(exog, np.ndarray) and exog.shape[1] == 1:
                    return
                else:
                    raise Exception(
                        f"`exog` must be: `pd.Series`, `np.ndarray` with 1 dimension "
                        f"or `np.ndarray` with 1 column in the second dimension. "
                        f"Got `np.ndarray` with {exog.shape[1]} columns."
                    )
                    
            if ref_type == np.ndarray:
                if exog.ndim == 1 and ref_shape[1] == 1:
                    return
                elif exog.ndim == 1 and ref_shape[1] > 1:
                    raise Exception(
                        f"`exog` must have {ref_shape[1]} columns. "
                        f"Got `np.ndarray` with 1 dimension or `pd.Series`."
                    )
                elif ref_shape[1] != exog.shape[1]:
                    raise Exception(
                        f"`exog` must have {ref_shape[1]} columns. "
                        f"Got `np.ndarray` with {exog.shape[1]} columns."
                    )     
                    
            if ref_type == pd.DataFrame:
                if ref_shape[1] != exog.shape[1]:
                    raise Exception(
                        f"`exog` must have {ref_shape[1]} columns. "
                        f"Got `pd.DataFrame` with {exog.shape[1]} columns."
                    )
        return
    
        
    def _preproces_y(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        
        '''
        Transforms `y` to 1D `np.ndarray` if it is `pd.Series`.
        
        Parameters
        ----------        
        y :1D np.ndarray, pd.Series
            Time series values
        Returns 
        -------
        y: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(y, pd.Series):
            return y.to_numpy(copy=True)
        else:
            return y
            
        
    def _preproces_last_window(self, last_window: Union[np.ndarray, pd.Series]) -> np.ndarray:
        
        '''
        Transforms `last_window` to 1D `np.ndarray` if it is `pd.Series`.
        
        Parameters
        ----------        
        last_window :1D np.ndarray, pd.Series
            Time series values
        Returns 
        -------
        last_window: 1D np.ndarray, shape(samples,)
        '''
        
        if isinstance(last_window, pd.Series):
            return last_window.to_numpy(copy=True)
        else:
            return last_window
        
        
    def _preproces_exog(self, exog: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        
        '''
        Transforms `exog` to `np.ndarray` if it is `pd.Series` or `pd.DataFrame`.
        If 1D `np.ndarray` reshape it to (n_samples, 1)
        
        Parameters
        ----------        
        exog : np.ndarray, pd.Series, pd.DataFrame
            Time series values
        Returns 
        -------
        exog: np.ndarray, shape(samples,)
        '''
        
        if isinstance(exog, pd.Series):
            exog = exog.to_numpy(copy=True).reshape(-1, 1)
        elif isinstance(exog, np.ndarray) and exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        elif isinstance(exog, pd.DataFrame):
            exog = exog.to_numpy(copy=True)
            
        return exog
    
    
    def set_params(self, **params: dict) -> None:
        '''
        Set new values to the parameters of the scikit learn model stored in the
        ForecasterAutoreg.
        
        Parameters
        ----------
        params : dict
            Parameters values.
        Returns 
        -------
        self
        
        '''
        
        self.regressor.set_params(**params)
        
        
    def set_lags(self, lags: int) -> None:
        '''      
        Set new value to the attribute `lags`.
        Attributes `max_lag` and `window_size` are also updated.
        
        Parameters
        ----------
        lags : int, list, 1D np.array, range
        Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
            `int`: include lags from 1 to `lags`.
            `list` or `np.array`: include only lags present in `lags`.
        Returns 
        -------
        self
        
        '''
        
        if isinstance(lags, int) and lags < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, (list, range, np.ndarray)) and min(lags) < 1:
            raise Exception('min value of lags allowed is 1')
            
        if isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, (list, range)):
            self.lags = np.array(lags)
        elif isinstance(lags, np.ndarray):
            self.lags = lags
        else:
            raise Exception(
                f"`lags` argument must be `int`, `1D np.ndarray`, `range` or `list`. "
                f"Got {type(lags)}"
            )
            
        self.max_lag  = max(self.lags)
        self.window_size = max(self.lags)
        
        
    def set_out_sample_residuals(self, residuals: np.ndarray, append: bool=True)-> None:
        '''
        Set new values to the attribute `out_sample_residuals`. Out of sample
        residuals are meant to be calculated using observations that did not
        participate in the training process.
        
        Parameters
        ----------
        params : 1D np.ndarray
            Values of residuals. If len(residuals) > 1000, only a random sample
            of 1000 values are stored.
            
        append : bool, default `True`
            If `True`, new residuals are added to the once already stored in the attribute
            `out_sample_residuals`. Once the limit of 1000 values is reached, no more values
            are appended. If False, `out_sample_residuals` is overwrited with the new residuals.
            
        Returns 
        -------
        self
        
        '''
        if not isinstance(residuals, np.ndarray):
            raise Exception(
                f"`residuals` argument must be `1D np.ndarray`. Got {type(residuals)}"
            )
            
        if len(residuals) > 1000:
            residuals = np.random.choice(a=residuals, size=1000, replace=False)
                                 
        if not append or self.out_sample_residuals is None:
            self.out_sample_residuals = residuals
        else:
            free_space = max(0, 1000 - len(self.out_sample_residuals))
            if len(residuals) < free_space:
                self.out_sample_residuals = np.hstack((self.out_sample_residuals, residuals))
            else:
                self.out_sample_residuals = np.hstack((self.out_sample_residuals, residuals[:free_space]))
        

    def get_coef(self) -> np.ndarray:
        '''      
        Return estimated coefficients for the linear regression model stored in
        the forecaster. Only valid when the forecaster has been trained using
        as `regressor: `LinearRegression()`, `Lasso()` or `Ridge()`.
        
        Parameters
        ----------
        self
        Returns 
        -------
        coef : 1D np.ndarray
            Value of the coefficients associated with each predictor (lag).
            Coefficients are aligned so that `coef[i]` is the value associated
            with `self.lags[i]`.
        
        '''
        
        valid_instances = (sklearn.linear_model._base.LinearRegression,
                          sklearn.linear_model._coordinate_descent.Lasso,
                          sklearn.linear_model._ridge.Ridge
                          )
        
        if not isinstance(self.regressor, valid_instances):
            warnings.warn(
                ('Only forecasters with `regressor` `LinearRegression()`, ' +
                 ' `Lasso()` or `Ridge()` have coef.')
            )
            return
        else:
            coef = self.regressor.coef_
            
        return coef

    
    def get_feature_importances(self) -> np.ndarray:
        '''      
        Return impurity-based feature importances of the model stored in the
        forecaster. Only valid when the forecaster has been trained using
        `regressor=GradientBoostingRegressor()` or `regressor=RandomForestRegressor`.
        Parameters
        ----------
        self
        Returns 
        -------
        feature_importances : 1D np.ndarray
        Impurity-based feature importances associated with each predictor (lag).
        Values are aligned so that `feature_importances[i]` is the value
        associated with `self.lags[i]`.
        '''

        if not isinstance(self.regressor,
                        (sklearn.ensemble._forest.RandomForestRegressor,
                        sklearn.ensemble._gb.GradientBoostingRegressor)):
            warnings.warn(
                ('Only forecasters with `regressor=GradientBoostingRegressor()` '
                 'or `regressor=RandomForestRegressor`.')
            )
            return
        else:
            feature_importances = self.regressor.feature_importances_

        return feature_importances
      
      
      
from pylab import rcParams
import statsmodels.api as sm


import matplotlib.pyplot as plt
import seaborn as sns
#warnings.filterwarnings('ignore')


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def formato_linea(data):
  data["Mes"] = data["Mes"].astype(str)
  data["Año"] = data["Año"].astype(str)
  data['Fecha'] = data['Mes'].str.cat(data['Año'],sep="/")
  data['Fecha'] = pd.to_datetime(data['Fecha'], format='%m/%Y')
  data = data.set_index('Fecha')
  data.drop(labels=["Mes","Año"],axis=1,inplace=True)
  return data




def descomposicion(data, name="Aguascaslientes", model="additive"):
  plt.figure(figsize=(20,10))
  decomposition = sm.tsa.seasonal_decompose(data, model=model)
  plt.subplot(4,1,1)
  plt.plot(decomposition.observed)
  plt.ylabel("Observado")
  plt.title(f"Modelo {model}",fontsize=18)
  plt.subplot(4,1,4)
  plt.plot(decomposition.resid)
  plt.ylabel("Reciduo")
  plt.subplot(4,1,2)
  plt.plot(decomposition.seasonal)
  plt.ylabel("Seasonal")
  plt.subplot(4,1,3)
  plt.plot(decomposition.trend)
  plt.ylabel("Trend")

  plt.suptitle(f"{name}",fontsize=25)

  plt.show()
  
  
  
  
def visualizacion(data, steps, dato=None, name=None):
  datos_train = data[:-steps]
  datos_test  = data[(-steps):]

  plt.figure(figsize=(20,7))
  datos_train.plot()
  datos_test.plot()
  plt.legend(["train","test"],fontsize=19, loc="upper left")
  plt.xlabel("Mes", fontsize=15)
  if dato != None:
    plt.ylabel(dato, fontsize=15)
  if name != None:
    plt.title(name, fontsize=20)
  return datos_train,datos_test





def forecast(datos_train, datos_test, steps,lags=10, forest=True, name=None, dato=None):
  if forest:
    predictor = ForecasterAutoreg(
        regressor = RandomForestRegressor(random_state=3),
        lags = 10
    )
    predictor.fit(y=datos_train)
    predicciones = predictor.predict(steps = steps)
    predicciones = pd.Series(data=predicciones, index= datos_test.index)

    error_mse = mean_squared_error(
                y_true = datos_test,
                y_pred = predicciones
    )
    
    plt.figure(figsize=(20,8))
    datos_train.plot()
    datos_test.plot()
    predicciones.plot()
    plt.legend(["train","test","predicct"],fontsize=18,loc="upper left")
    plt.xlabel("Mes",fontsize=18)
    if dato != None:
      plt.ylabel(dato, fontsize=18)
    if name != None:
      plt.suptitle(name, fontsize=23)
    plt.title(f"Random Forest con mse {error_mse}")
    plt.show()

    return predicciones, error_mse

  if forest == False:
    predictor = ForecasterAutoreg(
        regressor = LinearRegression(),
        lags = 10
    )
    predictor.fit(y=datos_train)
    predicciones = predictor.predict(steps = steps)
    predicciones = pd.Series(data=predicciones, index= datos_test.index)

    error_mse = mean_squared_error(
                y_true = datos_test,
                y_pred = predicciones
    )
    
    plt.figure(figsize=(20,8))
    datos_train.plot()
    datos_test.plot()
    predicciones.plot()
    plt.legend(["train","test","predicct"],fontsize=18, loc="upper left")
    plt.xlabel("Mes",fontsize=18)
    if dato != None:
      plt.ylabel(dato, fontsize=18)
    if name != None:
      plt.suptitle(name, fontsize=23)
    plt.title(f"Random Forest con mse {error_mse}")
    plt.show()

    return predicciones, predictor
  
  
  
  
  
  
  #Red neuronal- Funciones de Omar
  
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import plotly.express as px




def red_neuronal(estado, municipio, df):
  filtro = (df['Estado'] == estado) & (df['Municipio'] == municipio)
  df_filter = df[filtro]

  # Construccion de la red
  close_data = df_filter['SPI'].values
  close_data = close_data.reshape((-1,1))

  split_percent = 0.80
  split = int(split_percent*len(close_data))

  close_train = close_data[:split]
  close_test = close_data[split:]

  date_train = df_filter['Date'][:split]
  date_test = df_filter['Date'][split:]

  look_back = 12 

  train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
  test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

  model = Sequential()
  model.add(
      LSTM(8,
          activation='relu',
          input_shape=(look_back,1))
  )
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')

  num_epochs = 700 
  model.fit(train_generator, epochs=num_epochs, verbose=0)

  prediction = model.predict(test_generator)

  close_train = close_train.reshape((-1))
  close_test = close_test.reshape((-1))
  prediction = prediction.reshape((-1))

  print("Mean Squared Error: ", mean_squared_error(close_test[:len(prediction)], prediction))

  graficar_datos(date_train, date_test, close_train, close_test, prediction, estado + ', ' + municipio)
  
  
  
  
def graficar_datos(date_train, date_test, close_train, close_test, prediction, nombre):
  trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
  )
  trace2 = go.Scatter(
      x = date_test,
      y = prediction,
      mode = 'lines',
      name = 'Prediction'
  )
  trace3 = go.Scatter(
      x = date_test,
      y = close_test,
      mode='lines',
      name = 'Ground Truth'
  )
  layout = go.Layout(
      title = "Drought Prediction of " + nombre,
      xaxis = {'title' : "Date"},
      yaxis = {'title' : "SPI"}
  )
  fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
  fig.show() #800 modificado
