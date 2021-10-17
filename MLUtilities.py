import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
def conf_matrix(y_esperados, y_predichos)
  matrix = confusion_matrix(y_esperados, y_predichos)
  return matrix
#Funciones de obtención de Precisión (Accuracy), Sensibilidad y Especificidad.
(TP, FN, FP, TN) = np.ravel(matriz, order = 'C')

def accuracy(TP, FN, FP, TN):
  a = (TP + TN)/(TP+TN+FP+FN)
  return a

def sensitivity(TP, FN, FP, TN):
  s = TP/(TP +FN)
  return s

def specificity(TP, FN, FP, TN):
  sp = TN/(TN + FP)
  return sp

def precision(TP, FN, FP, TN): #un extra pero no necesario
  p = TP/(TP+FP)
  return p

#Funciones que comparen dos clasificadores:
#Obtengas precisión, sensibilidad y especificidad del clasificador 1
#Obtengas precisión, sensibilidad y especificidad del clasificador 2

#Digas cual es mejor en terminos de precisión
#Digas cual es mejor en términos de sensibilidad
#Digas cual es mejor en términos de especificidad.

#Funciones de evaluación multiclase.
