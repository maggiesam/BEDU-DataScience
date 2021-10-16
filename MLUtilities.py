import numpy as np
from sklearn.metrics import confusion_matrix

#Funciones de separación de entrenamiento, validación y prueba.

#Funciones de separación de datasets con K-Fold (el usuario debe poner el K, si K = 1 debe generar un Leave-One-Out Cross Validation).

#Funciones de evaluación con matriz de confusión.

#Funciones de obtención de Precisión (Accuracy), Sensibilidad y Especificidad.
def conf_matrix(y_esperados, y_predichos)
  matrix = confusion_matrix(y_esperados, y_predichos)
  return matrix

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
