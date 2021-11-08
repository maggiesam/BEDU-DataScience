
## Bitácora del poswork 3 por Jesus Omar Magaña Medina

Fecha y hora: 16 / oc / 2021 -- 06:00pm

Algoritmo: K-means

Dataset: dataframe-junto.csv

Configuración:
Algoritmo K-means
Numero de Clusters: 4
Métodos usados para elegir el numero óptimo de Clusters: Metodo del codo y silueta
Método para reducción de datos: PCA

Resultado:
Varianza acumulada explicada por mis dos nuevas variables: 61.44% 


## Bitácora del poswork 4 por Efraín Soto Olmos
Se modelaron utilizando la función regresión_lineal de la librería MLUtilities

Se utilizan las variables independientes de, Temperatura_maxima, Temperatura_minima, Temperatura_promedio, precipitacion y Siembrea_ha. Como variable dependiente se creo una nueva columna con el porcentaje de siembra cosechado.
La función permite dividir los datos por planta y por mes (esta es opcional)
Se modelaran las 3 plantas con diferente grado ademas de separarlos o no por meses

Modelo del maiz

Grado 1 sin meses
Entrenamiento: MSE =1350.4899113383008 Pruebas: MSE =1315.9361971731496

Grado 2 sin meses
Entrenamiento: MSE =1193.8025578504667 Pruebas: MSE =1181.962467999478

Grado 3 sin meses
Entrenamiento: MSE =2146.7440439211973 Pruebas: MSE =2328.853317160301

Grado 1 con mes de enero
Entrenamiento: MSE =709.8948736076497 Pruebas: MSE =698.004746379093

Grado 2 con mes de enero
Entrenamiento: MSE =613.6261702756863 Pruebas: MSE =613.1658306750875

Grado 3 con mes de enero
Entrenamiento: MSE =3245.8451564170414 Pruebas: MSE =6782.508696355911



Modelo del frijol

Grado 1 sin meses
Entrenamiento: MSE =1190.4027280451758 Pruebas: MSE =1188.889181493713

Grado 2 sin meses
Entrenamiento: MSE =1074.031556663748 Pruebas: MSE =1084.753949301942

Grado 3 sin meses
Entrenamiento: MSE =1054.3071924551366 Pruebas: MSE =1053.832373738116

Grado 4 sin meses
Entrenamiento: MSE =2670.559050149221 Pruebas: MSE =4187.452722719679

Grado 1 con mes enero
Entrenamiento: MSE =1074.094407493021 Pruebas: MSE =1514.8008799995862

Grado 2 con mes enero
Entrenamiento: MSE =953.7825278346365 Pruebas: MSE =50669.44777254354}

Grado 3 con mes enero
Entrenamiento: MSE =911.6838244060021 Pruebas: MSE =16539.71072906514



Modelo del trigo

Grado 1 sin meses
Entrenamiento: MSE =962.9665431268379 Pruebas: MSE =882.6306900843082


Grado 2 sin meses
Entrenamiento: MSE =908.8798758909097 Pruebas: MSE =825.0957162003385

Grado 3 sin meses
Entrenamiento: MSE =856.8501662808189 Pruebas: MSE =1632.0819788669771

Grado 1 con mes enero
Entrenamiento: MSE =557.434026049089 Pruebas: MSE =855.016977392184

Grado 2 con mes enero
Entrenamiento: MSE =502.2289446688361 Pruebas: MSE =879.5097654972043

Grado 3 con mes enero
Entrenamiento: MSE =410.9400772042837 Pruebas: MSE =1435.588469170785

## Bitácora forecasting Efraín Soto (regresión lineal, random forest, red neutonal de estados del sur)

El forecasting a 24 meses con  regresión lineal obtuvo un error de 1.79914032999 para la temperatura promedio

El forecasting a 24 meses con random forest obtuvo un error de 1.111217212150515 para la temperatura promedio

Por lo que se esperan mejores resultados con el método de regresión de random forest en los hiperparametros.



Forecasting a 24 meses con Random Forest obtuvo un error de 1.0498043693650942 para la temperatura mínima

Forecasting a 24 meses con Linear Regression obtuvo un error de 1.7527619998746913 para la temperatura mínima

Por lo que se espera un mejor desempeño en los hiperparametros



Forecasting para la temperatura máxima a 24 meses de random forest con error de 1.4495699848220678

Forecasting para la temperatura máxima a 24 meses de regresión lineal con error de 2.5734093305159917



Forecasting para la precipitación a 24 meses con random forest y error 1008.3917395427161

Forecasting para la precipitación a 24 meses con regresión lineal y error 1632.1577580238127

Forecasting para la precipitación a 12 mese con random forest y error 331.59

Forecasting para la precipitación a 6 meses con random forest y error de 3872





Forecasting con la función de selección de los hiperparametros a 24 meses para temperatura promedio con un error a 1.2040226273221546 por lo que el factor de
aleatoriedad es muy importante

La diferencia en cuanto a los hiperparametros y los parámetros es muy poca, por lo que se excluirá el uso de la función para encontrar los hiperparametros debido a que interfiera con
otras funciones por las versiones de las librerías que se requieren


Red neuronal Muna 1.36095568 sin los datos outliers 3.9

Red neuronal Yucatán Motul 2.0661826997254567

Red neuronal Yucatán Peto 1.5167553000494511

Red neuronal Yucatán Rio Lagartos 1.0607660893450739

Red neuronal Yucatán Tizimín 0.9429591903644475

Red neuronal Yucatán Mocochá 1.0553002765349526

Red neuronal Yucatán Oxkutzcab 1.4019736380710366


Red neuronal Campeche Palizada 1.6436496467391326

Red neuronal Campeche Sabancuy 1.3403760082052025

Red neuronal Campeche Campeche 1.2867122981000927


Red neuronal Tabasco Tenosique de Pino Suárez 1.6673301460089072

Red neuronal Tabasco Macuspana 1.4023028993952262

Red neuronal Tabasco Habanero 1.2867790386001323

Red neuronal Tabasco Capitan Felipe Castellanos Díaz (San Pedro) 1.9325713092380865


Red neuronal Oaxaca San Pedro Quiatoni 0.5066049279666287

Red neuronal Oaxaca San Jerónimo Ixtepec 0.8102670953388078

Red neuronal Oaxaca Santo Domingo Zanatepec 0.9264886092537115

Red neuronal Oaxaca Magdalena Tequisistlán 1.3891505233595367


Red neuronal Quintana Roo Felipe Carrillo Puerto 1.1579004923829412

Red neuronal Quintana Roo Chetumal 1.7424919057492327


Red neuronal Puebla San Pedro Zacachimalpa 1.3564002992735806

Red neuronal Puebla Chietla 1.0928813774818509

Red neuronal Puebla San Miguel Tenextatiloyan 1.2071596413976173

Red neuronal Puebla Axutla 0.46127609898263844

Red neuronal Puebla Tehuacan 1.039809188573151

Red neuronal Puebla Teziutlan 1.2840090006554783

Red neuronal Puebla Huauchinango 0.8822362658018856

Red neuronal Puebla Xicotepec de Juarez 1.2073015022249691
