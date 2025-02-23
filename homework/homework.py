# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Para evitar advertencias de casting de datos en pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Librerías utilizadas
import os
import json
import gzip
import pandas as pd
import pickle
from sklearn import pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


#  Carga de datos
train_data = pd.read_csv('./files/input/train_data.csv.zip')
test_data = pd.read_csv('./files/input/test_data.csv.zip')


# Paso 1 - Limpieza de datos
train_data.rename(columns={'default payment next month': 'default'}, inplace=True)
test_data.rename(columns={'default payment next month': 'default'}, inplace=True)

train_data.drop('ID', axis=1, inplace=True)
test_data.drop('ID', axis=1, inplace=True)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

train_data.loc[train_data['EDUCATION'] > 4, 'EDUCATION'] = 5
test_data.loc[test_data['EDUCATION'] > 4, 'EDUCATION'] = 5


# Paso 2 - Partición de entrenamiento y validación
X_train = train_data.drop('default', axis=1)
y_train = train_data['default']
X_test = test_data.drop('default', axis=1)
y_test = test_data['default']


# Paso 3 - Creación de pipeline
encoder = OneHotEncoder(handle_unknown='ignore')
reductor = PCA()
selector = SelectKBest()
scaler = StandardScaler()
classifier = SVC()
pipeline = Pipeline(
    steps=[
        ('encoder', encoder),
        ('reductor', reductor),
        ('scaler', scaler),
        ('selector', selector),
        ('classifier', classifier)
    ]
)


# Paso 4 - Optimización de hiperparámetros
param_grid = {
    'selector__k': [2, 3, 4],
    # 'reductor__n_components': [2, 3, 4],
    'classifier__C': [0.1, 1, 10],
}
grid_search = GridSearchCV(
    pipeline, param_grid, cv=2, scoring='balanced_accuracy'
)
grid_search.fit(X_train, y_train)


# Paso 5 - Guardar modelo
if not os.path.exists('./files/models'):
    os.makedirs('./files/models')

with gzip.open('./files/models/model.pkl.gz', 'wb') as file:
    file.write(pickle.dumps(grid_search))


# Paso 6 - Cálculo de métricas del modelo
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)

train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)
train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)


# Paso 7 - Cálculo de matriz de confusión del modelo
train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, y_train_pred).ravel()
test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, y_test_pred).ravel()


# Guardar métricas y matrices
if not os.path.exists('./files/output'):
    os.makedirs('./files/output')

with open('./files/output/metrics.json', 'w') as file:
    train_dict = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': train_precision,
        'balanced_accuracy': train_balanced_accuracy,
        'recall': train_recall,
        'f1_score': train_f1
    }
    test_dict = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': test_precision,
        'balanced_accuracy': test_balanced_accuracy,
        'recall': test_recall,
        'f1_score': test_f1
    }
    cm_train_dict = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {'predicted_0': int(train_tn), 'predicted_1': int(train_fp)},
        'true_1': {'predicted_0': int(train_fn), 'predicted_1': int(train_tp)}
    }
    cm_test_dict = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {'predicted_0': int(test_tn), 'predicted_1': int(test_fp)},
        'true_1': {'predicted_0': int(test_fn), 'predicted_1': int(test_tp)}
    }
    dictstrings = [
        json.dumps(train_dict),
        json.dumps(test_dict),
        json.dumps(cm_train_dict),
        json.dumps(cm_test_dict)
    ]
    file.write('\n'.join(dictstrings))