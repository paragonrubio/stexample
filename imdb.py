import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Cargar el dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Configurar los sliders para los filtros
st.sidebar.header('Parámetros para la Predicción')

sepal_length = st.sidebar.slider('Longitud del Sépalo (cm)', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider('Ancho del Sépalo (cm)', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider('Longitud del Pétalo (cm)', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width = st.sidebar.slider('Ancho del Pétalo (cm)', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# Crear un dataframe con los valores seleccionados
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

# Realizar la predicción
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Mostrar los resultados de la predicción
st.title('Predicción del Iris')
st.write(f"Clase Predicha: {iris.target_names[prediction][0]}")
st.write(f"Probabilidad de la Predicción: {prediction_proba[0][prediction][0]:.2f}")

# Mostrar la tabla de probabilidades
st.subheader('Probabilidades de todas las Clases')
prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.table(prob_df)
