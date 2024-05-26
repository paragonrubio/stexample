import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Crear la aplicación en Streamlit
st.title("Clasificador de Iris")
st.write(f"Exactitud del modelo: {accuracy:.2f}")

st.sidebar.header("Ingrese las características de la flor:")

# Crear un formulario en la barra lateral
sepal_length = st.sidebar.number_input("Longitud del sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.sidebar.number_input("Ancho del sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.sidebar.number_input("Longitud del pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.sidebar.number_input("Ancho del pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Predicción
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

# Mapear la predicción a la especie correspondiente
species = iris.target_names[prediction][0]
st.write(f"La especie predicha es: **{species}**")

# Mostrar las probabilidades de predicción
st.write("Probabilidades de predicción:")
for i, prob in enumerate(prediction_proba[0]):
    st.write(f"{iris.target_names[i]}: {prob:.2f}")
