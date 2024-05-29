import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Función para calcular los valores mínimos y máximos razonables excluyendo outliers
def reasonable_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


modelo = joblib.load('modelo_california_housing.pkl')

# Cargar el conjunto de datos de California Housing para obtener estadísticas
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedHouseVal')

# Aplicación de Streamlit
st.title("Predicción de Precios de Viviendas en California")

st.write("""
Esta aplicación utiliza un modelo de aprendizaje automático para predecir el valor medio de viviendas ocupadas por sus dueños en California.
""")

# Crear un formulario para la entrada del usuario
st.header("Características de Entrada")
form = st.form(key='input_form')
inputs = {}
for feature in X.columns:
    min_val, max_val = reasonable_bounds(X[feature])
    inputs[feature] = form.slider(
        feature, 
        float(min_val), 
        float(max_val), 
        float(X[feature].mean())
    )
submit_button = form.form_submit_button(label='Predecir')

if submit_button:
    # Preparar los datos de entrada para la predicción
    input_data = np.array([list(inputs.values())])
    prediction = modelo.predict(input_data)[0]
    
    st.subheader("Predicción")
    st.write(f"El valor medio predicho de la vivienda es ${prediction * 100000:.2f}")


