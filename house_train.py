import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Función para calcular los valores mínimos y máximos razonables excluyendo outliers
def reasonable_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# Cargar el conjunto de datos de California Housing
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedHouseVal')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un Random Forest Regressor
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

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

# Evaluar el modelo
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Predicción base (media)
mean_pred = np.full_like(y_test, y_train.mean())
baseline_mse = mean_squared_error(y_test, mean_pred)
baseline_rmse = np.sqrt(baseline_mse)

st.write(f"RMSE del Modelo: {rmse:.2f}")
st.write(f"RMSE de la Línea Base (Predicción Media): {baseline_rmse:.2f}")
