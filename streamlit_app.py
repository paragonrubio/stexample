import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

file_path = 'imdb2.csv'
imdb_data = load_data(file_path)

# Configurar los sliders para los filtros
st.sidebar.header('Filtros')

year_range = st.sidebar.slider('Año de lanzamiento', 
                               int(imdb_data['Year'].min()), 
                               int(imdb_data['Year'].max()), 
                               (2000, 2020))

rating_range = st.sidebar.slider('Rating', 
                                 float(imdb_data['Rating'].min()), 
                                 float(imdb_data['Rating'].max()), 
                                 (imdb_data['Rating'].min(), imdb_data['Rating'].max()))

revenue_range = st.sidebar.slider('Revenue (Millions)', 
                                  float(imdb_data['Revenue (Millions)'].min()), 
                                  float(imdb_data['Revenue (Millions)'].max()), 
                                  (imdb_data['Revenue (Millions)'].min(), imdb_data['Revenue (Millions)'].max()))

runtime_range = st.sidebar.slider('Duración (Minutos)', 
                                  int(imdb_data['Runtime (Minutes)'].min()), 
                                  int(imdb_data['Runtime (Minutes)'].max()), 
                                  (imdb_data['Runtime (Minutes)'].min(), imdb_data['Runtime (Minutes)'].max()))

votes_range = st.sidebar.slider('Número de votos', 
                                int(imdb_data['Votes'].min()), 
                                int(imdb_data['Votes'].max()), 
                                (imdb_data['Votes'].min(), imdb_data['Votes'].max()))

# Filtrar el dataframe según los valores seleccionados en los sliders
filtered_data = imdb_data[(imdb_data['Year'] >= year_range[0]) & (imdb_data['Year'] <= year_range[1]) &
                          (imdb_data['Rating'] >= rating_range[0]) & (imdb_data['Rating'] <= rating_range[1]) &
                          (imdb_data['Revenue (Millions)'] >= revenue_range[0]) & (imdb_data['Revenue (Millions)'] <= revenue_range[1]) &
                          (imdb_data['Runtime (Minutes)'] >= runtime_range[0]) & (imdb_data['Runtime (Minutes)'] <= runtime_range[1]) &
                          (imdb_data['Votes'] >= votes_range[0]) & (imdb_data['Votes'] <= votes_range[1])]


st.title('Dashboard de Películas')

# Visualización de facturación por año
st.subheader('Facturación por Año')
revenue_per_year = filtered_data.groupby('Year')['Revenue (Millions)'].sum()
fig, ax = plt.subplots()
revenue_per_year.plot(kind='bar', ax=ax)
ax.set_xlabel('Año')
ax.set_ylabel('Revenue (Millions)')
ax.set_title('Revenue por Año')
st.pyplot(fig)

# Mostrar las métricas
st.title('Dashboard de Películas')

# Top 20 películas con mayor rating
st.subheader('Top 20 Películas con Mayor Rating')
top_20_rating = filtered_data.nlargest(20, 'Rating')[['Title', 'Rating']]
st.table(top_20_rating)

# Top 20 películas con mayor revenue
st.subheader('Top 20 Películas con Mayor Revenue')
top_20_revenue = filtered_data.nlargest(20, 'Revenue (Millions)')[['Title', 'Revenue (Millions)']]
st.table(top_20_revenue)

# Top 20 películas con mayores votos
st.subheader('Top 20 Películas con Mayores Votos')
top_20_votes = filtered_data.nlargest(20, 'Votes')[['Title', 'Votes']]
st.table(top_20_votes)

# Recuento de actores más repetidos
st.subheader('Actores Más Repetidos')
actors_series = filtered_data['Actors'].str.split(', ').explode().value_counts()
top_actors = actors_series.head(20)
st.table(top_actors)
