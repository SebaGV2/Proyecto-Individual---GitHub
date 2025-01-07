# Importar la librería pandas para trabajar con los datos
import pandas as pd
# Definir las rutas de los archivos CSV
movies_path = './data/movies_dataset.csv'
credits_path = './data/credits.csv'

# Leer los archivos CSV en DataFrames
movies_df = pd.read_csv(movies_path, low_memory=False)
credits_df = pd.read_csv(credits_path, low_memory=False)
# Mostrar las primeras filas de cada archivo para revisar su contenido
print("Primeras filas de movies_dataset.csv:")
print(movies_df.head())  # Muestra las primeras 5 filas

print("\nPrimeras filas de credits.csv:")
print(credits_df.head())  # Muestra las primeras 5 filas
# Mostrar información general de movies_dataset.csv
print("Información general de movies_dataset.csv:")
print(movies_df.info())  # Información sobre columnas, tipos de datos y valores nulos

print("\nPrimeras filas de movies_dataset.csv:")
print(movies_df.head())  # Primeras 5 filas completas

# Mostrar información general de credits.csv
print("\nInformación general de credits.csv:")
print(credits_df.info())  # Información sobre columnas, tipos de datos y valores nulos

print("\nPrimeras filas de credits.csv:")
print(credits_df.head())  # Primeras 5 filas completas
# Convertir las columnas 'budget' y 'revenue' a números
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce').fillna(0)
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce').fillna(0)

# Mostrar un resumen para verificar el cambio
print("\nResumen de 'budget' y 'revenue' después de convertirlos:")
print(movies_df[['budget', 'revenue']].describe())
# Eliminar filas con 'release_date' nulo
movies_df = movies_df.dropna(subset=['release_date'])

# Mostrar la cantidad de filas restantes
print(f"\nCantidad de filas después de eliminar 'release_date' nulo: {len(movies_df)}")
# Crear la columna 'return' (retorno de inversión)
movies_df['return'] = movies_df.apply(
    lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0,
    axis=1
)

# Mostrar un resumen de la nueva columna
print("\nResumen de la columna 'return':")
print(movies_df['return'].describe())
# Convertir 'release_date' a formato datetime y extraer el año
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
movies_df['release_year'] = movies_df['release_date'].dt.year

# Mostrar las primeras filas para verificar
print("\nPrimeras filas con 'release_year':")
print(movies_df[['release_date', 'release_year']].head())
# Eliminar columnas innecesarias
columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
movies_df = movies_df.drop(columns=columns_to_drop, errors='ignore')

# Mostrar las columnas restantes
print("\nColumnas restantes después de eliminar las innecesarias:")
print(movies_df.columns)
import ast  # Para trabajar con strings que representan listas o diccionarios

import pandas as pd
import json

def extract_collection(value):
    try:
        if pd.isna(value):  # Maneja NaN o valores nulos
            return None
        elif isinstance(value, dict):  # Ya es un diccionario
            return value.get('name', None)
        elif isinstance(value, str):  # Es una cadena JSON
            collection_dict = json.loads(value)
            return collection_dict.get('name', None)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    return None  # Valor por defecto

movies_df['belongs_to_collection'] = movies_df['belongs_to_collection'].apply(extract_collection)

print(movies_df['belongs_to_collection'].apply(type).value_counts())  # Ver tipos de datos
print(movies_df[['title', 'belongs_to_collection']].head(10))  # Verifica las primeras filas

print(movies_df['belongs_to_collection'].value_counts(dropna=False))  # Para verificar valores únicos

import ast

# Ejemplo: Extraer nombres de géneros
def extract_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres] if isinstance(genres, list) else None
    except (ValueError, SyntaxError):
        return None

movies_df['genres'] = movies_df['genres'].apply(extract_genres)
print(movies_df[['title', 'genres']].head())

print(movies_df.isnull().sum())

movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')

movies_df['roi'] = (movies_df['revenue'] - movies_df['budget']) / movies_df['budget']

movies_df['popularity'] = pd.to_numeric(movies_df['popularity'], errors='coerce')
print(movies_df['popularity'].isnull().sum())

movies_df = movies_df.dropna(subset=['popularity'])

movies_df['popularity'] = movies_df['popularity'].fillna(movies_df['popularity'].median())

movies_df['popularity_category'] = pd.qcut(movies_df['popularity'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

print(movies_df[['title', 'popularity', 'popularity_category']].head())

movies_df = movies_df[movies_df['revenue'] > 0]

# Luego aplica la clasificación
movies_df['revenue_category'] = pd.qcut(movies_df['revenue'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

print(movies_df[['title', 'revenue', 'revenue_category']].head())

# Distribución de categorías de revenue
print(movies_df['revenue_category'].value_counts())

# Distribución de categorías de popularity
print(movies_df['popularity_category'].value_counts())

# Promedio de vote_average por categoría de revenue
revenue_avg_votes = movies_df.groupby('revenue_category')['vote_average'].mean()
print(revenue_avg_votes)

# Promedio de vote_average por categoría de popularity
popularity_avg_votes = movies_df.groupby('popularity_category')['vote_average'].mean()
print(popularity_avg_votes)

import matplotlib.pyplot as plt

# Visualizar distribución de categorías de revenue
movies_df['revenue_category'].value_counts().plot(kind='bar', title='Distribución de Revenue Categories')
plt.show()

# Visualizar promedio de votos por categoría de revenue
revenue_avg_votes.plot(kind='bar', title='Promedio de Votos por Revenue Categories')
plt.show()

# Explorar géneros en películas de "Very High" popularity
popular_genres = movies_df[movies_df['popularity_category'] == 'Very High']['genres'].value_counts()
print(popular_genres)

# O para "Very High" revenue
high_revenue_genres = movies_df[movies_df['revenue_category'] == 'Very High']['genres'].value_counts()
print(high_revenue_genres)

import matplotlib.pyplot as plt

# Scatter plot entre presupuesto y revenue
movies_df.plot.scatter(x='budget', y='revenue', title='Relación Presupuesto vs. Revenue')
plt.show()

# Revenue promedio por año
revenue_by_year = movies_df.groupby('release_year')['revenue'].mean()
print(revenue_by_year)

# Visualizar
revenue_by_year.plot(title='Revenue Promedio por Año')
plt.show()

print(credits_df.columns)

# Asegúrate de que ambas columnas 'id' sean del mismo tipo
movies_df['id'] = movies_df['id'].astype(str)
credits_df['id'] = credits_df['id'].astype(str)

# Realiza la unión
movies_df = movies_df.merge(credits_df, on='id', how='left')

# Verifica que la unión se haya realizado correctamente
print(movies_df.head())

# Función para extraer el director de la lista 'crew'
def extract_director(crew_list):
    for person in crew_list:
        if person.get('job') == 'Director':
            return person.get('name')
    return None

# Aplicar la función a la columna 'crew'
movies_df['director'] = movies_df['crew'].apply(lambda x: extract_director(eval(x)))

# Verificar las primeras filas con la columna 'director'
print(movies_df[['title', 'director']].head())

# Función para extraer los nombres del elenco principal
def extract_cast(cast_list, num_cast=3):
    return [person.get('name') for person in cast_list[:num_cast]]

# Aplicar la función a la columna 'cast'
movies_df['main_cast'] = movies_df['cast'].apply(lambda x: extract_cast(eval(x)))

# Verificar las primeras filas con la columna 'main_cast'
print(movies_df[['title', 'main_cast']].head())

# Eliminar columnas innecesarias
movies_df.drop(['cast', 'crew'], axis=1, inplace=True)

# Verificar la estructura final del DataFrame
print(movies_df.info())

# Calcular ingresos totales por director
director_revenue = movies_df.groupby('director')['revenue'].sum().sort_values(ascending=False).head(10)

# Mostrar los resultados
print(director_revenue)

from collections import Counter

# Contar las apariciones de cada actor
all_cast = movies_df['main_cast'].explode()
actor_counts = Counter(all_cast).most_common(10)

# Mostrar los 10 actores más frecuentes
print(actor_counts)

# Verificar los primeros valores de la columna 'genres'
print(movies_df['genres'].head())

def split_genres(genres):
    if isinstance(genres, str):
        return eval(genres)
    return genres  # Si ya es lista, la retorna como está

# Aplicar la función corregida
movies_df['genres'] = movies_df['genres'].apply(split_genres)

# Explorar los géneros y calcular métricas
genre_metrics = movies_df.explode('genres').groupby('genres')[['revenue', 'vote_average']].mean()

# Mostrar las métricas por género
print(genre_metrics)

# Explosión de géneros para manejar cada género por separado
movies_exploded = movies_df.explode('genres')

# Cálculo de métricas por género
genre_metrics = movies_exploded.groupby('genres')[['revenue', 'vote_average']].mean()

# Ordenar por ingresos promedio
genre_metrics = genre_metrics.sort_values(by='revenue', ascending=False)

# Mostrar las métricas por género
print(genre_metrics)

print("Top 5 géneros por ingresos promedio:")
print(genre_metrics['revenue'].head(5))

print("Top 5 géneros por puntuación promedio:")
print(genre_metrics['vote_average'].sort_values(ascending=False).head(5))

import matplotlib.pyplot as plt

# Top 5 géneros más rentables
genre_metrics['revenue'].head(5).plot(kind='bar', title='Top 5 géneros por ingresos promedio')
plt.ylabel('Ingresos promedio')
plt.show()

# Top 5 géneros mejor calificados
genre_metrics['vote_average'].sort_values(ascending=False).head(5).plot(kind='bar', title='Top 5 géneros por puntuación promedio')
plt.ylabel('Puntuación promedio')
plt.show()

# Conteo de películas por año
movies_per_year = movies_df.groupby('release_year')['title'].count()

# Visualizar
print("Cantidad de películas lanzadas por año:")
print(movies_per_year)

# Graficar la cantidad de películas por año
movies_per_year.plot(kind='line', title='Cantidad de películas lanzadas por año')
plt.xlabel('Año')
plt.ylabel('Cantidad de películas')
plt.show()

# Ingresos promedio por año
revenue_per_year = movies_df.groupby('release_year')['revenue'].mean()

# Visualizar
print("Ingresos promedio por año:")
print(revenue_per_year)

# Graficar ingresos promedio por año
revenue_per_year.plot(kind='line', title='Ingresos promedio por año')
plt.xlabel('Año')
plt.ylabel('Ingresos promedio')
plt.show()

# Puntuaciones promedio por año
ratings_per_year = movies_df.groupby('release_year')['vote_average'].mean()

# Visualizar
print("Puntuaciones promedio por año:")
print(ratings_per_year)

# Graficar puntuaciones promedio por año
ratings_per_year.plot(kind='line', title='Puntuaciones promedio por año')
plt.xlabel('Año')
plt.ylabel('Puntuación promedio')
plt.show()

# Conteo de frecuencia de géneros
from collections import Counter

# Aplanar la lista de géneros
all_genres = [genre for genres in movies_df['genres'] for genre in genres]
genre_counts = Counter(all_genres)

# Visualizar los géneros más frecuentes
print("Frecuencia de géneros:")
print(genre_counts.most_common())

# Graficar los géneros más frecuentes
pd.Series(dict(genre_counts)).sort_values(ascending=False).plot(kind='bar', title='Frecuencia de géneros')
plt.xlabel('Género')
plt.ylabel('Frecuencia')
plt.show()

# Ingresos promedio por género
genre_revenues = movies_df.explode('genres').groupby('genres')['revenue'].mean()

# Visualizar
print("Ingresos promedio por género:")
print(genre_revenues.sort_values(ascending=False))

# Graficar ingresos promedio por género
genre_revenues.sort_values(ascending=False).plot(kind='bar', title='Ingresos promedio por género')
plt.xlabel('Género')
plt.ylabel('Ingresos promedio')
plt.show()

# Puntuaciones promedio por género
genre_ratings = movies_df.explode('genres').groupby('genres')['vote_average'].mean()

# Visualizar
print("Puntuaciones promedio por género:")
print(genre_ratings.sort_values(ascending=False))

# Graficar puntuaciones promedio por género
genre_ratings.sort_values(ascending=False).plot(kind='bar', title='Puntuaciones promedio por género')
plt.xlabel('Género')
plt.ylabel('Puntuación promedio')
plt.show()

# Frecuencia de géneros por categoría de popularidad
genre_popularity = movies_df.explode('genres').groupby(['popularity_category', 'genres']).size()

# Visualizar
print("Frecuencia de géneros por categoría de popularidad:")
print(genre_popularity.unstack().fillna(0))

# Graficar
genre_popularity.unstack().fillna(0).plot(kind='bar', stacked=True, title='Frecuencia de géneros por categoría de popularidad')
plt.xlabel('Categoría de Popularidad')
plt.ylabel('Frecuencia')
plt.show()

# Aplanar la lista de compañías de producción
from collections import Counter

all_companies = [company for companies in movies_df['production_companies'] for company in companies]
company_counts = Counter(all_companies)

# Visualizar las compañías más frecuentes
print("Frecuencia de compañías de producción:")
print(company_counts.most_common(10))

# Graficar las compañías más frecuentes
pd.Series(dict(company_counts)).sort_values(ascending=False).head(10).plot(kind='bar', title='Frecuencia de compañías de producción')
plt.xlabel('Compañía')
plt.ylabel('Frecuencia')
plt.show()

# Ingresos promedio por compañía
company_revenues = movies_df.explode('production_companies').groupby('production_companies')['revenue'].mean()

# Visualizar
print("Ingresos promedio por compañía:")
print(company_revenues.sort_values(ascending=False).head(10))

# Graficar ingresos promedio por compañía
company_revenues.sort_values(ascending=False).head(10).plot(kind='bar', title='Ingresos promedio por compañía')
plt.xlabel('Compañía')
plt.ylabel('Ingresos promedio')
plt.show()

# Puntuaciones promedio por compañía
company_ratings = movies_df.explode('production_companies').groupby('production_companies')['vote_average'].mean()

# Visualizar
print("Puntuaciones promedio por compañía:")
print(company_ratings.sort_values(ascending=False).head(10))

# Graficar puntuaciones promedio por compañía
company_ratings.sort_values(ascending=False).head(10).plot(kind='bar', title='Puntuaciones promedio por compañía')
plt.xlabel('Compañía')
plt.ylabel('Puntuación promedio')
plt.show()

print(movies_df['production_companies'].head())
print(movies_df['production_companies'].dtype)

movies_df = movies_df[movies_df['production_companies'].notnull()]

print(type(movies_df['production_companies'].iloc[0]))

import ast

# Inspeccionar un ejemplo de la columna
print("\nEjemplo de un valor en production_companies:")
print(movies_df['production_companies'].iloc[0])

# Convertir cadenas que representan listas en listas reales
movies_df['production_companies'] = movies_df['production_companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Explota la columna para que cada fila tenga una sola compañía
movies_df = movies_df.explode('production_companies')

# Si los elementos son diccionarios, extraemos solo el nombre de la compañía
movies_df['production_companies'] = movies_df['production_companies'].apply(
    lambda x: x['name'] if isinstance(x, dict) else x
)

# Verifica el resultado después de explotar
print("\nDespués de explotar production_companies:")
print(movies_df[['title', 'production_companies']].head())

# Filtrar filas con production_companies nulo o vacío
movies_df = movies_df[movies_df['production_companies'].notnull()]

# Verificar el tipo de un ejemplo
print("\nTipo después de limpieza:", type(movies_df['production_companies'].iloc[0]))

# Calcular ROI si aún no lo has hecho
movies_df['roi'] = (movies_df['revenue'] - movies_df['budget']) / movies_df['budget']
movies_df['roi'] = movies_df['roi'].replace([float('inf'), -float('inf')], float('nan'))

# Agrupa por compañía y calcula el ROI promedio
company_roi = movies_df.groupby('production_companies')['roi'].mean()

# Mostrar las compañías con el ROI más alto
print("\nTop 10 compañías por ROI:")
print(company_roi.sort_values(ascending=False).head(10))

import matplotlib.pyplot as plt

# Graficar las compañías con mayor ROI promedio
company_roi.sort_values(ascending=False).head(10).plot(kind='bar', title='Top 10 compañías por ROI promedio')
plt.xlabel('Compañía')
plt.ylabel('ROI promedio')
plt.show()

# Explosión de géneros
movies_df = movies_df.explode('genres')

# Agrupamos por género para calcular métricas clave
genre_metrics = movies_df.groupby('genres').agg({
    'revenue': 'mean',
    'vote_average': 'mean',
    'id': 'count'
}).rename(columns={'id': 'movie_count'})

# Mostrar los géneros más rentables
print("\nTop géneros por ingresos promedio:")
print(genre_metrics.sort_values('revenue', ascending=False).head(10))

# Gráfica de ingresos promedio por género
genre_metrics['revenue'].sort_values(ascending=False).head(10).plot(
    kind='bar', title='Top 10 géneros por ingresos promedio'
)
plt.xlabel('Género')
plt.ylabel('Ingresos promedio')
plt.show()

# Agrupamos por año
yearly_metrics = movies_df.groupby('release_year').agg({
    'revenue': 'mean',
    'vote_average': 'mean',
    'id': 'count'
}).rename(columns={'id': 'movie_count'})

# Gráfico de ingresos promedio por año
yearly_metrics['revenue'].plot(title='Ingresos promedio por año', figsize=(10, 6))
plt.xlabel('Año')
plt.ylabel('Ingresos promedio')
plt.show()

# Gráfico de puntuaciones promedio por año
yearly_metrics['vote_average'].plot(title='Puntuaciones promedio por año', figsize=(10, 6))
plt.xlabel('Año')
plt.ylabel('Puntuación promedio')
plt.show()

# Agrupamos por director
director_metrics = movies_df.groupby('director').agg({
    'revenue': 'mean',
    'id': 'count'
}).rename(columns={'id': 'movie_count'})

# Mostrar los directores más exitosos
print("\nTop directores por ingresos promedio:")
print(director_metrics.sort_values('revenue', ascending=False).head(10))

# Graficar directores por ingresos promedio
director_metrics['revenue'].sort_values(ascending=False).head(10).plot(
    kind='bar', title='Top 10 directores por ingresos promedio'
)
plt.xlabel('Director')
plt.ylabel('Ingresos promedio')
plt.show()

# Explosión de main_cast
movies_df = movies_df.explode('main_cast')

# Contar las apariciones de cada actor
actor_counts = movies_df['main_cast'].value_counts()

# Mostrar los actores más frecuentes
print("\nTop 10 actores más frecuentes:")
print(actor_counts.head(10))

# Graficar actores más frecuentes
actor_counts.head(10).plot(kind='bar', title='Top 10 actores más frecuentes')
plt.xlabel('Actor')
plt.ylabel('Cantidad de apariciones')
plt.show()

# Gráfica de dispersión entre popularidad e ingresos
movies_df.plot.scatter(x='popularity', y='revenue', title='Popularidad vs Ingresos', alpha=0.5, figsize=(10, 6))
plt.xlabel('Popularidad')
plt.ylabel('Ingresos')
plt.show()

def buscar_peliculas_por_director(director, df):
    """
    Busca películas dirigidas por un director específico.

    Args:
        director (str): Nombre del director.
        df (DataFrame): Dataset de películas.

    Returns:
        DataFrame: Subconjunto del dataset con las películas del director.
    """
    peliculas_director = df[df['director'] == director]
    return peliculas_director[['title', 'release_year', 'revenue', 'vote_average']].sort_values(by='release_year')

# Ejemplo de uso:
resultado = buscar_peliculas_por_director("Steven Spielberg", movies_df)
print(resultado)

def buscar_peliculas_por_genero(genero, df):
    """
    Busca películas que pertenezcan a un género específico.

    Args:
        genero (str): Nombre del género.
        df (DataFrame): Dataset de películas.

    Returns:
        DataFrame: Subconjunto del dataset con las películas del género.
    """
    # Asegurarse de que 'genres' sea una lista y manejar valores nulos
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Filtrar películas que contienen el género específico
    peliculas_genero = df[df['genres'].apply(lambda x: genero in x)]
    
    return peliculas_genero[['title', 'release_year', 'revenue', 'vote_average']].sort_values(by='release_year')

# Ejemplo de uso:
resultado = buscar_peliculas_por_genero("Action", movies_df)
print(resultado)

movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])

def buscar_peliculas_por_genero(genero, df):
    """
    Busca películas que pertenezcan a un género específico.

    Args:
        genero (str): Nombre del género.
        df (DataFrame): Dataset de películas.

    Returns:
        DataFrame: Subconjunto del dataset con las películas del género.
    """
    # Asegurarse de que 'genres' sea una lista y manejar valores nulos
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Filtrar películas que contienen el género específico
    peliculas_genero = df[df['genres'].apply(lambda x: genero in x)]
    
    return peliculas_genero[['title', 'release_year', 'revenue', 'vote_average']].sort_values(by='release_year')

# Ejemplo de uso:
movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])  # Limpieza inicial
resultado = buscar_peliculas_por_genero("Action", movies_df)
print(resultado)

def buscar_peliculas_por_director(nombre_director, df):
    """
    Busca películas dirigidas por un director específico.
    
    Parámetros:
    - nombre_director (str): Nombre del director.
    - df (DataFrame): DataFrame que contiene los datos de las películas.

    Retorna:
    - Un DataFrame con las películas del director incluyendo título, año de lanzamiento,
      ingresos y calificación promedio.
    """
    # Filtrar las películas por el director especificado
    peliculas = df[df['director'] == nombre_director]

    if peliculas.empty:
        print(f"No se encontraron películas dirigidas por {nombre_director}.")
        return peliculas

    # Seleccionar columnas relevantes
    peliculas = peliculas[['title', 'release_year', 'revenue', 'vote_average']]
    # Ordenar por ingresos de mayor a menor
    peliculas = peliculas.sort_values(by='revenue', ascending=False)

    return peliculas

    # Probar la función con el director James Cameron
resultado_cameron = buscar_peliculas_por_director("James Cameron", movies_df)

# Mostrar los resultados
print(resultado_cameron)

def buscar_peliculas_por_año(anio, df):
    """
    Busca películas lanzadas en un año específico.

    Args:
        anio (int): Año de lanzamiento.
        df (DataFrame): Dataset de películas.

    Returns:
        DataFrame: Subconjunto del dataset con las películas del año.
    """
    try:
        # Filtrar por el año proporcionado
        peliculas_anio = df[df['release_year'] == anio]
        
        if peliculas_anio.empty:
            print(f"No se encontraron películas lanzadas en el año {anio}.")
            return peliculas_anio
        
        # Seleccionar columnas relevantes
        peliculas_anio = peliculas_anio[['title', 'director', 'revenue', 'vote_average', 'genres']]
        
        # Eliminar duplicados basados en el título
        peliculas_anio = peliculas_anio.drop_duplicates(subset=['title'])
        
        # Ordenar por ingresos (revenue)
        peliculas_anio = peliculas_anio.sort_values(by='revenue', ascending=False)
        return peliculas_anio
    except KeyError as e:
        print(f"Error al buscar películas: Falta una columna clave en el DataFrame ({e}).")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")
        return pd.DataFrame()

        # Ejemplo 1: Películas de 2015
resultado_2015 = buscar_peliculas_por_año(2015, movies_df)
print("Películas lanzadas en 2015 (sin duplicados):")
print(resultado_2015)

def peliculas_populares_por_decada(df):
    """
    Encuentra las películas más populares por década.

    Args:
        df (DataFrame): El DataFrame de películas.

    Returns:
        DataFrame: Películas más populares por cada década.
    """
    # Asegurarse de que 'release_year' no tenga valores nulos y convertir a enteros
    df = df[df['release_year'].notnull()]
    df['release_year'] = df['release_year'].astype(int)

    # Calcular la década
    df['decade'] = (df['release_year'] // 10) * 10

    # Encontrar la película más popular por década
    populares_por_decada = (
        df.loc[df.groupby('decade')['popularity'].idxmax(), ['title', 'decade', 'release_year', 'popularity', 'vote_average']]
    )

    # Eliminar duplicados
    populares_por_decada = populares_por_decada.drop_duplicates()

    return populares_por_decada.sort_values(by='decade')

    # Ejecutar la función actualizada
resultado_populares = peliculas_populares_por_decada(movies_df)

# Mostrar el resultado
print("Películas más populares por década (sin duplicados):")
print(resultado_populares)

def directores_mas_exitosos(df, criterio="ingresos"):
    """
    Identifica los directores con más películas exitosas.

    Args:
        df (DataFrame): El DataFrame de películas.
        criterio (str): El criterio de éxito ('ingresos' o 'calificaciones').

    Returns:
        DataFrame: Top 10 directores con más películas exitosas.
    """
    if criterio == "ingresos":
        exitosas = df[df['revenue_category'].isin(['High', 'Very High'])]
    elif criterio == "calificaciones":
        exitosas = df[df['vote_average'] > 7]
    else:
        raise ValueError("Criterio no válido. Usa 'ingresos' o 'calificaciones'.")

    # Contar películas exitosas por director
    directores_exitosos = exitosas['director'].value_counts().head(10)

    return directores_exitosos

    # Directores con más películas exitosas por ingresos
resultado_ingresos = directores_mas_exitosos(movies_df, criterio="ingresos")
print("Directores con más películas exitosas por ingresos:")
print(resultado_ingresos)

# Directores con más películas exitosas por calificación
resultado_calificaciones = directores_mas_exitosos(movies_df, criterio="calificaciones")
print("\nDirectores con más películas exitosas por calificación:")
print(resultado_calificaciones)


