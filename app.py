from flask import Flask, jsonify
import pandas as pd
import ast  # Para manejar listas en formato string

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar los datos desde la carpeta 'data'
movies_path = './data/movies_dataset_reduced_1000.csv'
credits_path = './data/credits_reduced_1000.csv'



try:
    # Cargar los datos
    movies_df = pd.read_csv(movies_path, low_memory=False)
    credits_df = pd.read_csv(credits_path, low_memory=False)
    print("Datos cargados exitosamente.")

    # Asegurarse de que ambas columnas 'id' sean del mismo tipo
    movies_df['id'] = movies_df['id'].astype(str)
    credits_df['id'] = credits_df['id'].astype(str)

    # Extraer el director desde credits_df
    def extract_director(crew):
        try:
            crew_list = eval(crew)  # Convertir el string en lista
            for person in crew_list:
                if person.get('job') == 'Director':
                    return person.get('name')
        except:
            return None

    credits_df['director'] = credits_df['crew'].apply(extract_director)

    # Unir movies_df con la nueva columna 'director' usando 'id'
    credits_df = credits_df[['id', 'director']]
    movies_df = movies_df.merge(credits_df, on='id', how='left')

    # Asegurarse de que 'release_date' sea datetime y extraer 'release_year'
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
    movies_df['release_year'] = movies_df['release_date'].dt.year

    # Convertir la columna 'genres' a listas simples de nombres
    def extract_genres(genres):
        try:
            genres_list = ast.literal_eval(genres)  # Convierte el string en lista
            return [genre['name'] for genre in genres_list] if isinstance(genres_list, list) else []
        except (ValueError, SyntaxError):
            return []

    movies_df['genres'] = movies_df['genres'].apply(extract_genres)

    print("Procesamiento de datos completado.")
    print(movies_df.columns)
    print(movies_df['genres'].head(10))  # Ver los primeros valores

except Exception as e:
    print(f"Error cargando los datos: {e}")
    movies_df = None
    credits_df = None

# Ruta principal (para probar que la API funciona)
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "¡Bienvenido a la API de análisis de películas!"
    })

# Ruta para buscar películas por director
@app.route('/movies/director/<string:director_name>', methods=['GET'])
def get_movies_by_director(director_name):
    """
    Devuelve las películas dirigidas por un director específico.
    """
    if movies_df is None:
        return jsonify({"error": "Los datos no fueron cargados correctamente"}), 500

    filtered_movies = movies_df[movies_df['director'] == director_name]
    if filtered_movies.empty:
        return jsonify({"error": f"No se encontraron películas dirigidas por {director_name}"}), 404

    result = filtered_movies[['title', 'release_year', 'revenue', 'vote_average']].to_dict(orient='records')
    return jsonify(result)

# Ruta para buscar películas por género
@app.route('/movies/genre/<string:genre_name>', methods=['GET'])
def get_movies_by_genre(genre_name):
    """
    Devuelve las películas que pertenecen a un género específico.
    """
    if movies_df is None:
        return jsonify({"error": "Los datos no fueron cargados correctamente"}), 500

    # Filtrar películas por género
    filtered_movies = movies_df[movies_df['genres'].apply(lambda genres: genre_name in genres)]
    if filtered_movies.empty:
        return jsonify({"error": f"No se encontraron películas del género {genre_name}"}), 404

    result = filtered_movies[['title', 'release_year', 'revenue', 'vote_average']].to_dict(orient='records')
    return jsonify(result)

# Ruta para buscar películas por palabra clave en el título
@app.route('/movies/search/<string:keyword>', methods=['GET'])
def search_movies_by_keyword(keyword):
    """
    Devuelve las películas cuyo título contiene una palabra clave específica.
    """
    if movies_df is None:
        return jsonify({"error": "Los datos no fueron cargados correctamente"}), 500

    # Filtrar películas cuyo título contenga la palabra clave (sin importar mayúsculas/minúsculas)
    filtered_movies = movies_df[movies_df['title'].str.contains(keyword, case=False, na=False)]
    if filtered_movies.empty:
        return jsonify({"error": f"No se encontraron películas que contengan '{keyword}' en el título"}), 404

    result = filtered_movies[['title', 'release_year', 'revenue', 'vote_average']].to_dict(orient='records')
    return jsonify(result)

# Ruta para obtener las películas más populares de un año específico
@app.route('/movies/year/<int:year>', methods=['GET'])
def get_movies_by_year(year):
    """
    Devuelve las películas más populares de un año específico.
    """
    if movies_df is None:
        return jsonify({"error": "Los datos no fueron cargados correctamente"}), 500

    # Filtrar las películas por el año dado
    filtered_movies = movies_df[movies_df['release_year'] == year]

    if filtered_movies.empty:
        return jsonify({"error": f"No se encontraron películas lanzadas en el año {year}"}), 404

    # Ordenar las películas por popularidad
    filtered_movies = filtered_movies.sort_values(by='popularity', ascending=False)

    # Seleccionar columnas relevantes
    result = filtered_movies[['title', 'popularity', 'vote_average', 'revenue']].to_dict(orient='records')
    return jsonify(result)

# Ruta para obtener las películas más taquilleras por década
@app.route('/movies/top_by_decade', methods=['GET'])
def get_top_movies_by_decade():
    """
    Devuelve las películas más taquilleras por década.
    """
    if movies_df is None:
        return jsonify({"error": "Los datos no fueron cargados correctamente"}), 500

    # Crear una nueva columna de década
    movies_df['decade'] = (movies_df['release_year'] // 10) * 10

    # Filtrar solo películas con ingresos mayores a 0
    filtered_movies = movies_df[movies_df['revenue'] > 0]

    if filtered_movies.empty:
        return jsonify({"error": "No se encontraron películas con ingresos válidos"}), 404

    # Obtener la película más taquillera por cada década
    top_movies = (
        filtered_movies.loc[filtered_movies.groupby('decade')['revenue'].idxmax()]
        [['title', 'release_year', 'revenue', 'vote_average', 'decade']]
    )

    

    # Convertir el resultado a un formato JSON
    result = top_movies.sort_values(by='decade').to_dict(orient='records')
    return jsonify(result)


# Ruta para obtener los directores con más películas exitosas por calificación
@app.route('/movies/top_directors', methods=['GET'])
def get_top_directors():
    """
    Devuelve los directores con más películas exitosas (calificación > 7).
    """
    if movies_df is None:
        return jsonify({"error": "Los datos no fueron cargados correctamente"}), 500

    # Filtrar películas con calificación mayor a 7
    successful_movies = movies_df[movies_df['vote_average'] > 7]

    if successful_movies.empty:
        return jsonify({"error": "No se encontraron películas exitosas"}), 404

    # Contar el número de películas exitosas por director
    top_directors = successful_movies['director'].value_counts().head(10)

    # Convertir a un formato JSON
    result = [{"director": director, "successful_movies": count} for director, count in top_directors.items()]
    return jsonify(result)





# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)



























    

    
