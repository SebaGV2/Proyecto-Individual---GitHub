# Proyecto de API de Análisis de Películas 🎥

Esta API permite analizar información sobre películas a partir de un conjunto de datos reducido. Incluye funcionalidades como la búsqueda de películas por director, género o palabra clave, y permite explorar información de taquilla y popularidad. La API está desplegada en Render y es accesible públicamente.

URL base de la API desplegada:
https://proyecto-individual-github.onrender.com

## Estructura del Proyecto 📁

La estructura básica del proyecto es la siguiente:

Proyecto/ 
├── data/
│ ├── movies_dataset.csv # Datos principales de las películas 
│ ├── credits.csv # Información sobre el elenco y equipo 
├── scripts/
│ ├── main.py # Archivo de procesamiento de datos 
├── app.py # Código principal de la API 
├── README.md # Documentación del proyecto


## Requisitos del Entorno 🛠️

Antes de ejecutar este proyecto, asegúrate de tener instalado:

- Python 3.8 o superior
- `pip` para instalar dependencias

Requisitos Previos

Para ejecutar este proyecto localmente necesitarás:

Python 3.10 o superior.

Librerías: Flask, pandas.

Archivo requirements.txt con las dependencias necesarias.

Archivos de datos reducidos: credits_reduced.csv y movies_dataset_reduced.csv.


## Generación de archivos de datos reducidos

Debido a las limitaciones de almacenamiento, los archivos de datos no se incluyen directamente en este repositorio. Sin embargo, puedes generarlos localmente siguiendo estos pasos:

1. Asegúrate de tener los archivos de datos originales:
   - **credits.csv**
   - **movies_dataset.csv**

   Estos archivos deben estar colocados en la carpeta `data/` dentro del proyecto. Si no los tienes, puedes descargarlos desde la fuente original o solicitarlos.

2. Ejecuta el script `reduce_data.py` para generar los archivos reducidos:
   ```bash
   python reduce_data.py

   ### Generar los Archivos Reducidos

Antes de ejecutar la API, asegúrate de generar los archivos reducidos ejecutando el siguiente comando:

```bash
python reduce_data.py

Rutas de la API

1. Búsqueda de Películas por Director

Endpoint: /movies/director/<string:director_name>

Descripción: Devuelve todas las películas dirigidas por un director específico.

Parámetros:

director_name (string): Nombre del director a buscar.

Método: GET

Ejemplo de Uso:
https://proyecto-individual-github.onrender.com/movies/director/James%20Cameron

2. Búsqueda de Películas por Género

Endpoint: /movies/genre/<string:genre_name>

Descripción: Devuelve todas las películas que pertenecen a un género específico.

Parámetros:

genre_name (string): Nombre del género a buscar (en inglés, por ejemplo, Action, Drama).

Método: GET

Ejemplo de Uso:
https://proyecto-individual-github.onrender.com/movies/genre/Action

3. Búsqueda de Películas por Palabra Clave

Endpoint: /movies/search/<string:keyword>

Descripción: Devuelve todas las películas cuyo título contenga una palabra clave específica.

Parámetros:

keyword (string): Palabra clave para buscar en los títulos de las películas.

Método: GET

Ejemplo de Uso:
https://proyecto-individual-github.onrender.com/movies/search/Love

4. Películas Más Populares de un Año

Endpoint: /movies/year/<int:year>

Descripción: Devuelve las películas más populares lanzadas en un año específico.

Parámetros:

year (int): Año para buscar las películas.

Método: GET

Ejemplo de Uso:
https://proyecto-individual-github.onrender.com/movies/year/1960

5. Películas Más Taquilleras por Década

Endpoint: /movies/top_by_decade

Descripción: Devuelve la película más taquillera de cada década.

Método: GET

Ejemplo de Uso:
https://proyecto-individual-github.onrender.com/movies/top_by_decade

6. Directores con Más Películas Exitosas

Endpoint: /movies/top_directors

Descripción: Devuelve los 10 directores con más películas exitosas (calificación > 7).

Método: GET

Ejemplo de Uso:
https://proyecto-individual-github.onrender.com/movies/top_directors

Despliegue

La API está desplegada en Render. Para desplegarla localmente, sigue estos pasos:

Clona el repositorio:

git clone <https://github.com/SebaGV2/Proyecto-Individual---GitHub.git>

Instala las dependencias:

pip install -r requirements.txt

Ejecuta la aplicación:

python app.py

Accede a la API en http://127.0.0.1:5000.

Autor

SebaGV2

¡Gracias por usar esta API!


#Encontrar peliculas segun director
https://proyecto-individual-github.onrender.com/movies/director/Frank%20Capra

#Encontrar peliculas segun genero
https://proyecto-individual-github.onrender.com/movies/genre/Action

#Encontrar peliculas mediante palabras o letras clave
https://proyecto-individual-github.onrender.com/movies/search/w

# Peliculas mas populares de cada año
https://proyecto-individual-github.onrender.com/movies/year/1955

# Encontrar la película más popular por década
https://proyecto-individual-github.onrender.com/movies/top_by_decade

# Directores con más películas exitosas por calificación
https://proyecto-individual-github.onrender.com/movies/top_directors  

URL video explicativo Youtube
https://www.youtube.com/watch?v=8EqhyzlxEIE

