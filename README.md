# Proyecto de API de Análisis de Películas 🎥

Este proyecto implementa una API desarrollada en Python utilizando Flask para analizar datos de películas. Proporciona información sobre películas según diversas métricas como director, género, popularidad, entre otras.

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

## Instalación 🔧

1. Clona el repositorio o descarga el código fuente.
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd Proyecto

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