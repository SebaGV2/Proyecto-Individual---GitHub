import pandas as pd

# Reducir el archivo 'credits_reduced_5000.csv'
credits_path = './data/credits_reduced_5000.csv'
credits_reduced_path = './data/credits_reduced_1000.csv'

try:
    credits_df = pd.read_csv(credits_path)
    # Mantener solo las columnas necesarias
    credits_df = credits_df[['id', 'crew']]
    # Reducir a las primeras 1,000 filas
    credits_reduced_df = credits_df.head(1000)
    # Guardar el archivo reducido
    credits_reduced_df.to_csv(credits_reduced_path, index=False)
    print(f"'credits_reduced_5000.csv' reducido guardado como {credits_reduced_path}")
except Exception as e:
    print(f"Error al procesar 'credits_reduced_5000.csv': {e}")

# Reducir el archivo 'movies_dataset_reduced_5000.csv'
movies_path = './data/movies_dataset_reduced_5000.csv'
movies_reduced_path = './data/movies_dataset_reduced_1000.csv'

try:
    movies_df = pd.read_csv(movies_path)
    # Mantener solo las columnas necesarias
    columns_to_keep = ['id', 'title', 'genres', 'release_date', 'revenue', 'budget', 'vote_average', 'popularity']
    movies_df = movies_df[columns_to_keep]
    # Reducir a las primeras 1,000 filas
    movies_reduced_df = movies_df.head(1000)
    # Guardar el archivo reducido
    movies_reduced_df.to_csv(movies_reduced_path, index=False)
    print(f"'movies_dataset_reduced_5000.csv' reducido guardado como {movies_reduced_path}")
except Exception as e:
    print(f"Error al procesar 'movies_dataset_reduced_5000.csv': {e}")
