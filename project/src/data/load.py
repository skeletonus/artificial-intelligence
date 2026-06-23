from pathlib import Path

import pandas as pd


def load_movielens(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает основные таблицы MovieLens: ratings.csv и movies.csv.
    Все целочисленные численные колонки приводит к int32, а rating — к float32.
    Параметры:
        raw_dir - Путь к папке raw, где лежат CSV-файлы MovieLens.
    Возвращает:
        Кортеж из двух DataFrame: ratings_df и movies_df.
    """
    ratings_df = pd.read_csv(raw_dir / "ratings.csv")
    movies_df = pd.read_csv(raw_dir / "movies.csv")

    for column in ratings_df.select_dtypes(include="integer").columns:
        ratings_df[column] = ratings_df[column].astype("int32")

    for column in movies_df.select_dtypes(include="integer").columns:
        movies_df[column] = movies_df[column].astype("int32")

    ratings_df["rating"] = ratings_df["rating"].astype("float32")

    return ratings_df, movies_df