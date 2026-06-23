from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_json


def build_movies_catalog(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    lambda_: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Создаёт справочник фильмов с one-hot жанрами, количеством оценок и регуляризованной средней оценкой.
    Средняя оценка сглаживается через жанровый prior, чтобы фильмы с малым числом оценок не получали слишком шумные значения.
    Параметры:
        movies_df - DataFrame с фильмами из movies.csv.
        ratings_df - DataFrame с рейтингами из ratings.csv.
        lambda_ - Сила регуляризации для средней оценки фильма.
    Возвращает:
        Кортеж из DataFrame со справочником фильмов и списка жанровых колонок.
    """
    genres_one_hot = (
        movies_df["genres"]
        .str.get_dummies(sep="|")
        .drop(columns=["(no genres listed)"], errors="ignore")
        .astype("int8")
    )

    genre_columns = genres_one_hot.columns.tolist()

    movies_catalog_df = pd.concat(
        [
            movies_df[["movieId", "title"]].reset_index(drop=True),
            genres_one_hot.reset_index(drop=True),
        ],
        axis=1,
    )

    movie_stats_df = (
        ratings_df.groupby("movieId")["rating"]
        .agg(rating_sum="sum", rating_count="count")
        .reset_index()
    )

    movie_stats_df["rating_sum"] = movie_stats_df["rating_sum"].astype("float32")
    movie_stats_df["rating_count"] = movie_stats_df["rating_count"].astype("int32")

    movies_catalog_df = movies_catalog_df.merge(
        movie_stats_df,
        on="movieId",
        how="left",
    )

    movies_catalog_df["rating_sum"] = (
        movies_catalog_df["rating_sum"].fillna(0).astype("float32")
    )
    movies_catalog_df["rating_count"] = (
        movies_catalog_df["rating_count"].fillna(0).astype("int32")
    )

    global_mean = float(ratings_df["rating"].mean())

    genre_means = {}

    for genre in genre_columns:
        genre_movies_mask = movies_catalog_df[genre] == 1
        genre_rating_count = movies_catalog_df.loc[
            genre_movies_mask,
            "rating_count",
        ].sum()

        if genre_rating_count > 0:
            genre_rating_sum = movies_catalog_df.loc[
                genre_movies_mask,
                "rating_sum",
            ].sum()

            genre_means[genre] = float(genre_rating_sum / genre_rating_count)
        else:
            genre_means[genre] = global_mean

    genre_matrix = movies_catalog_df[genre_columns].to_numpy(dtype="float32")
    genre_values = np.array(
        [genre_means[genre] for genre in genre_columns],
        dtype="float32",
    )

    genre_sum = genre_matrix @ genre_values
    genre_count = genre_matrix.sum(axis=1)

    prior = np.divide(
        genre_sum,
        genre_count,
        out=np.full(len(movies_catalog_df), global_mean, dtype="float32"),
        where=genre_count > 0,
    )

    movies_catalog_df["movie_avg_rating"] = (
        (
            movies_catalog_df["rating_sum"].to_numpy(dtype="float32")
            + lambda_ * prior
        )
        / (
            movies_catalog_df["rating_count"].to_numpy(dtype="float32")
            + lambda_
        )
    ).astype("float32")

    movies_catalog_df = movies_catalog_df[
        ["movieId", "title", "movie_avg_rating", "rating_count"] + genre_columns
    ].copy()

    return movies_catalog_df, genre_columns


def save_movies_catalog(
    movies_catalog_df: pd.DataFrame,
    genre_columns: list[str],
    movies_catalog_path: Path,
    genre_columns_path: Path,
) -> None:
    """
    Сохраняет справочник фильмов в CSV и список жанровых колонок в JSON.
    Эти файлы потом используются в production-коде для cold-start рекомендаций.
    Параметры:
        movies_catalog_df - DataFrame со справочником фильмов.
        genre_columns - Список названий жанровых колонок.
        movies_catalog_path - Путь для сохранения movies_catalog.csv.
        genre_columns_path - Путь для сохранения genre_columns.json.
    Возвращает:
        None.
    """
    movies_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    genre_columns_path.parent.mkdir(parents=True, exist_ok=True)

    movies_catalog_df.to_csv(movies_catalog_path, index=False)

    save_json(
        data=genre_columns,
        path=genre_columns_path,
    )

    print(f"Movies catalog saved to: {movies_catalog_path}")
    print(f"Genre columns saved to: {genre_columns_path}")