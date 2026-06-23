from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from src.recommenders.common import (
    filter_seen_movies,
    format_recommendations,
    load_movies_catalog,
    load_user_seen_movie_ids,
)
from src.utils.io import load_json


def load_catboost_model(model_path: Path) -> CatBoostRegressor:
    """
    Загружает сохранённую CatBoost-модель.
    Используется для cold-start рекомендаций.
    Параметры:
        model_path - Путь к файлу CatBoost-модели.
    Возвращает:
        Загруженную CatBoostRegressor-модель.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model not found: {model_path}")

    model = CatBoostRegressor()
    model.load_model(str(model_path))

    return model


def load_catboost_metadata(metadata_path: Path) -> dict:
    """
    Загружает metadata CatBoost-модели.
    Metadata содержит список признаков, параметры и метрики модели.
    Параметры:
        metadata_path - Путь к catboost_metadata.json.
    Возвращает:
        Словарь с metadata CatBoost.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"CatBoost metadata not found: {metadata_path}")

    metadata = load_json(metadata_path)

    if "feature_columns" not in metadata:
        raise ValueError("CatBoost metadata must contain feature_columns.")

    return metadata


def build_cold_start_features(
    movies_catalog_df: pd.DataFrame,
    questionnaire: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Строит признаки для CatBoost по каталогу фильмов и анкете пользователя.
    Порядок колонок соответствует feature_columns из metadata модели.
    Параметры:
        movies_catalog_df - Production catalog с фильмами.
        questionnaire - Пользовательская анкета жанр-рейтинг.
        feature_columns - Список признаков CatBoost.
    Возвращает:
        DataFrame с признаками для CatBoost.
    """
    if len(questionnaire) == 0:
        raise ValueError("User questionnaire must not be empty.")

    features_df = pd.DataFrame(index=movies_catalog_df.index)

    questionnaire_mean = sum(questionnaire.values()) / len(questionnaire)

    for column in feature_columns:
        if column in movies_catalog_df.columns:
            features_df[column] = movies_catalog_df[column]

        elif column == "user_avg_rating":
            features_df[column] = questionnaire_mean

        elif column.startswith("user_avg_"):
            genre = column.replace("user_avg_", "")

            if genre not in questionnaire:
                raise ValueError(f"User questionnaire is missing genre: {genre}")

            features_df[column] = questionnaire[genre]

        else:
            raise ValueError(f"Cannot build CatBoost feature column: {column}")

    return features_df[feature_columns]


def filter_movies_by_genre_ratings(
    candidates_df: pd.DataFrame,
    questionnaire: dict[str, float],
    min_genre_rating: float,
) -> pd.DataFrame:
    """
    Убирает фильмы с жанрами, оценёнными ниже допустимого значения.
    Фильм исключается, если хотя бы один его жанр имеет низкую оценку.
    Параметры:
        candidates_df - DataFrame с фильмами-кандидатами.
        questionnaire - Пользовательская анкета жанр-рейтинг.
        min_genre_rating - Минимальная допустимая оценка жанра.
    Возвращает:
        DataFrame без фильмов с нежелательными жанрами.
    """
    disliked_genres = [
        genre
        for genre, rating in questionnaire.items()
        if rating < min_genre_rating and genre in candidates_df.columns
    ]

    if len(disliked_genres) == 0:
        return candidates_df.copy()

    disliked_movies_mask = candidates_df[disliked_genres].eq(1).any(axis=1)

    return candidates_df[~disliked_movies_mask].copy()


def recommend_cold_start(
    movies_catalog_path: Path,
    user_data_dir: Path,
    catboost_model_path: Path,
    catboost_metadata_path: Path,
    top_n: int = 10,
    min_genre_rating: float = 2.5,
) -> list[dict]:
    """
    Строит cold-start рекомендации через CatBoost.
    Использует пользовательскую анкету и исключает уже оценённые фильмы.
    Параметры:
        movies_catalog_path - Путь к movies_catalog.csv.
        user_data_dir - Папка пользовательских данных.
        catboost_model_path - Путь к CatBoost-модели.
        catboost_metadata_path - Путь к metadata CatBoost.
        top_n - Количество рекомендаций.
        min_genre_rating - Минимальная оценка жанра для попадания в рекомендации
    Возвращает:
        Список рекомендаций.
    """
    movies_catalog_df = load_movies_catalog(movies_catalog_path)
    questionnaire = load_json(user_data_dir / "user_questionnaire.json")

    model = load_catboost_model(catboost_model_path)
    metadata = load_catboost_metadata(catboost_metadata_path)

    feature_columns = metadata["feature_columns"]

    seen_movie_ids = load_user_seen_movie_ids(user_data_dir)

    candidates_df = filter_seen_movies(
        candidates_df=movies_catalog_df,
        seen_movie_ids=seen_movie_ids,
    )

    candidates_df = filter_movies_by_genre_ratings(
        candidates_df=candidates_df,
        questionnaire=questionnaire,
        min_genre_rating=min_genre_rating,
    )

    features_df = build_cold_start_features(
        movies_catalog_df=candidates_df,
        questionnaire=questionnaire,
        feature_columns=feature_columns,
    )

    candidates_df = candidates_df.copy()
    candidates_df["score"] = model.predict(features_df)

    recommendations_df = candidates_df.sort_values(
        by="score",
        ascending=False,
    )

    return format_recommendations(
        recommendations_df=recommendations_df,
        score_column="score",
        top_n=top_n,
    )