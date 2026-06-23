from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.recommenders.common import (
    filter_seen_movies,
    format_recommendations,
    load_movies_catalog,
    load_user_seen_movie_ids,
)
from src.users.storage import (
    get_rated_movies_count,
    load_user_ratings,
)


@lru_cache(maxsize=1)
def load_global_svd_artifact(
    model_path: str,
) -> dict:
    """
    Загружает глобальный SVD-артефакт.
    Повторные вызовы используют модель из оперативной памяти.
    Параметры:
        model_path - Путь к сохранённому SVD-артефакту.
    Возвращает:
        Словарь с параметрами глобальной SVD-модели.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Global SVD model not found: {path}"
        )

    artifact = joblib.load(path)

    required_keys = {
        "global_mean",
        "movie_id_to_index",
        "movie_biases",
        "movie_factors",
        "rating_scale",
    }

    missing_keys = required_keys - set(artifact)

    if missing_keys:
        raise ValueError(
            f"Global SVD artifact is missing keys: {sorted(missing_keys)}"
        )

    return artifact


def build_svd_user_profile(
    global_model: dict,
    user_ratings_df: pd.DataFrame,
    regularization: float,
) -> tuple[float, np.ndarray]:
    """
    Вычисляет смещение и скрытые факторы текущего пользователя.
    Использует сохранённые факторы фильмов глобальной SVD-модели.
    Параметры:
        global_model - Словарь с параметрами глобальной SVD-модели.
        user_ratings_df - DataFrame с пользовательскими оценками.
        regularization - Коэффициент регуляризации.
    Возвращает:
        Смещение пользователя и вектор его скрытых факторов.
    """
    required_columns = {
        "movieId",
        "rating",
    }

    missing_columns = required_columns - set(user_ratings_df.columns)

    if missing_columns:
        raise ValueError(
            f"User ratings are missing columns: {sorted(missing_columns)}"
        )

    if len(user_ratings_df) == 0:
        raise ValueError("User ratings must not be empty.")

    if regularization <= 0:
        raise ValueError("SVD regularization must be positive.")

    movie_id_to_index = global_model["movie_id_to_index"]

    movie_biases = np.asarray(
        global_model["movie_biases"],
        dtype=np.float64,
    )

    movie_factors = np.asarray(
        global_model["movie_factors"],
        dtype=np.float64,
    )

    global_mean = float(global_model["global_mean"])

    rated_movie_factors = []
    rating_residuals = []

    for row in user_ratings_df.itertuples(index=False):
        movie_id = int(row.movieId)

        inner_movie_id = movie_id_to_index.get(movie_id)

        if inner_movie_id is None:
            continue

        rated_movie_factors.append(
            movie_factors[inner_movie_id]
        )

        rating_residuals.append(
            float(row.rating)
            - global_mean
            - float(movie_biases[inner_movie_id])
        )

    if len(rated_movie_factors) == 0:
        raise ValueError(
            "The movies rated by the user are missing "
            "from the global SVD model."
        )

    factors_matrix = np.vstack(rated_movie_factors)

    design_matrix = np.column_stack(
        [
            np.ones(len(factors_matrix)),
            factors_matrix,
        ]
    )

    regularization_matrix = (
        np.eye(design_matrix.shape[1])
        * regularization
    )

    parameters = np.linalg.solve(
        design_matrix.T @ design_matrix
        + regularization_matrix,
        design_matrix.T @ np.asarray(
            rating_residuals,
            dtype=np.float64,
        ),
    )

    user_bias = float(parameters[0])
    user_factors = parameters[1:]

    return user_bias, user_factors


def predict_svd_user_scores(
    global_model: dict,
    candidates_df: pd.DataFrame,
    user_bias: float,
    user_factors: np.ndarray,
) -> pd.DataFrame:
    """
    Вычисляет оценки фильмов для текущего пользователя.
    Использует глобальные факторы фильмов и пользовательский вектор.
    Параметры:
        global_model - Словарь с параметрами глобальной SVD-модели.
        candidates_df - DataFrame с фильмами-кандидатами.
        user_bias - Смещение текущего пользователя.
        user_factors - Скрытые факторы текущего пользователя.
    Возвращает:
        DataFrame с фильмами и предсказанными оценками.
    """
    movie_id_to_index = global_model["movie_id_to_index"]

    movie_biases = np.asarray(
        global_model["movie_biases"],
        dtype=np.float64,
    )

    movie_factors = np.asarray(
        global_model["movie_factors"],
        dtype=np.float64,
    )

    global_mean = float(global_model["global_mean"])

    inner_movie_ids = candidates_df["movieId"].map(
        movie_id_to_index
    )

    known_movies_mask = inner_movie_ids.notna()

    result_df = candidates_df.loc[
        known_movies_mask
    ].copy()

    if len(result_df) == 0:
        raise ValueError(
            "The catalog contains no movies known to the global SVD model."
        )

    known_inner_movie_ids = (
        inner_movie_ids.loc[known_movies_mask]
        .astype(int)
        .to_numpy()
    )

    scores = (
        global_mean
        + user_bias
        + movie_biases[known_inner_movie_ids]
        + movie_factors[known_inner_movie_ids] @ user_factors
    )

    min_rating, max_rating = global_model["rating_scale"]

    result_df["score"] = np.clip(
        scores,
        float(min_rating),
        float(max_rating),
    )

    return result_df


def get_warm_start_status(
    user_data_dir: Path,
    warm_start_threshold: int,
) -> dict:
    """
    Проверяет, хватает ли пользовательских оценок для warm-start.
    Параметры:
        user_data_dir - Папка пользовательских данных.
        warm_start_threshold - Минимальное число оценок для warm-start.
    Возвращает:
        Словарь со статусом готовности warm-start.
    """
    rated_movies_count = get_rated_movies_count(
        user_data_dir=user_data_dir,
    )

    remaining_ratings_count = max(
        warm_start_threshold - rated_movies_count,
        0,
    )

    return {
        "rated_movies_count": rated_movies_count,
        "required_ratings_count": warm_start_threshold,
        "remaining_ratings_count": remaining_ratings_count,
        "warm_start_available": (
            rated_movies_count >= warm_start_threshold
        ),
    }


def recommend_warm_start(
    movies_catalog_path: Path,
    user_data_dir: Path,
    global_svd_model_path: Path,
    regularization: float,
    top_n: int = 10,
) -> list[dict]:
    """
    Строит warm-start рекомендации через глобальную SVD-модель.
    Вычисляет факторы текущего пользователя без переобучения модели.
    Параметры:
        movies_catalog_path - Путь к movies_catalog.csv.
        user_data_dir - Папка пользовательских данных.
        global_svd_model_path - Путь к глобальному SVD-артефакту.
        regularization - Коэффициент регуляризации факторов пользователя.
        top_n - Количество рекомендаций.
    Возвращает:
        Список рекомендаций.
    """
    global_model = load_global_svd_artifact(
        str(global_svd_model_path)
    )

    user_ratings_df = load_user_ratings(user_data_dir)

    user_bias, user_factors = build_svd_user_profile(
        global_model=global_model,
        user_ratings_df=user_ratings_df,
        regularization=regularization,
    )

    movies_catalog_df = load_movies_catalog(
        movies_catalog_path
    )

    seen_movie_ids = load_user_seen_movie_ids(
        user_data_dir
    )

    candidates_df = filter_seen_movies(
        candidates_df=movies_catalog_df,
        seen_movie_ids=seen_movie_ids,
    )

    recommendations_df = predict_svd_user_scores(
        global_model=global_model,
        candidates_df=candidates_df,
        user_bias=user_bias,
        user_factors=user_factors,
    )

    recommendations_df = recommendations_df.sort_values(
        by="score",
        ascending=False,
    )

    return format_recommendations(
        recommendations_df=recommendations_df,
        score_column="score",
        top_n=top_n,
    )


def get_warm_start_recommendations(
    movies_catalog_path: Path,
    user_data_dir: Path,
    global_svd_model_path: Path,
    regularization: float,
    warm_start_threshold: int,
    top_n: int = 10,
) -> str | list[dict]:
    """
    Возвращает warm-start рекомендации или сообщение о нехватке оценок.
    Использует предварительно обученную глобальную SVD-модель.
    Параметры:
        movies_catalog_path - Путь к movies_catalog.csv.
        user_data_dir - Папка пользовательских данных.
        global_svd_model_path - Путь к глобальному SVD-артефакту.
        regularization - Коэффициент регуляризации факторов пользователя.
        warm_start_threshold - Минимальное число оценок для warm-start.
        top_n - Количество рекомендаций.
    Возвращает:
        Сообщение для пользователя или список рекомендаций.
    """
    status = get_warm_start_status(
        user_data_dir=user_data_dir,
        warm_start_threshold=warm_start_threshold,
    )

    if not status["warm_start_available"]:
        return (
            f"Оцените ещё {status['remaining_ratings_count']} "
            f"фильмов, чтобы получить персональные рекомендации."
        )

    return recommend_warm_start(
        movies_catalog_path=movies_catalog_path,
        user_data_dir=user_data_dir,
        global_svd_model_path=global_svd_model_path,
        regularization=regularization,
        top_n=top_n,
    )