from math import isfinite
from numbers import Number

import pandas as pd

def validate_numeric_rating(
    rating: object,
    field_name: str,
    min_rating: float,
    max_rating: float,
) -> float:
    """
    Валидирует числовую оценку и приводит её к float.
    Проверяет тип, конечность значения и допустимый диапазон.
    Параметры:
        rating - Проверяемая оценка.
        field_name - Название поля для сообщения об ошибке.
        min_rating - Минимально допустимая оценка.
        max_rating - Максимально допустимая оценка.
    Возвращает:
        Валидированную оценку типа float.
    """
    if isinstance(rating, bool) or not isinstance(rating, Number):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    rating = float(rating)

    if not isfinite(rating):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if rating < min_rating or rating > max_rating:
        raise ValueError(
            f"{field_name} must be between "
            f"{min_rating} and {max_rating}."
        )

    return rating


def validate_user_questionnaire(
    questionnaire: dict,
    genre_columns: list[str],
    min_rating: float,
    max_rating: float,
) -> dict[str, float]:
    """
    Валидирует пользовательскую анкету с оценками жанров.
    Проверяет набор жанров, типы значений и диапазон оценок.
    Параметры:
        questionnaire - Словарь жанр-рейтинг, полученный от пользователя.
        genre_columns - Список допустимых жанров.
        min_rating - Минимально допустимая оценка.
        max_rating - Максимально допустимая оценка.
    Возвращает:
        Валидированный словарь жанр-рейтинг.
    """
    if not isinstance(questionnaire, dict):
        raise ValueError("Questionnaire must be a dictionary.")

    expected_genres = set(genre_columns)
    received_genres = set(questionnaire.keys())

    missing_genres = expected_genres - received_genres
    extra_genres = received_genres - expected_genres

    if missing_genres:
        raise ValueError(f"Missing genres in questionnaire: {sorted(missing_genres)}")

    if extra_genres:
        raise ValueError(f"Unknown genres in questionnaire: {sorted(extra_genres)}")

    validated_questionnaire = {}

    for genre, rating in questionnaire.items():
        rating = validate_numeric_rating(
            rating=rating,
            field_name=f"Rating for genre '{genre}'",
            min_rating=min_rating,
            max_rating=max_rating,
        )

        validated_questionnaire[genre] = rating

    return validated_questionnaire

def validate_min_genre_rating(
    min_genre_rating: float,
    questionnaire: dict,
    min_rating: float,
    max_rating: float,
) -> float:
    """
    Валидирует минимальную допустимую оценку жанра.
    Проверяет тип, диапазон и максимальную оценку из анкеты пользователя.
    Параметры:
        min_genre_rating - Минимальная допустимая оценка жанра.
        questionnaire - Словарь жанр-рейтинг от пользователя.
        min_rating - Минимально допустимая оценка.
        max_rating - Максимально допустимая оценка.
    Возвращает:
        Валидированную минимальную оценку жанра.
    """
    min_genre_rating = validate_numeric_rating(
        rating=min_genre_rating,
        field_name="Minimum genre rating",
        min_rating=min_rating,
        max_rating=max_rating,
    )

    if not isinstance(questionnaire, dict) or len(questionnaire) == 0:
        raise ValueError("Questionnaire must not be empty.")

    questionnaire_max_rating = max(questionnaire.values())

    if min_genre_rating > questionnaire_max_rating:
        raise ValueError(
            "Minimum genre rating must not be greater than "
            f"the maximum questionnaire rating: {questionnaire_max_rating}."
        )

    return min_genre_rating


def validate_movie_search_query(query: str) -> str:
    """
    Валидирует строку поиска фильма.
    Проверяет тип и наличие текста для поиска.
    Параметры:
        query - Строка поиска, полученная от пользователя.
    Возвращает:
        Очищенную строку поиска.
    """
    if not isinstance(query, str):
        raise ValueError("No movies found.")

    query = query.strip()

    if query == "":
        raise ValueError("No movies found.")

    return query


def validate_user_ratings(
    user_ratings: dict[int, float],
    movies_catalog_df: pd.DataFrame,
    min_rating: float,
    max_rating: float,
) -> pd.DataFrame:
    """
    Валидирует пользовательские оценки фильмов.
    Проверяет идентификаторы фильмов, типы значений и диапазон оценок..
    Параметры:
        user_ratings - Словарь movieId-рейтинг, полученный от пользователя.
        movies_catalog_df - Каталог фильмов с колонками movieId и title.
        min_rating - Минимально допустимая оценка.
        max_rating - Максимально допустимая оценка.
    Возвращает:
        DataFrame с колонками movieId, title и rating.
    """
    if not isinstance(user_ratings, dict):
        raise ValueError("User ratings must be a dictionary.")

    if not user_ratings:
        raise ValueError("User ratings must not be empty.")
    catalog_movie_ids = set(
        movies_catalog_df["movieId"].astype(int)
    )

    validated_rows = []

    for movie_id, rating in user_ratings.items():
        if isinstance(movie_id, bool) or not isinstance(movie_id, Number):
            raise ValueError("Movie ID must be an integer.")

        movie_id = int(movie_id)

        if movie_id not in catalog_movie_ids:
            raise ValueError(
                f"Unknown movie ID: {movie_id}"
            )

        rating = validate_numeric_rating(
            rating=rating,
            field_name=f"Rating for movie {movie_id}",
            min_rating=min_rating,
            max_rating=max_rating,
        )

        validated_rows.append(
            {
                "movieId": movie_id,
                "rating": rating,
            }
        )

    return pd.DataFrame(validated_rows)