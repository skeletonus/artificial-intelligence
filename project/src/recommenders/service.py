from src.recommenders.cold_start import recommend_cold_start
from src.recommenders.warm_start import get_warm_start_recommendations
from src.users.storage import (
    load_user_ratings,
    save_user_questionnaire,
    save_user_ratings,
)
from src.utils.config import resolve_project_path
from src.utils.io import load_json
from src.recommenders.common import (
    load_movies_catalog,
    search_movies_by_title,
)
from src.users.validation import (
    validate_min_genre_rating,
    validate_movie_search_query,
    validate_user_questionnaire,
)


def run_cold_start_recommendations(
    config: dict,
    questionnaire: dict,
    top_n: int = 10,
    min_genre_rating: float = 2.5,
) -> list[dict]:
    """
    Валидирует и сохраняет анкету пользователя, затем возвращает cold-start рекомендации.
    Используется для отдельного cold-start окна.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
        questionnaire - Словарь жанр-рейтинг от пользователя.
        top_n - Количество рекомендаций.
        min_genre_rating - Минимальная оценка жанра для попадания в рекомендации
    Возвращает:
        Список рекомендаций.
    """
    genre_columns = load_json(
        resolve_project_path(config["paths"]["genre_columns_path"])
    )

    user_data_dir = resolve_project_path(
        config["paths"]["user_data_dir"]
    )

    validated_questionnaire = validate_user_questionnaire(
        questionnaire=questionnaire,
        genre_columns=genre_columns,
        min_rating=config["rating"]["min_rating"],
        max_rating=config["rating"]["max_rating"],
    )

    validated_min_genre_rating = validate_min_genre_rating(
        min_genre_rating=min_genre_rating,
        questionnaire=validated_questionnaire,
        min_rating=config["rating"]["min_rating"],
        max_rating=config["rating"]["max_rating"],
    )

    save_user_questionnaire(
        user_data_dir=user_data_dir,
        questionnaire=validated_questionnaire,
    )

    return recommend_cold_start(
        movies_catalog_path=resolve_project_path(
            config["paths"]["movies_catalog_path"]
        ),
        user_data_dir=user_data_dir,
        catboost_model_path=resolve_project_path(
            config["model"]["catboost_model_path"]
        ),
        catboost_metadata_path=resolve_project_path(
            config["model"]["catboost_metadata_path"]
        ),
        top_n=top_n,
        min_genre_rating=validated_min_genre_rating,
    )


def run_movie_search(
    config: dict,
    query: str,
    limit: int = 20,
) -> list[dict]:
    """
    Валидирует строку поиска и возвращает фильмы из каталога.
    Сначала возвращает фильмы, название которых начинается со строки поиска.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
        query - Строка поиска фильма.
        limit - Максимальное количество найденных фильмов.
    Возвращает:
        Список найденных фильмов.
    """
    validated_query = validate_movie_search_query(query)

    movies_catalog_df = load_movies_catalog(
        resolve_project_path(config["paths"]["movies_catalog_path"])
    )

    movies_df = search_movies_by_title(
        movies_catalog_df=movies_catalog_df,
        query=validated_query,
        limit=limit,
    )

    if len(movies_df) == 0:
        raise LookupError("No movies found.")

    movies = []

    for _, row in movies_df.iterrows():
        movies.append(
            {
                "movieId": int(row["movieId"]),
                "title": str(row["title"]),
            }
        )

    return movies


def get_user_ratings_with_titles(
    config: dict,
) -> list[dict]:
    """
    Возвращает пользовательские оценки вместе с названиями фильмов.
    Названия подтягиваются из каталога по movieId.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
    Возвращает:
        Список оценённых фильмов с movieId, title и rating.
    """
    user_ratings_df = load_user_ratings(
        resolve_project_path(
            config["paths"]["user_data_dir"]
        )
    )

    movies_catalog_df = load_movies_catalog(
        resolve_project_path(
            config["paths"]["movies_catalog_path"]
        )
    )

    ratings_with_titles_df = user_ratings_df.merge(
        movies_catalog_df[["movieId", "title"]],
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    ratings_with_titles_df = ratings_with_titles_df[
        ["movieId", "title", "rating"]
    ]

    return ratings_with_titles_df.to_dict(
        orient="records"
    )


def save_ratings_window(
    config: dict,
    ratings: dict[int, float],
) -> str:
    """
    Валидирует и сохраняет пользовательские оценки фильмов.
    Используется для отдельного окна оценки фильмов.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
        ratings - Словарь movieId-рейтинг от пользователя.
    Возвращает:
        Сообщение о результате сохранения.
    """
    movies_catalog_df = load_movies_catalog(
        resolve_project_path(config["paths"]["movies_catalog_path"])
    )

    save_user_ratings(
        user_data_dir=resolve_project_path(config["paths"]["user_data_dir"]),
        user_ratings=ratings,
        movies_catalog_df=movies_catalog_df,
        min_rating=config["rating"]["min_rating"],
        max_rating=config["rating"]["max_rating"],
    )

    return "Оценки сохранены."


def run_warm_start_recommendations(
    config: dict,
    top_n: int = 10,
) -> str | list[dict]:
    """
    Возвращает warm-start рекомендации или сообщение о нехватке оценок.
    Использует предварительно обученную глобальную SVD-модель.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
        top_n - Количество рекомендаций.
    Возвращает:
        Сообщение для пользователя или список рекомендаций.
    """
    return get_warm_start_recommendations(
        movies_catalog_path=resolve_project_path(config["paths"]["movies_catalog_path"]),
        user_data_dir=resolve_project_path(config["paths"]["user_data_dir"]),
        global_svd_model_path=resolve_project_path(config["model"]["svd_global_model_path"]),
        regularization=config["svd"]["reg_all"],
        warm_start_threshold=config["warm_start"]["threshold"],
        top_n=top_n,
    )