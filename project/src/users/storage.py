from pathlib import Path

import pandas as pd

from src.utils.io import load_json, save_json

from src.users.validation import validate_user_questionnaire, validate_user_ratings


def build_default_questionnaire(
    movies_catalog_df: pd.DataFrame,
    genre_columns: list[str],
    default_rating: float = 3.5,
) -> dict[str, float]:
    """
    Создаёт стартовую анкету пользователя со средними оценками жанров.
    Для каждого жанра берётся средний movie_avg_rating среди фильмов этого жанра.
    Параметры:
        movies_catalog_df - Каталог фильмов с жанровыми колонками и movie_avg_rating.
        genre_columns - Список жанровых колонок.
        default_rating - Значение по умолчанию, если для жанра нет фильмов.
    Возвращает:
        Словарь жанр-рейтинг.
    """
    questionnaire = {}

    for genre in genre_columns:
        genre_movies = movies_catalog_df[movies_catalog_df[genre] == 1]

        if len(genre_movies) == 0:
            questionnaire[genre] = round(float(default_rating), 1)
        else:
            questionnaire[genre] = round(
                float(genre_movies["movie_avg_rating"].mean()),
                1,
            )

    return questionnaire


def initialize_user_storage(
    user_data_dir: Path,
    movies_catalog_df: pd.DataFrame,
    genre_columns: list[str],
    default_rating: float = 3.5,
) -> None:
    """
    Создаёт папку user_data и стартовые файлы пользователя.
    Анкета пользователя инициализируется средними оценками жанров.
    Параметры:
        user_data_dir - Папка для пользовательских данных.
        movies_catalog_df - Каталог фильмов с жанровыми колонками и movie_avg_rating.
        genre_columns - Список жанров для анкеты пользователя.
        default_rating - Значение по умолчанию, если для жанра нет фильмов.
    Возвращает:
        None.
    """
    user_data_dir.mkdir(parents=True, exist_ok=True)

    questionnaire_path = user_data_dir / "user_questionnaire.json"
    ratings_path = user_data_dir / "user_ratings.csv"

    if not questionnaire_path.exists():
        questionnaire = build_default_questionnaire(
            movies_catalog_df=movies_catalog_df,
            genre_columns=genre_columns,
            default_rating=default_rating,
        )
        save_json(questionnaire, questionnaire_path)

    if not ratings_path.exists():
        ratings_df = pd.DataFrame(
            columns=[
                "movieId",
                "rating",
            ]
        )
        ratings_df.to_csv(ratings_path, index=False)


def load_user_questionnaire(user_data_dir: Path) -> dict[str, float]:
    """
    Загружает анкету пользователя из user_questionnaire.json.
    Анкета хранит пользовательские оценки жанров.
    Параметры:
        user_data_dir - Папка пользовательских данных.
    Возвращает:
        Словарь жанр-рейтинг.
    """
    return load_json(user_data_dir / "user_questionnaire.json")


def save_user_questionnaire(
    user_data_dir: Path,
    questionnaire: dict[str, float],
) -> None:
    """
    Сохраняет валидированную анкету пользователя в user_questionnaire.json.
    Используется после проверки cold-start анкеты.
    Параметры:
        user_data_dir - Папка пользовательских данных.
        questionnaire - Валидированный словарь жанр-рейтинг.
    Возвращает:
        None.
    """
    save_json(
        data=questionnaire,
        path=user_data_dir / "user_questionnaire.json",
    )


def load_user_ratings(user_data_dir: Path) -> pd.DataFrame:
    """
    Загружает оценки пользователя из user_ratings.csv.
    Используется для warm-start и исключения уже оценённых фильмов из рекомендаций.
    Параметры:
        user_data_dir - Папка пользовательских данных.
    Возвращает:
        DataFrame с пользовательскими оценками.
    """
    return pd.read_csv(
        user_data_dir / "user_ratings.csv",
        dtype={
            "movieId": "int64",
            "rating": "float64",
        },
    )


def save_user_ratings(
    user_data_dir: Path,
    user_ratings: dict[int, float],
    movies_catalog_df: pd.DataFrame,
    min_rating: float,
    max_rating: float,
) -> None:
    """
    Валидирует и сохраняет пользовательские оценки фильмов.
    Пользователь передаёт словарь movieId-рейтинг.
    Параметры:
        user_data_dir - Папка пользовательских данных.
        user_ratings - Словарь movieId-рейтинг.
        movies_catalog_df - Каталог фильмов с колонками movieId и title.
        min_rating - Минимально допустимая оценка.
        max_rating - Максимально допустимая оценка.
    Возвращает:
        None.
    """
    ratings_path = user_data_dir / "user_ratings.csv"

    existing_ratings_df = load_user_ratings(user_data_dir)

    validated_ratings_df = validate_user_ratings(
        user_ratings=user_ratings,
        movies_catalog_df=movies_catalog_df,
        min_rating=min_rating,
        max_rating=max_rating,
    )

    existing_ratings_df = existing_ratings_df[
        ~existing_ratings_df["movieId"].isin(validated_ratings_df["movieId"])
    ]

    ratings_df = pd.concat(
        [existing_ratings_df, validated_ratings_df],
        axis=0,
        ignore_index=True,
    )

    ratings_df = ratings_df[["movieId", "rating"]]

    ratings_df.to_csv(ratings_path, index=False)


def get_rated_movies_count(user_data_dir: Path) -> int:
    """
    Считает количество фильмов, оценённых пользователем.
    Используется для проверки готовности warm-start.
    Параметры:
        user_data_dir - Папка пользовательских данных.
    Возвращает:
        Количество уникальных оценённых фильмов.
    """
    ratings_df = load_user_ratings(user_data_dir)
    return int(ratings_df["movieId"].nunique())