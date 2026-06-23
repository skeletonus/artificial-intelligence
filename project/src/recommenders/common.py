from pathlib import Path

import pandas as pd


def load_movies_catalog(movies_catalog_path: Path) -> pd.DataFrame:
    """
    Загружает production catalog с фильмами.
    Используется recommender-слоем для построения кандидатов.
    Параметры:
        movies_catalog_path - Путь к movies_catalog.csv.
    Возвращает:
        DataFrame с каталогом фильмов.
    """
    if not movies_catalog_path.exists():
        raise FileNotFoundError(f"Movies catalog not found: {movies_catalog_path}")

    return pd.read_csv(movies_catalog_path)


def load_user_seen_movie_ids(user_data_dir: Path) -> set[int]:
    """
    Загружает id фильмов, которые пользователь уже оценил.
    Используется для исключения просмотренных фильмов из рекомендаций.
    Параметры:
        user_data_dir - Папка пользовательских данных.
    Возвращает:
        Множество movieId уже оценённых фильмов.
    """
    ratings_path = user_data_dir / "user_ratings.csv"

    if not ratings_path.exists():
        return set()

    ratings_df = pd.read_csv(ratings_path)

    if len(ratings_df) == 0:
        return set()

    return set(ratings_df["movieId"].astype(int))


def filter_seen_movies(
    candidates_df: pd.DataFrame,
    seen_movie_ids: set[int],
) -> pd.DataFrame:
    """
    Убирает из кандидатов фильмы, которые пользователь уже оценил.
    Используется перед предсказанием модели.
    Параметры:
        candidates_df - DataFrame с фильмами-кандидатами.
        seen_movie_ids - Множество movieId уже оценённых фильмов.
    Возвращает:
        DataFrame без уже оценённых фильмов.
    """
    if len(seen_movie_ids) == 0:
        return candidates_df.copy()

    return candidates_df[~candidates_df["movieId"].isin(seen_movie_ids)].copy()


def format_recommendations(
    recommendations_df: pd.DataFrame,
    score_column: str,
    top_n: int,
) -> list[dict]:
    """
    Преобразует DataFrame с рекомендациями в список словарей для API.
    Оставляет movieId, title и score.
    Параметры:
        recommendations_df - DataFrame с отсортированными рекомендациями.
        score_column - Название колонки с предсказанной оценкой.
        top_n - Количество рекомендаций.
    Возвращает:
        Список рекомендаций для API-ответа.
    """
    result_df = recommendations_df.head(top_n).copy()

    recommendations = []

    for _, row in result_df.iterrows():
        recommendations.append(
            {
                "movieId": int(row["movieId"]),
                "title": str(row["title"]),
                "score": float(row[score_column]),
            }
        )

    return recommendations


def search_movies_by_title(
    movies_catalog_df: pd.DataFrame,
    query: str,
    limit: int,
) -> pd.DataFrame:
    """
    Ищет фильмы в каталоге по части названия.
    Сначала возвращает фильмы, названия которых начинаются со строки поиска,
    затем фильмы, названия которых содержат строку поиска.
    Параметры:
        movies_catalog_df - Каталог фильмов с колонками movieId и title.
        query - Строка поиска фильма.
        limit - Максимальное количество найденных фильмов.
    Возвращает:
        DataFrame с найденными фильмами.
    """
    normalized_query = query.casefold()

    normalized_titles = (
        movies_catalog_df["title"]
        .astype(str)
        .str.casefold()
    )

    starts_with_mask = normalized_titles.str.startswith(
        normalized_query,
        na=False,
    )

    contains_mask = normalized_titles.str.contains(
        normalized_query,
        regex=False,
        na=False,
    )

    starts_with_df = movies_catalog_df.loc[starts_with_mask]

    contains_df = movies_catalog_df.loc[
        contains_mask & ~starts_with_mask
    ]

    result_df = pd.concat(
        [
            starts_with_df,
            contains_df,
        ],
        axis=0,
        ignore_index=True,
    )

    return result_df.head(limit)