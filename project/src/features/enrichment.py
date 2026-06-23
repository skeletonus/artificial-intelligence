from pathlib import Path

import numpy as np
import pandas as pd


def build_movie_genres(movies_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Создаёт one-hot признаки жанров для каждого фильма из movies.csv.
    Жанры можно использовать в offline-обучении, потому что это метаданные фильма, а не информация из будущих оценок.
    Параметры:
        movies_df - DataFrame с фильмами из movies.csv.
    Возвращает:
        Кортеж из DataFrame с movieId и жанрами, а также списка жанровых колонок.
    """
    genres_df = (
        movies_df["genres"]
        .str.get_dummies(sep="|")
        .drop(columns=["(no genres listed)"], errors="ignore")
        .astype("int8")
    )

    genre_columns = genres_df.columns.tolist()

    movie_genres_df = pd.concat(
        [
            movies_df[["movieId"]].reset_index(drop=True),
            genres_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return movie_genres_df, genre_columns


def add_movie_genres(
    ratings_df: pd.DataFrame,
    movie_genres_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет к рейтингам one-hot признаки жанров фильма.
    Используется перед построением исторических и offline-признаков.
    Параметры:
        ratings_df - DataFrame с рейтингами пользователей.
        movie_genres_df - DataFrame с movieId и жанровыми колонками.
    Возвращает:
        DataFrame с рейтингами и жанрами фильмов.
    """
    return ratings_df.merge(
        movie_genres_df,
        on="movieId",
        how="left",
    )


def historical_data_enrichment(
    df: pd.DataFrame,
    genres: list[str],
    min_ratings: int = 5,
    default_rating: float = 3.5,
) -> pd.DataFrame:
    """
    Добавляет исторические признаки без leakage внутри train.
    Для каждой строки используются только предыдущие строки по времени, а фильмам с малым числом прошлых оценок ставится fallback по жанрам.
    Параметры:
        df - DataFrame с рейтингами и жанрами фильмов.
        genres - Список жанровых колонок.
        min_ratings - Минимальное количество прошлых оценок фильма, чтобы использовать его среднюю оценку.
        default_rating - Значение для заполнения, если нет никакой прошлой информации.
    Возвращает:
        DataFrame с признаками movie_avg_rating, user_avg_rating и user_avg_<genre>.
    """
    out = df.sort_values("timestamp").copy()

    out["_global_sum_before"] = out["rating"].cumsum() - out["rating"]
    out["_global_count_before"] = np.arange(len(out))

    out["_global_avg_before"] = (
        out["_global_sum_before"] / out["_global_count_before"]
    )
    out["_global_avg_before"] = out["_global_avg_before"].fillna(default_rating)

    out["_movie_sum_before"] = (
        out.groupby("movieId")["rating"].cumsum() - out["rating"]
    )
    out["_movie_count_before"] = out.groupby("movieId").cumcount()

    out["_movie_avg_before_raw"] = (
        out["_movie_sum_before"] / out["_movie_count_before"]
    )

    genre_avg_before_cols = []

    for genre in genres:
        sum_col = f"_genre_{genre}_sum_before"
        count_col = f"_genre_{genre}_count_before"
        avg_col = f"_genre_{genre}_avg_before"

        genre_rating = out["rating"] * out[genre]

        out[sum_col] = genre_rating.cumsum() - genre_rating
        out[count_col] = out[genre].cumsum() - out[genre]

        out[avg_col] = out[sum_col] / out[count_col]
        out[avg_col] = out[avg_col].fillna(out["_global_avg_before"])

        genre_avg_before_cols.append(avg_col)

    out["movie_avg_rating"] = np.where(
        out["_movie_count_before"] >= min_ratings,
        out["_movie_avg_before_raw"],
        np.nan,
    )

    def movie_genre_fallback(row: pd.Series) -> float:
        movie_genres = [genre for genre in genres if row[genre] == 1]

        if len(movie_genres) == 0:
            return row["_global_avg_before"]

        values = []

        for genre in movie_genres:
            value = row[f"_genre_{genre}_avg_before"]

            if not pd.isna(value):
                values.append(value)

        if len(values) > 0:
            return float(np.mean(values))

        return row["_global_avg_before"]

    missing_movie_avg = out["movie_avg_rating"].isna()

    out.loc[missing_movie_avg, "movie_avg_rating"] = (
        out.loc[missing_movie_avg].apply(movie_genre_fallback, axis=1)
    )

    out["movie_avg_rating"] = out["movie_avg_rating"].fillna(
        out["_global_avg_before"]
    )

    out["_user_sum_before"] = (
        out.groupby("userId")["rating"].cumsum() - out["rating"]
    )
    out["_user_count_before"] = out.groupby("userId").cumcount()

    out["user_avg_rating"] = out["_user_sum_before"] / out["_user_count_before"]
    out["user_avg_rating"] = out["user_avg_rating"].fillna(
        out["_global_avg_before"]
    )

    for genre in genres:
        user_genre_sum_col = f"_user_{genre}_sum_before"
        user_genre_count_col = f"_user_{genre}_count_before"
        user_genre_avg_col = f"user_avg_{genre}"

        rating_for_genre = out["rating"] * out[genre]

        out[user_genre_sum_col] = (
            rating_for_genre.groupby(out["userId"]).cumsum() - rating_for_genre
        )
        out[user_genre_count_col] = (
            out[genre].groupby(out["userId"]).cumsum() - out[genre]
        )

        out[user_genre_avg_col] = (
            out[user_genre_sum_col] / out[user_genre_count_col]
        )

        out[user_genre_avg_col] = out[user_genre_avg_col].fillna(
            out[f"_genre_{genre}_avg_before"]
        )
        out[user_genre_avg_col] = out[user_genre_avg_col].fillna(
            out["user_avg_rating"]
        )
        out[user_genre_avg_col] = out[user_genre_avg_col].fillna(
            out["_global_avg_before"]
        )

    columns_to_drop = [
        column
        for column in out.columns
        if column.startswith("_")
    ]

    out = out.drop(columns=columns_to_drop)

    return out

def add_offline_features_from_train(
    target_df: pd.DataFrame,
    train_df: pd.DataFrame,
    genres: list[str],
    min_ratings: int = 5,
    default_rating: float = 3.5,
) -> pd.DataFrame:
    """
    Добавляет признаки в target_df, считая все агрегаты только по train_df.
    Используется для validation и test, чтобы не брать информацию из будущих оценок target_df.
    Параметры:
        target_df - DataFrame, который нужно обогатить.
        train_df - Исторический DataFrame, по которому считаются агрегаты.
        genres - Список жанровых колонок.
        min_ratings - Минимальное количество оценок фильма, чтобы использовать его среднюю оценку.
        default_rating - Значение для заполнения, если нет информации в train_df.
    Возвращает:
        DataFrame с offline-признаками без leakage.
    """
    out = target_df.copy()

    global_mean = train_df["rating"].mean()

    if pd.isna(global_mean):
        global_mean = default_rating

    movie_stats = (
        train_df.groupby("movieId")["rating"]
        .agg(movie_avg_train="mean", movie_count_train="count")
        .reset_index()
    )

    out = out.merge(
        movie_stats,
        on="movieId",
        how="left",
    )

    genre_means = {}

    for genre in genres:
        mask = train_df[genre] == 1

        if mask.sum() > 0:
            genre_means[genre] = train_df.loc[mask, "rating"].mean()
        else:
            genre_means[genre] = global_mean

    genre_matrix = out[genres].to_numpy()
    genre_values = np.array([genre_means[genre] for genre in genres])

    genre_sum = genre_matrix @ genre_values
    genre_count = genre_matrix.sum(axis=1)

    movie_genre_fallback = np.where(
        genre_count > 0,
        genre_sum / genre_count,
        global_mean,
    )

    out["movie_avg_rating"] = np.where(
        out["movie_count_train"] >= min_ratings,
        out["movie_avg_train"],
        movie_genre_fallback,
    )

    out["movie_avg_rating"] = (
        pd.Series(out["movie_avg_rating"], index=out.index)
        .fillna(global_mean)
        .values
    )

    out = out.drop(columns=["movie_avg_train", "movie_count_train"])

    user_avg = train_df.groupby("userId")["rating"].mean()

    out["user_avg_rating"] = out["userId"].map(user_avg)
    out["user_avg_rating"] = out["user_avg_rating"].fillna(global_mean)

    for genre in genres:
        user_genre_avg = (
            train_df[train_df[genre] == 1]
            .groupby("userId")["rating"]
            .mean()
        )

        genre_mean = genre_means[genre]

        out[f"user_avg_{genre}"] = out["userId"].map(user_genre_avg)
        out[f"user_avg_{genre}"] = out[f"user_avg_{genre}"].fillna(genre_mean)
        out[f"user_avg_{genre}"] = out[f"user_avg_{genre}"].fillna(global_mean)

    return out

def build_enriched_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings: int = 5,
    default_rating: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Создаёт enriched train, validation и test для CatBoost по логике из ноутбуков.
    Train обогащается исторически, validation считается по train, test считается по train + validation.
    Параметры:
        train_df - Train-часть рейтингов.
        val_df - Validation-часть рейтингов.
        test_df - Test-часть рейтингов.
        movies_df - DataFrame с фильмами из movies.csv.
        min_ratings - Минимальное количество оценок фильма для использования movie average.
        default_rating - Значение для заполнения при отсутствии истории.
    Возвращает:
        Кортеж из enriched train, validation, test.
    """
    movie_genres_df, genres = build_movie_genres(movies_df)

    train_with_genres = add_movie_genres(train_df, movie_genres_df)
    val_with_genres = add_movie_genres(val_df, movie_genres_df)
    test_with_genres = add_movie_genres(test_df, movie_genres_df)

    train_enriched = historical_data_enrichment(
        df=train_with_genres,
        genres=genres,
        min_ratings=min_ratings,
        default_rating=default_rating,
    )

    val_enriched = add_offline_features_from_train(
        target_df=val_with_genres,
        train_df=train_with_genres,
        genres=genres,
        min_ratings=min_ratings,
        default_rating=default_rating,
    )

    train_val_history = pd.concat(
        [train_with_genres, val_with_genres],
        axis=0,
        ignore_index=True,
    )

    test_enriched = add_offline_features_from_train(
        target_df=test_with_genres,
        train_df=train_val_history,
        genres=genres,
        min_ratings=min_ratings,
        default_rating=default_rating,
    )

    return train_enriched, val_enriched, test_enriched

def save_enriched_splits(
    train_enriched: pd.DataFrame,
    val_enriched: pd.DataFrame,
    test_enriched: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Сохраняет enriched train, validation и test в CSV-файлы.
    Параметры:
        train_enriched - Enriched train DataFrame.
        val_enriched - Enriched validation DataFrame.
        test_enriched - Enriched test DataFrame.
        output_dir - Папка для сохранения train.csv, val.csv и test.csv.
    Возвращает:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_enriched.to_csv(output_dir / "train.csv", index=False)
    val_enriched.to_csv(output_dir / "val.csv", index=False)
    test_enriched.to_csv(output_dir / "test.csv", index=False)

    print(f"Enriched train saved to: {output_dir / 'train.csv'}")
    print(f"Enriched val saved to:   {output_dir / 'val.csv'}")
    print(f"Enriched test saved to:  {output_dir / 'test.csv'}")