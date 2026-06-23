import argparse
import shutil
import gc
import pandas as pd

from src.data.download import prepare_movielens_dataset
from src.data.load import load_movielens
from src.data.split import make_raw_time_split
from src.features.enrichment import build_enriched_splits, save_enriched_splits
from src.features.movie_catalog import build_movies_catalog, save_movies_catalog
from src.utils.config import load_config, resolve_project_path

from src.models.base import (
    evaluate_regression_model,
    load_split,
    save_joblib,
    split_features_target,
)

from src.models.catboost_model import (
    build_catboost_model,
    fit_catboost_model,
    get_catboost_feature_columns,
    save_catboost_model,
)
from src.models.svd_wrapper import (
    SurpriseSVDRegressor,
    build_global_svd_artifact,
)

from src.utils.io import save_json, load_json

from src.users.storage import initialize_user_storage


def parse_args() -> argparse.Namespace:
    """
    Считывает аргументы командной строки для setup-скрипта.
    Позволяет передать путь к конфигу и включить пересоздание файлов.
    Параметры:
        Нет.
    Возвращает:
        Объект с аргументами командной строки.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate downloaded and processed files.",
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--cleanup-data",
        action="store_true",
        help="Delete training datasets after successful setup.",
    )

    return parser.parse_args()


def prepare_small_dataset(config: dict, force: bool) -> None:
    """
    Готовит small-датасет для обучения CatBoost.
    Скачивает данные, делает raw split и enriched split.
    Параметры:
        config - Словарь с настройками проекта.
        force - Нужно ли пересоздать файлы, если они уже существуют.
    Возвращает:
        None.
    """
    dataset_config = config["data"]["datasets"]["small"]

    raw_dir = prepare_movielens_dataset(
        config=config,
        dataset_key="small",
        force=force,
    )

    ratings_df, movies_df = load_movielens(raw_dir)

    train_df, val_df, test_df = make_raw_time_split(
        ratings_df=ratings_df,
        output_dir=resolve_project_path(dataset_config["raw_version_dir"]),
        test_size=config["split"]["test_size"],
        val_size_from_train=config["split"]["val_size_from_train"],
    )

    train_enriched, val_enriched, test_enriched = build_enriched_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        movies_df=movies_df,
        min_ratings=config["enrichment"]["min_ratings"],
        default_rating=config["enrichment"]["default_rating"],
    )

    save_enriched_splits(
        train_enriched=train_enriched,
        val_enriched=val_enriched,
        test_enriched=test_enriched,
        output_dir=resolve_project_path(dataset_config["enriched_version_dir"]),
    )


def prepare_full_dataset(config: dict, force: bool) -> None:
    """
    Готовит full-датасет для raw collaborative-данных и production-каталога.
    Скачивает данные, делает raw split и строит справочник фильмов.
    Параметры:
        config - Словарь с настройками проекта.
        force - Нужно ли пересоздать файлы, если они уже существуют.
    Возвращает:
        None.
    """
    dataset_config = config["data"]["datasets"]["full"]

    raw_dir = prepare_movielens_dataset(
        config=config,
        dataset_key="full",
        force=force,
    )

    ratings_df, movies_df = load_movielens(raw_dir)

    make_raw_time_split(
        ratings_df=ratings_df,
        output_dir=resolve_project_path(dataset_config["raw_version_dir"]),
        test_size=config["split"]["test_size"],
        val_size_from_train=config["split"]["val_size_from_train"],
    )

    movies_catalog_df, genre_columns = build_movies_catalog(
        movies_df=movies_df,
        ratings_df=ratings_df,
        lambda_=config["movie_catalog"]["regularization_lambda"],
    )

    save_movies_catalog(
        movies_catalog_df=movies_catalog_df,
        genre_columns=genre_columns,
        movies_catalog_path=resolve_project_path(
            config["paths"]["movies_catalog_path"]
        ),
        genre_columns_path=resolve_project_path(
            config["paths"]["genre_columns_path"]
        ),
    )


def train_catboost_for_project(config: dict) -> None:
    """
    Обучает CatBoost на enriched small split.
    Сохраняет модель, список признаков и метрики качества.
    Параметры:
        config - Словарь с настройками проекта.
    Возвращает:
        None.
    """
    small_config = config["data"]["datasets"]["small"]
    enriched_dir = resolve_project_path(small_config["enriched_version_dir"])

    train_df, val_df, test_df = load_split(enriched_dir)

    feature_columns = get_catboost_feature_columns(train_df)

    model = build_catboost_model(config["catboost"])

    model = fit_catboost_model(
        model=model,
        train_df=train_df,
        val_df=val_df,
        feature_columns=feature_columns,
    )

    X_val, y_val = split_features_target(
        df=val_df,
        feature_columns=feature_columns,
    )

    X_test, y_test = split_features_target(
        df=test_df,
        feature_columns=feature_columns,
    )

    metrics = {
        "validation": evaluate_regression_model(
            model=model,
            X=X_val,
            y=y_val,
        ),
        "test": evaluate_regression_model(
            model=model,
            X=X_test,
            y=y_test,
        ),
    }

    save_catboost_model(
        model=model,
        model_path=resolve_project_path(config["model"]["catboost_model_path"]),
    )

    catboost_metadata = {
        "model_name": "CatBoostRegressor",
        "target_column": "rating",
        "feature_columns": feature_columns,
        "metrics": metrics,
        "params": config["catboost"],
    }

    save_json(
        data=catboost_metadata,
        path=resolve_project_path(config["model"]["catboost_metadata_path"]),
    )


def initialize_user_files_from_artifacts(
    config: dict,
) -> None:
    """
    Загружает production catalog и список жанров из сохранённых файлов.
    Создаёт стартовые пользовательские файлы в user_data.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
    Возвращает:
        None.
    """
    movies_catalog_df = pd.read_csv(
        resolve_project_path(config["paths"]["movies_catalog_path"])
    )

    genre_columns = load_json(
        resolve_project_path(config["paths"]["genre_columns_path"])
    )

    initialize_user_storage(
        user_data_dir=resolve_project_path(config["paths"]["user_data_dir"]),
        movies_catalog_df=movies_catalog_df,
        genre_columns=genre_columns,
        default_rating=config["enrichment"]["default_rating"],
    )


def thin_svd_ratings(
    ratings_df: pd.DataFrame,
    max_ratings: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Сокращает набор оценок для обучения SVD.
    Сохраняет минимум одну оценку для каждого фильма и добавляет случайную выборку.
    Параметры:
        ratings_df - Полный DataFrame с оценками.
        max_ratings - Максимальное число оценок после прореживания.
        random_state - Seed для воспроизводимой выборки.
    Возвращает:
        Прореженный DataFrame с оценками.
    """
    if len(ratings_df) <= max_ratings:
        return ratings_df

    movie_coverage_df = ratings_df.drop_duplicates(subset="movieId")

    if len(movie_coverage_df) > max_ratings:
        raise ValueError("max_train_ratings cannot be less than the number of movies.")

    sampled_df = ratings_df.sample(
        n=max_ratings,
        random_state=random_state,
    )

    result_df = pd.concat([
        movie_coverage_df,
        sampled_df,
    ])

    result_df = result_df.loc[
        ~result_df.index.duplicated()
    ]

    result_df = result_df.head(max_ratings).reset_index(drop=True)

    print(
        f"SVD ratings reduced from {len(ratings_df)} "
        f"to {len(result_df)}."
    )

    return result_df


def train_global_svd_for_project(
    config: dict
) -> None:
    """
    Обучает глобальную SVD-модель на полном наборе MovieLens.
    Сохраняет компактный артефакт с факторами и смещениями фильмов.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
    Возвращает:
        None.
    """
    full_dataset_config = config["data"]["datasets"]["full"]

    ratings_path = (
        resolve_project_path(full_dataset_config["raw_dir"])
        / "ratings.csv"
    )

    model_path = resolve_project_path(
        config["model"]["svd_global_model_path"]
    )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Full ratings file not found: {ratings_path}"
        )

    print("Loading ratings for global SVD training.")

    ratings_df = pd.read_csv(
        ratings_path,
        usecols=[
            "userId",
            "movieId",
            "rating",
        ],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
        },
    )

    ratings_df = thin_svd_ratings(
        ratings_df=ratings_df,
        max_ratings=int(config["svd"]["max_train_ratings"]),
        random_state=int(config["svd"]["random_state"]),
    )

    print(
        f"Training global SVD on {len(ratings_df)} ratings."
    )

    X_train, y_train = split_features_target(
        df=ratings_df,
        feature_columns=[
            "userId",
            "movieId",
        ],
    )

    model = SurpriseSVDRegressor(
        n_factors=config["svd"]["n_factors"],
        n_epochs=config["svd"]["n_epochs"],
        lr_all=config["svd"]["lr_all"],
        reg_all=config["svd"]["reg_all"],
        random_state=config["svd"]["random_state"],
        rating_scale=(
            config["rating"]["min_rating"],
            config["rating"]["max_rating"],
        ),
    )

    model.fit(
        X=X_train,
        y=y_train,
    )

    global_svd_artifact = build_global_svd_artifact(model)

    del model
    del X_train
    del y_train
    del ratings_df

    save_joblib(
        model=global_svd_artifact,
        path=model_path,
    )

    print(
        f"Global SVD model saved: {model_path}"
    )


def cleanup_training_data(config: dict) -> None:
    """
    Удаляет исходные и промежуточные датасеты после обучения.
    Production-каталог и артефакты моделей не затрагиваются.
    Параметры:
        config - Словарь с настройками проекта из config.yaml.
    Возвращает:
        None.
    """
    for dataset_config in config["data"]["datasets"].values():
        dataset_dir = resolve_project_path(dataset_config["path"])

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            print(f"Training dataset deleted: {dataset_dir}")


def main() -> None:
    """
    Запускает полный pipeline подготовки данных проекта.
    Готовит small-датасет для CatBoost и full-датасет для raw split и каталога фильмов.
    Параметры:
        Нет.
    Возвращает:
        None.
    """
    args = parse_args()

    config = load_config(args.config)

    prepare_small_dataset(
        config=config,
        force=args.force,
    )

    prepare_full_dataset(
        config=config,
        force=args.force,
    )

    initialize_user_files_from_artifacts(config)

    train_catboost_for_project(config)

    gc.collect()

    train_global_svd_for_project(config=config)

    if args.cleanup_data:
        cleanup_training_data(config)

    print("Project setup completed successfully.")


if __name__ == "__main__":
    main()