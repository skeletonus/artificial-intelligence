import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.utils.config import load_config, resolve_project_path
from src.utils.io import load_json


config = load_config()


def test_setup_artifacts_exist():
    """
    Проверяет наличие основных файлов, создаваемых setup-скриптом.
    Параметры:
        Нет.
    Возвращает:
        None.
    """
    required_paths = [
        resolve_project_path(config["paths"]["movies_catalog_path"]),
        resolve_project_path(config["paths"]["genre_columns_path"]),
        resolve_project_path(config["model"]["catboost_model_path"]),
        resolve_project_path(config["model"]["catboost_metadata_path"]),
        resolve_project_path(config["model"]["svd_global_model_path"]),
    ]

    user_data_dir = resolve_project_path(config["paths"]["user_data_dir"])
    required_paths.extend([
        user_data_dir / "user_questionnaire.json",
        user_data_dir / "user_ratings.csv",
    ])

    for path in required_paths:
        assert path.exists(), f"File was not created: {path}"
        assert path.stat().st_size > 0, f"File is empty: {path}"


def test_movies_catalog():
    """
    Проверяет структуру production-каталога фильмов.
    Параметры:
        Нет.
    Возвращает:
        None.
    """
    catalog_path = resolve_project_path(config["paths"]["movies_catalog_path"])
    genre_path = resolve_project_path(config["paths"]["genre_columns_path"])

    catalog_df = pd.read_csv(catalog_path)
    genre_columns = load_json(genre_path)

    assert len(catalog_df) > 0
    assert {"movieId", "title", "movie_avg_rating"}.issubset(catalog_df.columns)
    assert set(genre_columns).issubset(catalog_df.columns)
    assert catalog_df["movieId"].is_unique
    assert catalog_df["movieId"].notna().all()
    assert catalog_df["title"].notna().all()


def test_catboost_artifacts():
    """
    Проверяет загрузку CatBoost-модели и связанных метаданных.
    Параметры:
        Нет.
    Возвращает:
        None.
    """
    model_path = resolve_project_path(config["model"]["catboost_model_path"])
    metadata_path = resolve_project_path(config["model"]["catboost_metadata_path"])

    model = CatBoostRegressor()
    model.load_model(str(model_path))

    metadata = load_json(metadata_path)

    assert model.tree_count_ > 0
    assert metadata["model_name"] == "CatBoostRegressor"
    assert metadata["target_column"] == "rating"
    assert len(metadata["feature_columns"]) > 0
    assert "validation" in metadata["metrics"]
    assert "test" in metadata["metrics"]


def test_svd_artifact():
    """
    Проверяет структуру глобального SVD-артефакта.
    Параметры:
        Нет.
    Возвращает:
        None.
    """
    model_path = resolve_project_path(config["model"]["svd_global_model_path"])
    artifact = joblib.load(model_path)

    required_keys = {
        "global_mean",
        "movie_id_to_index",
        "movie_biases",
        "movie_factors",
        "rating_scale",
    }

    assert required_keys.issubset(artifact)

    movie_biases = np.asarray(artifact["movie_biases"])
    movie_factors = np.asarray(artifact["movie_factors"])
    movie_id_to_index = artifact["movie_id_to_index"]

    assert movie_factors.ndim == 2
    assert len(movie_biases) == movie_factors.shape[0]
    assert len(movie_id_to_index) == movie_factors.shape[0]
    assert np.isfinite(movie_biases).all()
    assert np.isfinite(movie_factors).all()
    assert np.isfinite(float(artifact["global_mean"]))


def test_user_storage():
    """
    Проверяет стартовые пользовательские файлы.
    Параметры:
        Нет.
    Возвращает:
        None.
    """
    user_data_dir = resolve_project_path(config["paths"]["user_data_dir"])
    genre_columns = load_json(
        resolve_project_path(config["paths"]["genre_columns_path"])
    )

    questionnaire = load_json(user_data_dir / "user_questionnaire.json")
    ratings_df = pd.read_csv(user_data_dir / "user_ratings.csv")

    assert set(questionnaire) == set(genre_columns)
    assert all(isinstance(value, (int, float)) for value in questionnaire.values())
    assert list(ratings_df.columns) == ["movieId", "rating"]
