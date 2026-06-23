from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.recommenders.common import load_movies_catalog
from src.users.storage import initialize_user_storage
from src.utils.config import resolve_project_path
from src.utils.io import load_json


@pytest.fixture(scope="session")
def project_data():
    """
    Загружает каталог, жанры и конфигурацию для API-тестов.
    Параметры:
        Нет.
    Возвращает:
        Конфигурацию, каталог фильмов и список жанров.
    """
    config = app_module.config

    catalog_df = load_movies_catalog(
        resolve_project_path(config["paths"]["movies_catalog_path"])
    )

    genre_columns = load_json(
        resolve_project_path(config["paths"]["genre_columns_path"])
    )

    return config, catalog_df, genre_columns


@pytest.fixture
def api_client(tmp_path: Path, project_data):
    """
    Создаёт API-клиент с отдельной временной папкой пользователя.
    Реальные модели и production-каталог остаются без изменений.
    Параметры:
        tmp_path - Временная папка pytest.
        project_data - Конфигурация, каталог и список жанров.
    Возвращает:
        TestClient, каталог и конфигурацию проекта.
    """
    config, catalog_df, genre_columns = project_data
    original_user_data_dir = config["paths"]["user_data_dir"]
    test_user_data_dir = tmp_path / "user_data"

    initialize_user_storage(
        user_data_dir=test_user_data_dir,
        movies_catalog_df=catalog_df,
        genre_columns=genre_columns,
        default_rating=config["enrichment"]["default_rating"],
    )

    config["paths"]["user_data_dir"] = str(test_user_data_dir)

    with TestClient(app_module.app) as client:
        yield client, catalog_df, config

    config["paths"]["user_data_dir"] = original_user_data_dir


def assert_recommendation_structure(recommendations: list[dict]) -> None:
    """
    Проверяет структуру списка рекомендаций.
    Параметры:
        recommendations - Список рекомендаций API.
    Возвращает:
        None.
    """
    assert len(recommendations) > 0

    for recommendation in recommendations:
        assert {"movieId", "title", "score"} == set(recommendation)
        assert isinstance(recommendation["movieId"], int)
        assert isinstance(recommendation["title"], str)
        assert isinstance(recommendation["score"], (int, float))


def test_health(api_client):
    client, _, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_questionnaire(api_client):
    client, _, config = api_client

    response = client.get("/questionnaire")

    assert response.status_code == 200

    questionnaire = response.json()
    genre_columns = load_json(
        resolve_project_path(config["paths"]["genre_columns_path"])
    )

    assert set(questionnaire) == set(genre_columns)
    assert all(isinstance(value, (int, float)) for value in questionnaire.values())


def test_movie_search(api_client):
    client, catalog_df, _ = api_client

    selected_movie = catalog_df.iloc[0]
    query = str(selected_movie["title"])[:5]

    response = client.get("/movies", params={"query": query, "limit": 10})

    assert response.status_code == 200

    movies = response.json()

    assert 1 <= len(movies) <= 10
    assert all({"movieId", "title"} == set(movie) for movie in movies)
    assert any(movie["movieId"] == int(selected_movie["movieId"]) for movie in movies)


def test_empty_movie_search(api_client):
    client, _, _ = api_client

    response = client.get("/movies", params={"query": ""})

    assert response.status_code == 400


def test_save_and_get_ratings(api_client):
    client, catalog_df, _ = api_client

    selected_movies = catalog_df.head(2)
    ratings = {
        str(int(selected_movies.iloc[0]["movieId"])): 4.5,
        str(int(selected_movies.iloc[1]["movieId"])): 3.5,
    }

    save_response = client.post("/ratings", json=ratings)

    assert save_response.status_code == 200
    assert isinstance(save_response.json()["message"], str)

    get_response = client.get("/ratings")

    assert get_response.status_code == 200

    saved_ratings = get_response.json()
    saved_by_id = {item["movieId"]: item for item in saved_ratings}

    assert len(saved_ratings) == 2

    for movie_id, rating in ratings.items():
        saved_item = saved_by_id[int(movie_id)]

        assert saved_item["rating"] == rating
        assert isinstance(saved_item["title"], str)
        assert saved_item["title"]


def test_rating_is_updated(api_client):
    client, catalog_df, _ = api_client

    movie_id = int(catalog_df.iloc[0]["movieId"])

    assert client.post("/ratings", json={str(movie_id): 2.0}).status_code == 200
    assert client.post("/ratings", json={str(movie_id): 5.0}).status_code == 200

    response = client.get("/ratings")

    assert response.status_code == 200

    ratings = response.json()

    assert len(ratings) == 1
    assert ratings[0]["movieId"] == movie_id
    assert ratings[0]["rating"] == 5.0


def test_unknown_movie_rating(api_client):
    client, catalog_df, _ = api_client

    known_movie_ids = set(catalog_df["movieId"].astype(int))
    unknown_movie_id = max(known_movie_ids) + 1

    while unknown_movie_id in known_movie_ids:
        unknown_movie_id += 1

    response = client.post("/ratings", json={str(unknown_movie_id): 4.0})

    assert response.status_code == 400


def test_cold_start_recommendations(api_client):
    client, _, _ = api_client

    questionnaire_response = client.get("/questionnaire")
    questionnaire = questionnaire_response.json()

    response = client.post(
        "/cold-start/recommendations",
        json={
            "questionnaire": questionnaire,
            "top_n": 3,
            "min_genre_rating": min(questionnaire.values()),
        },
    )

    assert response.status_code == 200

    recommendations = response.json()

    assert len(recommendations) == 3
    assert len({item["movieId"] for item in recommendations}) == 3
    assert_recommendation_structure(recommendations)


def test_invalid_cold_start_top_n(api_client):
    client, _, _ = api_client

    response = client.post(
        "/cold-start/recommendations",
        json={"top_n": 0},
    )

    assert response.status_code == 422


def test_warm_start_requires_ratings(api_client):
    client, _, _ = api_client

    response = client.post(
        "/warm-start/recommendations",
        json={"top_n": 3},
    )

    assert response.status_code == 200
    assert isinstance(response.json()["result"], str)


def test_warm_start_recommendations(api_client):
    client, catalog_df, config = api_client

    threshold = int(config["warm_start"]["threshold"])
    selected_movie_ids = catalog_df["movieId"].head(threshold).astype(int).tolist()

    min_rating = float(config["rating"]["min_rating"])
    max_rating = float(config["rating"]["max_rating"])
    rating = (min_rating + max_rating) / 2

    ratings = {str(movie_id): rating for movie_id in selected_movie_ids}

    save_response = client.post("/ratings", json=ratings)

    assert save_response.status_code == 200

    response = client.post(
        "/warm-start/recommendations",
        json={"top_n": 3},
    )

    assert response.status_code == 200

    recommendations = response.json()["result"]

    assert isinstance(recommendations, list)
    assert len(recommendations) == 3
    assert not set(selected_movie_ids) & {
        item["movieId"] for item in recommendations
    }

    assert_recommendation_structure(recommendations)
