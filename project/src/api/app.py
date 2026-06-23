from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import (
    ColdStartRequest,
    MessageResponse,
    MovieSearchRequest,
    WarmStartRequest,
)

from src.recommenders.service import (
    get_user_ratings_with_titles,
    run_cold_start_recommendations,
    run_movie_search,
    run_warm_start_recommendations,
    save_ratings_window,
)
from src.users.storage import load_user_questionnaire
from src.utils.config import load_config, resolve_project_path

config = load_config()

app = FastAPI(
    title="Movie Recommendation Service",
    description="FastAPI service for cold-start and warm-start movie recommendations.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """
    Проверяет, что сервис запущен.
    Используется для быстрой проверки FastAPI приложения.
    Параметры:
        Нет.
    Возвращает:
        Статус сервиса.
    """
    return {
        "status": "ok",
    }


@app.get("/questionnaire")
def get_user_questionnaire() -> dict[str, float]:
    """
    Возвращает текущую сохранённую анкету пользователя.
    Параметры:
        Нет.
    Возвращает:
        Словарь жанр-рейтинг.
    """
    try:
        return load_user_questionnaire(
            user_data_dir=resolve_project_path(
                config["paths"]["user_data_dir"]
            )
        )

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/cold-start/recommendations")
def cold_start_recommendations(request: ColdStartRequest) -> list[dict]:
    """
    Сохраняет анкету пользователя и возвращает cold-start рекомендации.
    Использует CatBoost-модель.
    Параметры:
        request - Запрос с анкетой жанров и количеством рекомендаций.
    Возвращает:
        Список рекомендаций.
    """
    try:
        return run_cold_start_recommendations(
            config=config,
            questionnaire=request.questionnaire,
            top_n=request.top_n,
            min_genre_rating=request.min_genre_rating,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/movies")
def get_movies(
    request: MovieSearchRequest = Depends(),
) -> list[dict]:
    """
    Возвращает фильмы из каталога по части названия.
    Сначала выводит фильмы, название которых начинается со строки поиска.
    Параметры:
        request - Параметры поиска фильма.
    Возвращает:
        Список найденных фильмов.
    """
    try:
        return run_movie_search(
            config=config,
            query=request.query,
            limit=request.limit,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    

@app.get("/ratings")
def get_ratings() -> list[dict]:
    """
    Возвращает сохранённые пользовательские оценки
    вместе с названиями фильмов.
    """
    try:
        return get_user_ratings_with_titles(config=config)

    except FileNotFoundError as error:
        raise HTTPException(status_code=404,detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/ratings", response_model=MessageResponse)
def save_ratings(ratings: dict[int, float]) -> MessageResponse:
    """
    Валидирует и сохраняет пользовательские оценки фильмов.
    Пользователь передаёт словарь movieId-рейтинг.
    Параметры:
        ratings - Словарь movieId-рейтинг.
    Возвращает:
        Сообщение о результате сохранения.
    """
    try:
        message = save_ratings_window(
            config=config,
            ratings=ratings,
        )

        return MessageResponse(message=message)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/warm-start/recommendations")
def warm_start_recommendations(request: WarmStartRequest) -> dict:
    """
    Возвращает warm-start рекомендации или сообщение о нехватке оценок.
    При необходимости обучает или переобучает SVD по заданному правилу.
    Параметры:
        request - Запрос с количеством рекомендаций.
    Возвращает:
        Словарь с результатом warm-start окна.
    """
    try:
        result = run_warm_start_recommendations(
            config=config,
            top_n=request.top_n,
        )

        return {
            "result": result,
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))