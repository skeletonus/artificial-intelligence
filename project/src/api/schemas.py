from pydantic import BaseModel, Field

from src.recommenders.common import load_movies_catalog
from src.users.storage import build_default_questionnaire
from src.utils.config import load_config, resolve_project_path
from src.utils.io import load_json


DEFAULT_TOP_N = 10

config = load_config()

movies_catalog_df = load_movies_catalog(
    resolve_project_path(
        config["paths"]["movies_catalog_path"]
    )
)

genre_columns = load_json(
    resolve_project_path(
        config["paths"]["genre_columns_path"]
    )
)

DEFAULT_QUESTIONNAIRE = build_default_questionnaire(
    movies_catalog_df=movies_catalog_df,
    genre_columns=genre_columns,
    default_rating=config["enrichment"]["default_rating"],
)

MIN_RATING = float(config["rating"]["min_rating"])
MAX_RATING = float(config["rating"]["max_rating"])
DEFAULT_MIN_GENRE_RATING = float(config["cold_start"]["default_min_genre_rating"])

class ColdStartRequest(BaseModel):
    """
    Описывает запрос для cold-start рекомендаций.
    Пользователь передаёт оценки жанров, количество рекомендаций
    и минимальную допустимую оценку жанра.
    """
    questionnaire: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_QUESTIONNAIRE.copy(),
        description="Оценки пользователя для всех жанров.",
    )
    top_n: int = Field(
        default=DEFAULT_TOP_N,
        ge=1,
        le=100,
        description="Сколько фильмов вывести.",
    )
    min_genre_rating: float = Field(
        default=DEFAULT_MIN_GENRE_RATING,
        ge=MIN_RATING,
        le=MAX_RATING,
        description="Минимальная допустимая оценка жанра.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "questionnaire": DEFAULT_QUESTIONNAIRE,
                "top_n": DEFAULT_TOP_N,
                "min_genre_rating": 2.5,
            }
        }
    }


class WarmStartRequest(BaseModel):
    """
    Описывает запрос для warm-start рекомендаций.
    Пользователь передаёт только количество рекомендаций.
    """
    top_n: int = Field(default=DEFAULT_TOP_N, ge=1, le=100)


class MessageResponse(BaseModel):
    """
    Описывает простой текстовый ответ сервиса.
    """
    message: str


class MovieSearchRequest(BaseModel):
    """
    Описывает параметры поиска фильмов по названию.
    Пользователь передаёт часть названия и количество результатов.
    """
    query: str = Field(
        default="",
        description="Часть названия фильма.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Максимальное количество найденных фильмов.",
    )