from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from src.models.base import split_features_target


def get_catboost_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Формирует список признаков для обучения CatBoost.
    Исключает целевую переменную, время и id-колонки.
    Параметры:
        df - Enriched DataFrame с признаками и целевой переменной.
    Возвращает:
        Список названий колонок-признаков.
    """
    excluded_columns = [
        "rating",
        "timestamp",
        "userId",
        "movieId",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return feature_columns


def build_catboost_model(params: dict) -> CatBoostRegressor:
    """
    Создаёт CatBoostRegressor с параметрами из конфига.
    Используется перед обучением модели.
    Параметры:
        params - Словарь параметров CatBoost.
    Возвращает:
        Объект CatBoostRegressor.
    """
    params = params.copy()
    params["allow_writing_files"] = False
    
    return CatBoostRegressor(**params)


def fit_catboost_model(
    model: CatBoostRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
) -> CatBoostRegressor:
    """
    Обучает CatBoostRegressor на train-данных.
    Validation-данные используются для контроля качества во время обучения.
    Параметры:
        model - CatBoostRegressor до обучения.
        train_df - Enriched train DataFrame.
        val_df - Enriched validation DataFrame.
        feature_columns - Список колонок-признаков.
    Возвращает:
        Обученный CatBoostRegressor.
    """
    X_train, y_train = split_features_target(
        df=train_df,
        feature_columns=feature_columns,
    )

    X_val, y_val = split_features_target(
        df=val_df,
        feature_columns=feature_columns,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    return model


def save_catboost_model(
    model: CatBoostRegressor,
    model_path: Path,
) -> None:
    """
    Сохраняет обученную CatBoost-модель в файл.
    Используется для дальнейшего inference.
    Параметры:
        model - Обученная CatBoost-модель.
        model_path - Путь для сохранения модели.
    Возвращает:
        None.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(model_path)