from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_split(
    split_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает train, validation и test из указанной папки.
    Используется для чтения raw или enriched split перед обучением модели.
    Параметры:
        split_dir - Папка, где лежат train.csv, val.csv и test.csv.
    Возвращает:
        Кортеж из трёх DataFrame: train_df, val_df и test_df.
    """
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    return train_df, val_df, test_df


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rating",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Делит DataFrame на признаки и целевую переменную.
    Используется перед обучением и оценкой моделей.
    Параметры:
        df - DataFrame с признаками и целевой переменной.
        feature_columns - Список колонок, которые используются как признаки.
        target_column - Название колонки с целевой переменной.
    Возвращает:
        Кортеж из X и y.
    """
    X = df[feature_columns]
    y = df[target_column]

    return X, y


def regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Считает RMSE и MAE для предсказаний модели.
    Используется для одинаковой оценки CatBoost и SVD.
    Параметры:
        y_true - Настоящие значения рейтингов.
        y_pred - Предсказанные значения рейтингов.
    Возвращает:
        Словарь с метриками rmse и mae.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
    }


def evaluate_regression_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """
    Оценивает модель с методом predict на переданных данных.
    Подходит для моделей со sklearn-подобным интерфейсом.
    Параметры:
        model - Обученная модель с методом predict.
        X - DataFrame с признаками.
        y - Настоящие значения рейтингов.
    Возвращает:
        Словарь с метриками rmse и mae.
    """
    predictions = model.predict(X)

    return regression_metrics(
        y_true=y,
        y_pred=predictions,
    )


def save_joblib(
    model,
    path: Path,
) -> None:
    """
    Сохраняет модель через joblib.
    Используется для моделей, которые удобно сериализовать в pickle/joblib формат.
    Параметры:
        model - Обученная модель.
        path - Путь для сохранения модели.
    Возвращает:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)