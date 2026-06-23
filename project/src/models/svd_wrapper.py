import pandas as pd
import gc

from sklearn.base import BaseEstimator, RegressorMixin
from surprise import Dataset, Reader, SVD


class SurpriseSVDRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-подобная обёртка над Surprise SVD.
    Позволяет обучать SVD через fit(X, y) и получать предсказания через predict(X).
    Параметры:
        n_factors - Размерность скрытых факторов.
        n_epochs - Количество эпох обучения.
        lr_all - Learning rate для всех параметров SVD.
        reg_all - Регуляризация для всех параметров SVD.
        random_state - Seed для воспроизводимости.
        user_col - Название колонки с пользователем.
        item_col - Название колонки с фильмом.
        rating_scale - Минимальный и максимальный рейтинг.
    Возвращает:
        Объект SurpriseSVDRegressor.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_scale: tuple[float, float] = (0.5, 5.0),
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.user_col = user_col
        self.item_col = item_col
        self.rating_scale = rating_scale
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Обучает Surprise SVD на парах userId, movieId и рейтингах.
        Внутри преобразует pandas DataFrame в формат Surprise trainset.
        Параметры:
            X - DataFrame с колонками userId и movieId.
            y - Series с рейтингами.
        Возвращает:
            self.
        """
        train_df = X.copy(deep=False)
        train_df["rating"] = y.to_numpy(copy=False)

        reader = Reader(rating_scale=self.rating_scale)

        dataset = Dataset.load_from_df(
            train_df,
            reader,
        )

        del train_df
        gc.collect()

        trainset = dataset.build_full_trainset()

        del dataset
        gc.collect()

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )

        self.model.fit(trainset)

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> list[float]:
        """
        Предсказывает рейтинги для пар userId и movieId.
        Использует обученную Surprise SVD-модель.
        Параметры:
            X - DataFrame с колонками userId и movieId.
        Возвращает:
            Список предсказанных рейтингов.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        predictions = [
            self.model.predict(
                uid=row.userId,
                iid=row.movieId,
            ).est
            for row in X[[self.user_col, self.item_col]].itertuples(index=False)
        ]

        return predictions
    

def build_global_svd_artifact(
    model: SurpriseSVDRegressor,
) -> dict:
    """
    Собирает компактный артефакт глобальной SVD-модели.
    Сохраняет глобальное среднее, смещения и факторы фильмов.
    Параметры:
        model - Обученная SurpriseSVDRegressor-модель.
    Возвращает:
        Словарь с параметрами глобальной SVD-модели.
    """
    if model.model is None:
        raise ValueError("SVD model is not fitted.")

    algorithm = model.model
    trainset = algorithm.trainset

    movie_id_to_index = {}

    for inner_movie_id in range(trainset.n_items):
        raw_movie_id = trainset.to_raw_iid(inner_movie_id)

        movie_id_to_index[int(raw_movie_id)] = int(inner_movie_id)

    return {
        "global_mean": float(trainset.global_mean),
        "movie_id_to_index": movie_id_to_index,
        "movie_biases": algorithm.bi.astype("float32"),
        "movie_factors": algorithm.qi.astype("float32"),
        "rating_scale": tuple(model.rating_scale),
    }