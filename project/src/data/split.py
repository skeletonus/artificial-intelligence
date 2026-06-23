from pathlib import Path

import pandas as pd


def train_test_split_by_time(
    df: pd.DataFrame,
    time_col: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит DataFrame на две части по времени: ранние записи идут в train, поздние — в test.
    Такой split нужен, чтобы имитировать реальную ситуацию, где модель обучается на прошлом и предсказывает будущее.
    Параметры:
        df - DataFrame, который нужно разделить.
        time_col - Название колонки со временем.
        test_size - Доля данных, которая попадёт во вторую часть.
    Возвращает:
        Кортеж из двух DataFrame: train_df и test_df.
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def make_raw_time_split(
ratings_df: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    val_size_from_train: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делит ratings_df на train, validation и test по времени и сохраняет их в CSV-файлы.
    Сначала отделяется test, потом из оставшегося train отделяется validation.
    Параметры:
        ratings_df - DataFrame с рейтингами пользователей.
        output_dir - Папка, куда нужно сохранить train.csv, val.csv и test.csv.
        test_size - Доля данных для test.
        val_size_from_train - Доля validation от оставшейся train-части.
    Возвращает:
        Кортеж из трёх DataFrame: train_df, val_df и test_df.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_val_df, test_df = train_test_split_by_time(
        df=ratings_df,
        time_col="timestamp",
        test_size=test_size,
    )

    train_df, val_df = train_test_split_by_time(
        df=train_val_df,
        time_col="timestamp",
        test_size=val_size_from_train,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Raw split saved to: {output_dir}")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    return train_df, val_df, test_df